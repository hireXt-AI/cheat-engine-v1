"""
Cheat Engine v1 — LiveKit proctor server.

The candidate's audio+video is published to the LiveKit SFU. This server joins
the interview room as a proctor, subscribes to the candidate's tracks directly
from the SFU (no browser re-upload -> zero extra upload bandwidth), and runs the
v1 detection pipeline (MediaPipe face/face_mesh + face_recognition) per frame.

Violations are buffered by `ViolationTracker` (12-occurrence trigger threshold).
When a trigger point is reached it is written to CSV, a snapshot is saved, and
an AI alert is published over the room data channel (`trigger`) so the Jarvis
agent can react naturally. Continuous proctoring state is published as
`cheating_status` (mirroring v0) so the browser overlay + agent LLM threshold
injection keep working.

Endpoints (same REST surface as v0, so the admin UI keeps working):
    POST /register-face            (multipart: image + sessionId)
    POST /api/proctor/start        (json: sessionId + livekitUrl + token)
    GET  /report/{sessionId}
    GET  /evidence
    GET  /evidence/{sessionId}/{filename}
    GET  /evidence/{sessionId}/mapping/{filename}
    GET  /live-frame/{sessionId}

Usage:
    python stream_server.py [--port 6544] [--tolerance 0.6]
"""





import wave
import argparse
import asyncio
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
import uuid
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from s3_storage import S3_BUCKET_NAME, upload_evidence

# LiveKit is optional: keep REST/register-face/report working even if the
# `livekit` package is not installed yet.
try:
    from livekit import rtc
    LIVEKIT_AVAILABLE = True
except Exception:
    rtc = None
    LIVEKIT_AVAILABLE = False

# v1 detection core — imported at module level. Verified safe: main.py only
# defines functions/classes at import time (camera/recording only starts under
# `if __name__ == "__main__"`), so there are no module-level side effects.
from main import (
    ViolationTracker,
    analyze_lighting_conditions,
    calculate_ear,
    detect_black_glasses,
    detect_iris_direction,
    load_reference_encoding,
    combine_audio_video,
    MIN_BRIGHTNESS_PCT,
    MAX_BRIGHTNESS_PCT,
)

from aiohttp import web
import socketio


# ── Server setup ─────────────────────────────────────────────────────────────
def _json_default(o):
    """json.dumps default: make numpy scalar types JSON-serializable."""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def _json_safe(data):
    """Deep-copy `data` with numpy scalars converted to native Python types."""
    return json.loads(json.dumps(data, default=_json_default))


sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')
app = web.Application()
sio.attach(app)

EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), '.evidence')
TRIGGER_DIR = os.path.join(EVIDENCE_DIR, 'trigger_points')
os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs(TRIGGER_DIR, exist_ok=True)
RECORDING_DIR = os.path.join(EVIDENCE_DIR, 'recordings')
os.makedirs(RECORDING_DIR, exist_ok=True)
RECORDING_FPS = 20.0

MATCH_TOLERANCE = 0.6  # face_recognition face_distance tolerance (lower = stricter)

# ── Engine globals ───────────────────────────────────────────────────────────
# Per-session proctor state: sessionId -> ProctorSession
proctors = {}  # sessionId -> LiveKitProctor (LiveKit transport sessions)
latest_frames = {}  # sessionId -> jpeg_bytes (most recent frame for live monitoring)
sessions = {}  # sessionId -> detection state (mirrors v0's session dict)


# ── v1 detection config (mirrors main.start_interview_recording) ─────────────
EAR_CLOSED_THRESHOLD = 0.18
NO_BLINK_WARN_SECONDS = 60.0
REQUIRED_GLASSES_FRAMES = 3
EYE_OPEN_FOR_GAZE_EAR = 0.23
IRIS_STREAK_TRIGGER = 6
TRIGGER_THRESHOLD = 3          # occurrences per violation type -> trigger point
VERIFY_EVERY_SECONDS = 30.0      # face_recognition identity check cadence
SNAPSHOT_COOLDOWN_S = 10.0       # min seconds between standard evidence snapshots

# Head-pose thresholds (mirrors main.py's start_interview_recording loop)
HEAD_YAW_THRESHOLD = 0.20
HEAD_PITCH_UP_THRESHOLD = 0.26
HEAD_PITCH_DOWN_THRESHOLD = 0.72


class ProctorSession:
    """Per-session detection state + CSV/trigger tracker."""

    def __init__(self, session_id):
        ProctorSession.LAST_SESSION = self
        self.session_id = session_id
        self.reference_encoding = None
        self.instant_violation_active = set()
        # ViolationTracker writes trigger points to CSV under .evidence
        csv_path = Path(EVIDENCE_DIR) / f"trigger_points_{session_id}.csv"
        self.tracker = ViolationTracker(
            csv_path=csv_path,
            trigger_snapshot_dir=Path(TRIGGER_DIR) / session_id,
            user_id=f"USER_{session_id}",
            threshold=TRIGGER_THRESHOLD,
        )
        # Auto-register active session for dev server
        # Auto-register active session for dev server (Fix for __main__ module)
        try:
          import sys

          main_mod = sys.modules.get("__main__")
          if hasattr(main_mod, "register_active_session"):
            main_mod.register_active_session(self)
          else:
            import dev_server

            dev_server.register_active_session(self)
        except Exception:
          pass
        # ── Recording (raw video + audio → combined compressed mp4) ───────
        rec_dir = Path(RECORDING_DIR) / session_id
        rec_dir.mkdir(parents=True, exist_ok=True)
        self.raw_video_path = rec_dir / f"{session_id}_raw.mp4"
        self.audio_path = rec_dir / f"{session_id}.wav"
        self.final_video_path = rec_dir / f"{session_id}.mp4"
        self.video_writer = None      # lazily created (frame size not known yet)
        self._wav_file = None         # lazily created (sample rate not known yet)
        self.recording_finalized = False

        # Cooldown timers (seconds)
        self.last_verify_time = 0.0
        self.last_snap_time = 0.0
        self.last_movement_snap_time = 0.0
        self.last_glasses_snap_time = 0.0
        self.last_light_snap_time = 0.0
        self.last_eye_closure_snap_time = 0.0
        self.last_no_blink_snap_time = 0.0
        self.last_trigger_snap_time = 0.0
        self.last_missing_face_snap_time = 0.0
        self.last_mismatch_snap_time = 0.0

        # Streaming state
        self.eye_closed_start_time = None
        self.last_blink_timestamp = time.time()
        self.glasses_consecutive_frames = 0
        self.iris_streak_direction = None
        self.iris_streak_count = 0
        self.identity_ok = None

        # Mirror v0 session fields for /report + event log
        self.suspicion_score = 0.0
        self.events = []
        self.face_gone_start = None
        self.gaze_away_start = None
        self.last_screenshot = 0.0
        self.last_result = None  # latest detection result (for live admin view)

    # ── Event + score helpers (mirror v0) ─────────────────────────────────
    def add_event(self, event_type, detail=""):
        self.events.append({"time": round(time.time(), 2), "type": event_type, "detail": detail})
        if len(self.events) > 200:
            self.events = self.events[-200:]

    def bump_score(self, delta):
        self.suspicion_score = min(self.suspicion_score + delta, 100)

    def write_video_frame(self, bgr, fps=RECORDING_FPS):
        if self.video_writer is None:
            h, w = bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(str(self.raw_video_path), fourcc, fps, (w, h))
        self.video_writer.write(bgr)

    def write_audio_frame(self, pcm_bytes, sample_rate, channels, sample_width=2):
        if self._wav_file is None:
            self._wav_file = wave.open(str(self.audio_path), 'wb')
            self._wav_file.setnchannels(channels)
            self._wav_file.setsampwidth(sample_width)
            self._wav_file.setframerate(sample_rate)
        self._wav_file.writeframes(pcm_bytes)

    def finalize_recording(self, actual_fps=RECORDING_FPS):
        """Finalize recording and upload the final video to S3."""

        if self.recording_finalized:
            print(f"[RECORDING] Already finalized: {self.session_id}")
            return

        

        print("\n[RECORDING] ===============================")
        print(f"[RECORDING] Finalizing: {self.session_id}")

        # 1. Close video writer
        if self.video_writer is not None:
            try:
                print("[RECORDING] Closing video writer...")
                self.video_writer.release()
            except Exception as e:
                print(f"[RECORDING] Video writer close error: {e}")
            finally:
                self.video_writer = None

        # 2. Close WAV
        if self._wav_file is not None:
            try:
                print("[RECORDING] Closing audio file...")
                self._wav_file.close()
            except Exception as e:
                print(f"[RECORDING] WAV close error: {e}")
            finally:
                self._wav_file = None

        print(f"[RECORDING] Raw video: {self.raw_video_path}")
        print(f"[RECORDING] Audio: {self.audio_path}")
        print(f"[RECORDING] Final: {self.final_video_path}")

        if not self.raw_video_path.exists():
            print("[RECORDING] ERROR: Raw video does not exist")
            return

        raw_size = self.raw_video_path.stat().st_size
        print(f"[RECORDING] Raw video size: {raw_size / (1024 * 1024):.2f} MB")

        if raw_size < 1024:
            print("[RECORDING] ERROR: Raw video is too small / empty")
            return

        # 3. Create final video
        try:
            print("[RECORDING] Creating final MP4...")

            # Audio exists -> combine audio + video
            if self.audio_path.exists() and self.audio_path.stat().st_size > 1024:
                print("[RECORDING] Audio found — combining audio + video")

                combine_audio_video(
                    self.raw_video_path,
                    self.audio_path,
                    self.final_video_path,
                    actual_fps
                )
            else:
                # IMPORTANT:
                # Audio missing hone par bhi video S3 par jana chahiye
                print("[RECORDING] WARNING: Audio missing — uploading video only")

                import shutil

                if self.final_video_path.exists():
                    self.final_video_path.unlink()

                shutil.copy2(
                    self.raw_video_path,
                    self.final_video_path
                )

        except Exception as e:
            print(f"[RECORDING] COMBINE FAILED: {repr(e)}")

            # FALLBACK:
            # combine_audio_video fail hua to raw video ko final bana do
            try:
                import shutil

                print("[RECORDING] Falling back to raw video...")

                if self.final_video_path.exists():
                    self.final_video_path.unlink()

                shutil.copy2(
                    self.raw_video_path,
                    self.final_video_path
                )

            except Exception as fallback_error:
                print(
                    f"[RECORDING] FALLBACK FAILED: "
                    f"{repr(fallback_error)}"
                )
                return

        # 4. Verify final file
        if not self.final_video_path.exists():
            print("[RECORDING] ERROR: Final video was not created")
            return

        final_size = self.final_video_path.stat().st_size

        print(
            f"[RECORDING] Final video size: "
            f"{final_size / (1024 * 1024):.2f} MB"
        )

        if final_size < 1024:
            print("[RECORDING] ERROR: Final video is empty")
            return

        # 5. Upload to S3
        # 5. Upload to S3
        try:
            print("\n" + "=" * 60)
            print("[S3] UPLOADING INTERVIEW VIDEO TO S3...")
            print(f"[S3] Local File: {self.final_video_path}")
            print(f"[S3] Session ID: {self.session_id}")
            print("=" * 60)

            s3_result = upload_evidence(
                self.final_video_path,
                self.session_id
            )

            if s3_result:
                print("\n" + "★" * 60)
                print(" [SUCCESS] VIDEO UPLOADED TO S3 SUCCESSFULLY!")
                print(f" [SUCCESS] S3 Key: {s3_result}")
                print("★" * 60 + "\n")

                # Mark finalized ONLY after successful S3 upload
                self.recording_finalized = True
                return s3_result
            else:
                print("\n[S3 ERROR] Upload failed: upload_evidence returned None")
                print("[S3] Recording un-finalized to allow retry.\n")
                return None  

        except Exception as e:
            print("\n[S3 ERROR] VIDEO UPLOAD FAILED")
            print(f"[S3 Error]: {repr(e)}\n")
            return None

        except Exception as e:
            print("[S3] VIDEO UPLOAD FAILED")
            print(f"[S3] Error: {repr(e)}")
            print("[S3] Recording will remain un-finalized so upload can be retried.")
            return None 
class LiveKitProctor:
    def __init__(self, session_id, livekit_url, token):
        self.session_id = session_id
        self.livekit_url = livekit_url
        self.token = token
        self.candidate_identity = f"candidate_{session_id}"
        self.room = rtc.Room()
        self._processing = False
        self._last_status_time = 0.0
        self._status_min_interval = 0.5
        self._last_log_time = 0.0
        self._log_min_interval = 5.0
        self._logged_frame_info = False
        self._session = None
        self._stopped = False

        # MediaPipe solvers are created once and reused across frames.
        import mediapipe as mp
        self._mp = mp
        self._face_det = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True, min_detection_confidence=0.5
        )

        self.room.on("track_subscribed", self._on_track_subscribed)
        self.room.on("track_subscription_failed", self._on_track_subscription_failed)
        self.room.on("participant_disconnected", self._on_participant_disconnected)
        self.room.on("disconnected", self._on_disconnected)

    def _on_participant_disconnected(self, participant, *_):
        try:
            if participant is not None and participant.identity == self.candidate_identity:
                print(
                    f"[PROCTOR] Candidate left room — stopping proctor for {self.session_id}"
                )
                asyncio.create_task(self._cleanup())
        except Exception as e:
            print(f"[PROCTOR] participant_disconnected handler error: {e}")

    def _on_disconnected(self, *_):
        print(f"[PROCTOR] Room disconnected — cleaning up {self.session_id}")
        asyncio.create_task(self._cleanup())

    async def _cleanup(self):
        """Deregister the proctor and release resources so the session stops
        appearing in the admin live view once the interview is over."""

        print(f"[PROCTOR] Cleaning up proctor for {self.session_id}")
        try:
            if self._stopped:
                return
            self._stopped = True
            proctors.pop(self.session_id, None)
            latest_frames.pop(self.session_id, None)
            try:
                self._face_det.close()
                self._face_mesh.close()
            except Exception:
                pass
            session = self._session
            if session is not None:
                await asyncio.to_thread(session.finalize_recording)
               
            try:
                await self.room.disconnect()
            except Exception:
                pass
            print(f"[PROCTOR] Proctor stopped for {self.session_id}")
        except Exception as e:
            print(f"[PROCTOR] cleanup error: {e}")

    def _on_track_subscription_failed(self, participant, track_sid, error):
        try:
            print(
                f"[PROCTOR] SUBSCRIPTION FAILED for {getattr(participant, 'identity', '?')} "
                f"track_sid={track_sid}: {error}"
            )
        except Exception as e:
            print(f"[PROCTOR] subscription_failed log error: {e}")

    async def start(self):
        await self.room.connect(self.livekit_url, self.token)
        print(
            f"[PROCTOR] Joined room {self.session_id} "
            f"(identity={self.room.local_participant.identity})"
        )
        # No explicit add_subscription needed: livekit SDK >=1.x auto-subscribes
        # to all remote tracks by default (RoomOptions.auto_subscribe=True), and
        # `track_subscribed` fires for the candidate's tracks whether they join
        # before or after the proctor.
        # Periodically log what the proctor sees in the room so the admin can
        # confirm the candidate's video track is visible + subscribed even if no
        # frames arrive yet. Also force explicit video subscription: in practice
        # auto-subscribe only subscribes the candidate's audio, never video.
        self._last_room_log_time = 0.0
        asyncio.create_task(self._room_state_loop())

    async def _room_state_loop(self):
        while not self._stopped:
            self._ensure_candidate_video_subscription()
            now = time.time()
            if now - self._last_room_log_time >= 10:
                self._last_room_log_time = now
                self._log_room_state()
            await asyncio.sleep(2)

    def _ensure_candidate_video_subscription(self):
        """Explicitly subscribe to the candidate's video track AND request the
        HIGHEST simulcast layer.

        Auto-subscribe (RoomOptions.auto_subscribe=True) is NOT reliably
        subscribing the candidate's video — only audio arrives. Force the video
        publication to subscribe via RemoteTrackPublication.set_subscribed().
        The SDK then defaults to the LOW layer (e.g. 180x320), which is too low
        for reliable face detection — request HIGH (full resolution) instead."""
        try:
            candidate = self.room.remote_participants.get(self.candidate_identity)
            if candidate is None:
                return
            for sid, pub in list(candidate.track_publications.items()):
                if pub.kind == rtc.TrackKind.KIND_AUDIO:
                    if not pub.subscribed:
                        print(f"[PROCTOR] Explicitly subscribing AUDIO {sid}")
                        pub.set_subscribed(True)
                    continue

                if pub.kind != rtc.TrackKind.KIND_VIDEO:
                    continue
                if not pub.subscribed:
                    print(
                        f"[PROCTOR] Explicitly subscribing candidate video {sid} "
                        f"(auto-subscribe missed it)"
                    )
                    pub.set_subscribed(True)
                # Bump the simulcast layer to full resolution for face detection.
                if hasattr(pub, "set_video_quality"):
                    try:
                        pub.set_video_quality(rtc.VideoQuality.VIDEO_QUALITY_HIGH)
                    except Exception as e:
                        print(f"[PROCTOR] set_video_quality(HIGH) failed for {sid}: {e}")
        except Exception as e:
            print(f"[PROCTOR] ensure_candidate_video_subscription error: {e}")

    def _on_track_subscribed(self, track, publication, participant):
        try:
            if participant is None or track is None:
                return
            if participant.identity != self.candidate_identity:
                # Agent voice track — ignore (not the candidate).
                return
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                print(
                    f"[PROCTOR] Subscribed to candidate video track "
                    f"({participant.identity}, sid={publication.sid if publication else '?'}) — "
                    f"tracking started"
                )
                asyncio.create_task(self._consume_video(track))
            elif track.kind == rtc.TrackKind.KIND_AUDIO:
                print(f"[PROCTOR] Subscribed to candidate audio track — recording started")
                asyncio.create_task(self._consume_audio(track))
        except Exception as e:
            print(f"[PROCTOR] track_subscribed error: {e}")

    async def _consume_video(self, track):
        """Consume video frames via VideoStream (the livekit SDK 1.x way to read
        remote video — RemoteVideoTrack has no add_listener)."""
        try:
            stream = rtc.VideoStream(track, capacity=1)
            async for event in stream:
                if self._stopped:
                    break
                if self._processing:
                    continue
                frame = getattr(event, "frame", None)
                if frame is None:
                    continue
                if self._logged_frame_info is False:
                    try:
                        print(
                            f"[PROCTOR] Video frame arriving: {frame.width}x{frame.height} "
                            f"type={rtc.VideoBufferType.Name(frame.type) if hasattr(rtc.VideoBufferType, 'Name') else frame.type}"
                        )
                    except Exception:
                        pass
                    self._logged_frame_info = True
                bgr = self._frame_to_bgr(frame)
                if bgr is None:
                    continue
                self._processing = True
                try:
                    await self._process_video(bgr)
                finally:
                    self._processing = False
        except Exception as e:
            print(f"[PROCTOR] video stream ended/error: {e}")

    async def _consume_audio(self, track):
        """Consume the candidate's audio track and write raw PCM into the
        session's WAV file (muxed with video later in finalize_recording)."""
        try:
            stream = rtc.AudioStream(track)
            async for event in stream:
                if self._stopped:
                    break
                frame = getattr(event, "frame", None)
                if frame is None:
                    continue
                session = self._session
                if session is None:
                    continue
                try:
                    pcm_bytes = bytes(frame.data)
                    session.write_audio_frame(
                        pcm_bytes,
                        sample_rate=frame.sample_rate,
                        channels=frame.num_channels,
                    )
                except Exception as e:
                    print(f"[PROCTOR] audio write error: {e}")
        except Exception as e:
            print(f"[PROCTOR] audio stream ended/error: {e}")

    async def _process_video(self, bgr):
        try:
            session = self._session
            if session is None:
                return

            result, trigger = await asyncio.to_thread(
                process_frame, bgr, session, self._face_det, self._face_mesh
            )
            if result is None:
                return

            # process_frame annotates bgr in place — store the ANNOTATED frame
            # so /live-frame shows the candidate being traced with boxes.
            ok, enc = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            if ok:
                latest_frames[self.session_id] = enc.tobytes()

            session.last_result = result
            session.write_video_frame(bgr)
            sessions[self.session_id] = session
            self._maybe_publish_status(result)
            self._maybe_log_status(result)
            if trigger:
                await self._publish_trigger(trigger)
        except Exception as e:
            print(f"[PROCTOR] process error: {e}")
        finally:
            self._processing = False

    def _maybe_publish_status(self, result):
        if self._stopped:
            return
        now = time.time()
        if now - self._last_status_time < self._status_min_interval:
            return
        self._last_status_time = now
        asyncio.create_task(self._publish_status(result))

    def _maybe_log_status(self, result):
        """Throttled console log so the admin Engine Logs show live tracking."""
        now = time.time()
        if now - self._last_log_time < self._log_min_interval:
            return
        self._last_log_time = now
        flags = ", ".join(result.get("flags", [])) or "-"
        print(
            f"[TRACK] {self.session_id} | Faces:{result.get('face_count')} | "
            f"Match:{result.get('face_match')} | Score:{result.get('suspicion_score')} | "
            f"Flags:[{flags}]"
        )

    def _log_room_state(self):
        """Log what the proctor sees in the room: every remote participant and
        their published tracks + subscription state. Lets the admin confirm the
        candidate's video track is visible/subscribed."""
        try:
            for identity, participant in list(self.room.remote_participants.items()):
                tracks = []
                for pub in list(participant.track_publications.values()):
                    tracks.append({
                        "sid": pub.sid,
                        "kind": rtc.TrackKind.Name(pub.kind) if hasattr(rtc.TrackKind, "Name") else pub.kind,
                        "source": pub.source,
                        "subscribed": pub.subscribed,
                    })
                print(f"[PROCTOR] Room: {identity} tracks={tracks}")
        except Exception as e:
            print(f"[PROCTOR] room state log error: {e}")

    async def _publish_status(self, result):
        try:
            payload = json.dumps(
                {"type": "cheating_status", **result}, default=_json_default
            ).encode("utf-8")
            await self.room.local_participant.publish_data(payload, reliable=True)
        except Exception as e:
            print(f"[PROCTOR] publish status error: {e}")

    async def _publish_trigger(self, trigger):
        try:
            payload = json.dumps({"type": "trigger", **trigger}, default=_json_default).encode("utf-8")
            await self.room.local_participant.publish_data(payload, reliable=True)
        except Exception as e:
            print(f"[PROCTOR] publish trigger error: {e}")

    @staticmethod
    def _frame_to_bgr(frame):
        # Normalize ANY source pixel format (RGBA/BGRA/NV12/I422/... and stride)
        # to I420 via the SDK's native converter, then cvtColor to BGR. The old
        # buffer-length heuristics produced black frames on some formats.
        try:
            converted = frame.convert(rtc.VideoBufferType.I420)
        except Exception as e:
            print(f"[PROCTOR] frame convert to I420 failed: {e}")
            return None
        w, h = converted.width, converted.height
        data = np.frombuffer(converted.data, dtype=np.uint8)
        if data.size != w * h * 3 // 2:
            print(
                f"[PROCTOR] unexpected I420 buffer size {data.size} "
                f"(expected {w * h * 3 // 2}) for {w}x{h}"
            )
            return None
        yuv = data.reshape((h * 3 // 2, w))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    async def stop(self):
        await self._cleanup()


# ── Detection pipeline (v1) ──────────────────────────────────────────────────
def process_frame(bgr, session, face_det, face_mesh):
    """Run v1 detection on a BGR frame.

    Returns (result_dict, trigger_dict_or_None). `trigger` is set when a
    violation reaches the 12-occurrence threshold — publish it as an AI alert.
    """
    now = time.time()
    height, width = bgr.shape[:2]
    rgb_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    formatted_time = time.strftime("%H:%M:%S", time.localtime(now))

    results = face_det.process(rgb_frame)
    face_count = len(results.detections) if results.detections else 0

    result = {
        "sessionId": session.session_id,
        "person_in_frame": face_count > 0,
        "face_count": face_count,
        "multi_face": face_count > 1,
        "face_match": None,
        "match_distance": None,
        "eye_state": "N/A",
        "gaze": "N/A",
        "sunglasses": False,
        "head_pose": None,
        "face_box": None,
        "eye_boxes": [],
        "gaze_away_duration": 0,
        "face_gone_duration": 0,
        "flags": [],
        "suspicion_score": round(session.suspicion_score, 1),
        "events": session.events[-10:],
        "violations": list(session.tracker.violations.keys()),
    }
    trigger = None

    def _log(v_type, hr_text, ai_warning):
        nonlocal trigger

        ai_warning = f"[system_alert_cheating_engine] - {ai_warning}"

        # These two violations trigger instantly
        INSTANT_TRIGGER_VIOLATIONS = {
            "Missing Candidate",
            "Multiple People Detected",
        }

        if v_type in INSTANT_TRIGGER_VIOLATIONS:

            # Save/log the violation normally
            session.tracker.log_violation(
                v_type,
                formatted_time,
                hr_text,
                ai_warning,
                bgr
            )

            # Force instant trigger
            reached = True

        else:
            # All other violations need threshold = 3
            reached = session.tracker.log_violation(
                v_type,
                formatted_time,
                hr_text,
                ai_warning,
                bgr
            )

            # These two violations should trigger instantly
            INSTANT_VIOLATIONS = {
                "Missing Candidate",
                "Multiple People Detected",
            }

            if v_type in INSTANT_VIOLATIONS:
                if v_type not in session.instant_violation_active:
                    reached = True
                    session.instant_violation_active.add(v_type)
        # Your existing snapshot code remains here...
        if reached:
            trigger = {
                "violation": v_type,
                "aiWarning": ai_warning,
                "hrMessage": hr_text,
                "timestamp": formatted_time,
                "suspicion_score": round(session.suspicion_score, 1),
            }

            # ─────────────────────────────────────────────
            # Upload trigger snapshot to S3
            # ─────────────────────────────────────────────
            try:
                snapshot_dir = Path(TRIGGER_DIR) / session.session_id
                snapshot_dir.mkdir(parents=True, exist_ok=True)

                safe_name = (
                    v_type.lower()
                    .replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("/", "_")
                    .replace(":", "_")
                )

                snapshot_path = (
                    snapshot_dir /
                    f"{session.session_id}_{safe_name}_{int(now)}.jpg"
                )

                cv2.imwrite(str(snapshot_path), bgr)

                print(f"[SNAPSHOT] Saved: {snapshot_path}")

                # Upload to S3
                s3_result = upload_evidence(
                    snapshot_path,
                    session.session_id
                )

                print(
                    f"[S3] Snapshot uploaded successfully: "
                    f"{snapshot_path} -> {s3_result}"
                )

            except Exception as e:
                print(f"[S3] Snapshot upload failed: {e}")
            
    face_box = None
    if face_count > 0:
        det = results.detections[0]
        box = det.location_data.relative_bounding_box
        x, y = int(box.xmin * width), int(box.ymin * height)
        w_box, h_box = int(box.width * width), int(box.height * height)
        face_box = (x, y, w_box, h_box)
        result["face_box"] = [x / width, y / height, w_box / width, h_box / height]

        # Identity verification (face_recognition) at a throttled cadence.
        if session.reference_encoding is not None and now - session.last_verify_time > VERIFY_EVERY_SECONDS:
            session.last_verify_time = now
            top, left = max(0, y), max(0, x)
            bottom, right = min(height, y + h_box), min(width, x + w_box)
            try:
                encodings = face_recognition_encodings(rgb_frame, [(top, right, bottom, left)])
                if encodings:
                    distance = face_recognition_distance(session.reference_encoding, encodings[0])
                    session.identity_ok = bool(distance <= MATCH_TOLERANCE)
                    result["face_match"] = session.identity_ok
                    result["match_distance"] = round(distance, 3)
                    if not session.identity_ok and now - session.last_mismatch_snap_time > 3.0:
                        session.last_mismatch_snap_time = now
                        session.bump_score(3)
                        session.add_event("identity_mismatch", f"dist={distance:.3f}")
                        result["flags"].append("identity_mismatch")
                        _log(
                            "Identity Mismatch",
                            "Unrecognized person in candidate position",
                            "Identity mismatch detected",
                        )
            except Exception as e:
                print(f"[PROCTOR] verification skipped: {e}")

    # ── Lighting ─────────────────────────────────────────────────────────
    face_pct, bg_pct = analyze_lighting_conditions(bgr, face_box)
    lighting_warning = None
    if face_pct is not None:
        if face_pct < MIN_BRIGHTNESS_PCT:
            lighting_warning = "WARN: FRONT LIGHT LOW"
        elif face_pct > MAX_BRIGHTNESS_PCT:
            lighting_warning = "WARN: FRONT LIGHT TOO HIGH - USE NORMAL LIGHT"
    if not lighting_warning and bg_pct is not None:
        if bg_pct < MIN_BRIGHTNESS_PCT:
            lighting_warning = "WARN: BACKGROUND LIGHT IS LOW"
        elif bg_pct > MAX_BRIGHTNESS_PCT:
            lighting_warning = "WARN: BACKGROUND LIGHT HIGH - SIT IN NORMAL LIGHT"
    result["light_on"] = face_pct is not None and face_pct > MIN_BRIGHTNESS_PCT
    result["low_light"] = lighting_warning is not None
    result["brightness"] = round(face_pct, 1) if face_pct is not None else None
    if lighting_warning and now - session.last_light_snap_time > 3.0:
        session.last_light_snap_time = now
        session.bump_score(1)
        session.add_event("light_violation", lighting_warning)
        result["flags"].append("light_violation")
        _log(
            f"Lighting Violation ({lighting_warning.replace('WARN: ', '')})",
            f"Lighting Violation: {lighting_warning}",
            "Poor lighting detected",
        )

    # ── People count ─────────────────────────────────────────────────────
    if face_count > 1:
        result["flags"].append("multiple_faces")
        session.add_event("multi_face", f"{face_count} faces detected")
        if now - session.last_snap_time > 5.0:      # 0.0 → 5.0
            session.last_snap_time = now
            session.bump_score(5)                    
            _log(
                "Multiple People Detected",
                f"Multiple people detected ({face_count} people)",
                "Multiple people detected",
            )

    elif face_count == 0:
        session.identity_ok = None
        session.glasses_consecutive_frames = 0
        session.eye_closed_start_time = None
        session.last_blink_timestamp = now
        session.iris_streak_direction = None
        session.iris_streak_count = 0

        if session.face_gone_start is None:
            session.face_gone_start = now
        gone_dur = now - session.face_gone_start
        result["face_gone_duration"] = round(gone_dur, 1)
        if gone_dur > 5:
            result["flags"].append("face_gone_long")
            session.add_event("face_gone", f"{gone_dur:.1f}s")
            if now - session.last_missing_face_snap_time > 5.0:   # 0.0 → 5.0
                session.last_missing_face_snap_time = now
                session.bump_score(2)                               
                _log(
                    "Missing Candidate",
                    "Candidate missing from frame",
                    "Candidate missing from frame",
                )
    else:
        session.face_gone_start = None

        # ── Head Movement / Head Pose Detection (ported from main.py) ────
        # Uses the same MediaPipe FaceDetection relative_keypoints (no
        # dependency on face_mesh), so this runs even if FaceMesh below
        # fails to find landmarks on a given frame.
        det0 = results.detections[0]
        keypoints = det0.location_data.relative_keypoints
        rx, ry = int(keypoints[0].x * width), int(keypoints[0].y * height)
        lx, ly = int(keypoints[1].x * width), int(keypoints[1].y * height)
        nx, ny = int(keypoints[2].x * width), int(keypoints[2].y * height)
        mx, my = int(keypoints[3].x * width), int(keypoints[3].y * height)

        eye_center_x = (rx + lx) / 2.0
        eye_dist = abs(lx - rx)
        yaw_ratio = (nx - eye_center_x) / eye_dist if eye_dist > 0 else 0

        eye_center_y = (ry + ly) / 2.0
        face_length = my - eye_center_y
        pitch_ratio = (ny - eye_center_y) / face_length if face_length > 0 else 0.5

        movement_state = None
        if yaw_ratio < -HEAD_YAW_THRESHOLD:
            movement_state = "LOOKING RIGHT"
        elif yaw_ratio > HEAD_YAW_THRESHOLD:
            movement_state = "LOOKING LEFT"
        elif pitch_ratio < HEAD_PITCH_UP_THRESHOLD:
            movement_state = "LOOKING UP"
        elif pitch_ratio > HEAD_PITCH_DOWN_THRESHOLD:
            movement_state = "LOOKING DOWN"

        result["head_pose"] = movement_state

        if movement_state:
            result["flags"].append("head_movement")
            session.bump_score(2)
            session.add_event("head_movement", movement_state)
            if now - session.last_movement_snap_time > 3.0:
                session.last_movement_snap_time = now
                short_dir = movement_state.replace("LOOKING ", "").lower()
                _log(
                    movement_state.title(),
                    f"Head turned {short_dir}",
                    f"Candidate looking {short_dir}",
                )

        # ── FaceMesh: glasses, EAR, blink, iris ─────────────────────────
        mesh_results = face_mesh.process(rgb_frame)
        if mesh_results.multi_face_landmarks:
            landmarks = mesh_results.multi_face_landmarks[0].landmark

            # Black glasses / eye obstruction (3 consecutive frames)
            if detect_black_glasses(landmarks, bgr, width, height):
                session.glasses_consecutive_frames += 1
            else:
                session.glasses_consecutive_frames = max(0, session.glasses_consecutive_frames - 1)
            if session.glasses_consecutive_frames >= REQUIRED_GLASSES_FRAMES:
                result["sunglasses"] = True
                result["eye_state"] = "occluded"
                result["flags"].append("eye_obstruction")
                session.bump_score(2)
                if now - session.last_glasses_snap_time > 3.0:
                    session.last_glasses_snap_time = now
                    _log(
                        "Eye Obstruction",
                        "Dark sunglasses or covered eyes detected",
                        "Eye obstruction detected",
                    )

            ear = calculate_ear(landmarks, width, height)

            # Eye closure (> 5s)
            if ear < EAR_CLOSED_THRESHOLD:
                if session.eye_closed_start_time is None:
                    session.eye_closed_start_time = now
                    session.last_blink_timestamp = now
                closed_duration = now - session.eye_closed_start_time
                if closed_duration >= 5.0:
                    result["eye_state"] = "closed"
                    result["flags"].append("eye_closed")
                    session.bump_score(2)
                    if now - session.last_eye_closure_snap_time > 3.0:
                        session.last_eye_closure_snap_time = now
                        _log(
                            "Eye Closure (>5s)",
                            "Candidate eyes closed continuously for over 5 seconds",
                            "Eyes closed for extended duration",
                        )
            else:
                session.eye_closed_start_time = None

            # No-blink liveliness (> 8s)
            time_since_last_blink = now - session.last_blink_timestamp
            if time_since_last_blink >= NO_BLINK_WARN_SECONDS:
                result["flags"].append("no_blink")
                session.bump_score(2)
                if now - session.last_no_blink_snap_time > 3.0:
                    session.last_no_blink_snap_time = now
                    _log(
                        "No Blink (>60s)",
                        "No natural eye blink detected for over 60 seconds",
                        "Extended no-blink detected",
                    )

            if ear < EAR_CLOSED_THRESHOLD:
                session.last_blink_timestamp = now

            # Iris / gaze streaks
            if ear >= EYE_OPEN_FOR_GAZE_EAR:
                iris_dir, r_iris_pt, l_iris_pt = detect_iris_direction(landmarks, width, height)
                result["eye_boxes"] = [
                    [l_iris_pt[0] / width, l_iris_pt[1] / height, 3 / width, 3 / height],
                    [r_iris_pt[0] / width, r_iris_pt[1] / height, 3 / width, 3 / height],
                ]
                if iris_dir:
                    if iris_dir == session.iris_streak_direction:
                        session.iris_streak_count += 1
                    else:
                        session.iris_streak_direction = iris_dir
                        session.iris_streak_count = 1
                    result["gaze"] = iris_dir
                    if session.iris_streak_count >= IRIS_STREAK_TRIGGER:
                        result["flags"].append("gaze_away")
                        session.bump_score(2)
                        if now - session.last_trigger_snap_time > 3.0:
                            session.last_trigger_snap_time = now
                            _log(
                                iris_dir.title(),
                                f"Candidate showed repeated eye movement ({iris_dir})",
                                "Repeated gaze deviation detected",
                            )
                else:
                    session.iris_streak_direction = None
                    session.iris_streak_count = 0

    session.suspicion_score = min(session.suspicion_score, 100)
    result["suspicion_score"] = round(session.suspicion_score, 1)

    annotate_frame(bgr, results, width, height, face_count, result, session, now)
    return result, trigger


def annotate_frame(frame, results, width, height, face_count, result, session, now):
    """Draw detection overlays directly onto the BGR frame (in place).

    The annotated frame is what the admin dashboard sees via /live-frame, so the
    live view shows the candidate being traced with boxes + status text."""
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, f"SID: {session.session_id}", (10, height - 30), font, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)), (10, height - 10), font, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, f"Score: {result['suspicion_score']}  Engine: v1", (10, 30), font, 0.7, (0, 255, 255), 2)

    if face_count == 0:
        cv2.putText(frame, "WARNING: NO CANDIDATE", (20, 65), font, 1.0, (0, 165, 255), 3)
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 165, 255), 3)
    elif face_count > 1:
        cv2.putText(frame, f"WARNING: {face_count} PEOPLE!", (20, 65), font, 1.0, (0, 0, 255), 3)
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 4)

    if results.detections:
        for i, det in enumerate(results.detections):
            b = det.location_data.relative_bounding_box
            bx, by = int(b.xmin * width), int(b.ymin * height)
            bw, bh = int(b.width * width), int(b.height * height)
            if i == 0 and result.get("face_match") is True:
                color, label = (0, 255, 0), "Candidate (match)"
            elif i == 0 and result.get("face_match") is False:
                color, label = (0, 0, 255), "UNKNOWN PERSON"
            else:
                color, label = (0, 255, 255), "Unknown"
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)
            cv2.putText(frame, label, (bx, max(20, by - 10)), font, 0.6, color, 2)

    for ex, ey, ew, eh in result.get("eye_boxes", []):
        cx = int((ex + ew / 2.0) * width)
        cy = int((ey + eh / 2.0) * height)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

    if result.get("eye_state") and result["eye_state"] != "N/A":
        cv2.putText(frame, f"Eye: {result['eye_state']}  Gaze: {result.get('gaze')}", (20, height - 60), font, 0.55, (0, 255, 255), 2)

    if result.get("head_pose"):
        cv2.putText(frame, f"WARNING: {result['head_pose']}", (20, height - 110), font, 0.65, (0, 165, 255), 2)

    if result.get("flags"):
        flags_str = ", ".join(result["flags"])
        cv2.putText(frame, "FLAGS: " + flags_str, (20, height - 85), font, 0.55, (0, 0, 255), 2)


# face_recognition wrapped for clarity + JSON-safe (kept thin so failures fail soft)
def face_recognition_encodings(rgb_frame, known_face_locations):
    import face_recognition
    return face_recognition.face_encodings(rgb_frame, known_face_locations=known_face_locations)


def face_recognition_distance(reference_encoding, encoding):
    import face_recognition
    return face_recognition.face_distance([reference_encoding], encoding)[0]


# ── REST: Register face ──────────────────────────────────────────────────────
async def register_face(request):
    """POST /register-face  multipart: image + sessionId.

    Uses v1's `load_reference_encoding` (face_recognition) for identity matching.
    Rejects selfies with no/ambiguous face so the UI can re-prompt.
    """
    reader = await request.multipart()
    session_id = None
    image_data = None

    async for part in reader:
        if part.name == 'sessionId':
            session_id = (await part.read()).decode()
        elif part.name == 'image':
            image_data = await part.read()

    if not session_id or not image_data:
        return web.json_response({"error": "sessionId and image required"}, status=400)

    arr = np.frombuffer(image_data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return web.json_response({"error": "Invalid image"}, status=400)

    ref_path = os.path.join(EVIDENCE_DIR, session_id, "reference.jpg")
    os.makedirs(os.path.dirname(ref_path), exist_ok=True)
    cv2.imwrite(ref_path, img)

    # v1 identity encoding
    try:
        reference_encoding = await asyncio.to_thread(load_reference_encoding, Path(ref_path))
    except Exception as e:
        return web.json_response(
            {"error": f"Could not encode reference face: {e}"}, status=400
        )
    if reference_encoding is None:
        # load_reference_encoding returns None when no/ambiguous face found
        return web.json_response(
            {"error": "No face or ambiguous face detected in selfie. Ensure a single, clear face is visible."},
            status=400,
        )

    session = sessions.setdefault(session_id, ProctorSession(session_id))
    session.reference_encoding = reference_encoding

    proctor = proctors.get(session_id)
    if proctor:
        proctor._session = session

    print(f"[REGISTER] Face registered for session: {session_id}")
    return web.json_response({"status": "ok", "sessionId": session_id})


# ── REST: Get report ─────────────────────────────────────────────────────────
async def get_report(request):
    """GET /report/:sessionId — returns the cheating report for a session."""
    session_id = request.match_info['sessionId']
    session = sessions.get(session_id)
    if not session:
        return web.json_response({"error": "Session not found"}, status=404)

    return web.json_response(_json_safe({
        "sessionId": session_id,
        "suspicion_score": round(session.suspicion_score, 1),
        "events": session.events,
        "total_events": len(session.events),
        "violations": list(session.tracker.violations.keys()),
        "last_result": session.last_result,
    }))


# ── REST: Active proctors ─────────────────────────────────────────────────────
async def get_proctors(request):
    """GET /proctors — list active proctor sessions for the live admin view."""
    items = []
    for session_id, proctor in list(proctors.items()):
        session = sessions.get(session_id)
        items.append({
            "sessionId": session_id,
            "candidateIdentity": proctor.candidate_identity,
            "joined": True,
            "suspicion_score": round(session.suspicion_score, 1) if session else 0,
            "face_count": (session.last_result or {}).get("face_count", 0) if session else 0,
            "flags": (session.last_result or {}).get("flags", []) if session else [],
            "hasReference": bool(session and session.reference_encoding is not None),
        })
    return web.json_response(_json_safe({"proctors": items}))


# ── LiveKit proctor start ────────────────────────────────────────────────────
async def start_proctor(request):
    """POST /api/proctor/start  body: { sessionId, livekitUrl, token }"""
    if not LIVEKIT_AVAILABLE:
        return web.json_response(
            {"error": "livekit package not installed — proctor unavailable"},
            status=503,
        )
    body = await request.json()
    session_id = body.get("sessionId")
    url = body.get("livekitUrl")
    token = body.get("token")
    if not session_id or not url or not token:
        return web.json_response({"error": "sessionId, livekitUrl and token required"}, status=400)

    existing = proctors.get(session_id)
    if existing:
        return web.json_response({"status": "already_active", "sessionId": session_id})

    session = sessions.setdefault(session_id, ProctorSession(session_id))
    proctor = LiveKitProctor(session_id, url, token)
    proctor._session = session
    proctors[session_id] = proctor

    async def _run():
        try:
            await proctor.start()
        except Exception as e:
            print(f"[PROCTOR] failed to start for {session_id}: {e}")
            proctors.pop(session_id, None)

    asyncio.create_task(_run())
    return web.json_response({"status": "started", "sessionId": session_id})


# ── Evidence / live frame endpoints (admin UI) ───────────────────────────────
async def list_evidence_sessions(request):
    """GET /evidence — list all session IDs with evidence."""
    if not os.path.exists(EVIDENCE_DIR):
        return web.json_response({"sessions": []})
    session_dirs = []
    for d in sorted(os.listdir(EVIDENCE_DIR), reverse=True):
        path = os.path.join(EVIDENCE_DIR, d)
        if os.path.isdir(path):
            files = [f for f in os.listdir(path) if f.endswith('.jpg')]
            session_dirs.append({
                "sessionId": d,
                "fileCount": len(files),
                "files": sorted(files),
                "hasReference": "reference.jpg" in files,
            })
    return web.json_response({"sessions": session_dirs})


async def get_evidence_image(request):
    """GET /evidence/{sessionId}/{filename} — serve an evidence image."""
    session_id = request.match_info['sessionId']
    filename = request.match_info['filename']
    filepath = os.path.join(EVIDENCE_DIR, session_id, filename)
    if not os.path.exists(filepath):
        return web.json_response({"error": "File not found"}, status=404)
    return web.FileResponse(filepath)


async def get_evidence_mapping(request):
    """GET /evidence/{sessionId}/mapping/{filename} — serve mapping JSON for an image."""
    session_id = request.match_info['sessionId']
    filename = request.match_info['filename']
    json_file = filename.replace('.jpg', '.json')
    filepath = os.path.join(EVIDENCE_DIR, session_id, json_file)
    if not os.path.exists(filepath):
        return web.json_response({"error": "No mapping data"}, status=404)
    with open(filepath, 'r') as f:
        data = json.load(f)
    return web.json_response(data)


async def get_live_frame(request):
    """GET /live-frame/{sessionId} — serve the latest video frame as JPEG."""
    session_id = request.match_info['sessionId']
    frame_data = latest_frames.get(session_id)
    if not frame_data:
        return web.json_response({"error": "No frame available"}, status=404)
    return web.Response(body=frame_data, content_type='image/jpeg')


# ── Health / index (the Node health dashboard fetches `/` and expects 200) ───
async def index(request):
    """GET / — health ping. The Node admin health check marks the engine as
    Down unless this returns 200."""
    return web.json_response({"status": "ok", "engine": "cheat-engine-v1"})


# ── Engine logs over HTTP ────────────────────────────────────────────────────
# The Node server runs in Docker and its own PM2 daemon cannot see this host
# process, so the admin "Engine Logs" page cannot read `pm2 logs`. Instead the
# engine tees its stdout/stderr into its own log file and serves it here; the
# Node endpoint fetches it over HTTP (same channel as the working health check).
LOG_FILE = os.getenv("CHEATING_ENGINE_LOG", os.path.join(os.path.dirname(__file__), "engine.log"))


class _Tee:
    """Duplicates writes to multiple streams (e.g. real stdout + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def setup_log_file():
    """Redirect stdout/stderr so all output is also appended to LOG_FILE."""
    try:
        log_dir = os.path.dirname(LOG_FILE) or "."
        os.makedirs(log_dir, exist_ok=True)
        fh = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
        sys.stdout = _Tee(sys.__stdout__, fh)
        sys.stderr = _Tee(sys.__stderr__, fh)
        print(f"[SERVER] Teeing engine logs to {LOG_FILE}")
    except Exception as e:
        print(f"[SERVER] Could not open log file {LOG_FILE}: {e}")


async def get_logs(request):
    """GET /logs?lines=N — tail of this engine's own log file.

    Served over HTTP so the Node admin dashboard can read engine logs without
    needing access to the host's PM2 daemon."""
    lines = int(request.query.get("lines", 150))
    if not os.path.exists(LOG_FILE):
        return web.json_response({"logs": ""})
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            tail = deque(f, maxlen=lines)
        return web.json_response({"logs": "".join(tail)})
    except Exception as e:
        return web.json_response({"logs": f"[LOG READ ERROR] {e}"})


# ── Routes ───────────────────────────────────────────────────────────────────
app.router.add_get('/', index)
app.router.add_get('/logs', get_logs)
app.router.add_post('/register-face', register_face)
app.router.add_get('/report/{sessionId}', get_report)
app.router.add_get('/proctors', get_proctors)
app.router.add_post('/api/proctor/start', start_proctor)
app.router.add_get('/evidence', list_evidence_sessions)
app.router.add_get('/evidence/{sessionId}/{filename}', get_evidence_image)
app.router.add_get('/evidence/{sessionId}/mapping/{filename}', get_evidence_mapping)
app.router.add_get('/live-frame/{sessionId}', get_live_frame)


# ── Socket.IO (legacy websocket relay, kept for v0 parity) ───────────────────
@sio.on('video_frame', namespace='/interview')
async def on_video_frame(sid, data):
    # LiveKit transport: proctor subscribes to the SFU directly — ignore relayed
    # frames to avoid double-counting.
    session_id = data.get('sessionId') if isinstance(data, dict) else None
    if session_id in proctors:
        return


@sio.on('screen_recording', namespace='/interview')
async def on_screen_recording(sid, data):
    session_id = data.get('sessionId') if isinstance(data, dict) else None
    if not session_id or session_id in proctors:
        return
    session = sessions.get(session_id)
    if not session:
        return
    session.bump_score(10)
    session.add_event("screen_recording", data.get("detail", "detected"))
    print(f"[ALERT] Screen recording detected for {session_id}: {data}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Cheat Engine v1 LiveKit proctor server")
    parser.add_argument("--port", type=int, default=6544)
    parser.add_argument("--tolerance", type=float, default=0.6, help="Face match tolerance (lower=stricter)")
    args = parser.parse_args()

    global MATCH_TOLERANCE
    MATCH_TOLERANCE = args.tolerance

    setup_log_file()

    import socket as _sock
    s = _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        network_ip = s.getsockname()[0]
    except Exception:
        network_ip = '?.?.?.?'
    finally:
        s.close()

    print(f"[SERVER] Local:   http://localhost:{args.port}")
    print(f"[SERVER] Network: http://{network_ip}:{args.port}")
    print(f"[SERVER] Endpoints:")
    print(f"         POST /register-face  (multipart: image + sessionId)")
    print(f"         POST /api/proctor/start  (LiveKit join: sessionId + livekitUrl + token)")
    print(f"         GET  /report/{{sessionId}}")
    print(f"         GET  /evidence")
    print(f"         GET  /live-frame/{{sessionId}}")
    web.run_app(app, host='0.0.0.0', port=args.port, print=None)


if __name__ == "__main__":
    main()



