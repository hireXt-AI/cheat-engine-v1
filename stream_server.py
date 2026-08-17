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

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

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
    MIN_BRIGHTNESS_PCT,
    MAX_BRIGHTNESS_PCT,
)

from aiohttp import web
import socketio


# ── Server setup ─────────────────────────────────────────────────────────────
sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')
app = web.Application()
sio.attach(app)

EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), '.evidence')
TRIGGER_DIR = os.path.join(EVIDENCE_DIR, 'trigger_points')
os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs(TRIGGER_DIR, exist_ok=True)

MATCH_TOLERANCE = 0.6  # face_recognition face_distance tolerance (lower = stricter)

# ── Engine globals ───────────────────────────────────────────────────────────
# Per-session proctor state: sessionId -> ProctorSession
proctors = {}  # sessionId -> LiveKitProctor (LiveKit transport sessions)
latest_frames = {}  # sessionId -> jpeg_bytes (most recent frame for live monitoring)
sessions = {}  # sessionId -> detection state (mirrors v0's session dict)


# ── v1 detection config (mirrors main.start_interview_recording) ─────────────
EAR_CLOSED_THRESHOLD = 0.18
NO_BLINK_WARN_SECONDS = 8.0
REQUIRED_GLASSES_FRAMES = 3
EYE_OPEN_FOR_GAZE_EAR = 0.23
IRIS_STREAK_TRIGGER = 6
TRIGGER_THRESHOLD = 12          # occurrences per violation type -> trigger point
VERIFY_EVERY_SECONDS = 2.0      # face_recognition identity check cadence
SNAPSHOT_COOLDOWN_S = 3.0       # min seconds between standard evidence snapshots


class ProctorSession:
    """Per-session detection state + CSV/trigger tracker."""

    def __init__(self, session_id):
        self.session_id = session_id
        self.reference_encoding = None

        # ViolationTracker writes trigger points to CSV under .evidence
        csv_path = Path(EVIDENCE_DIR) / f"trigger_points_{session_id}.csv"
        self.tracker = ViolationTracker(
            csv_path=csv_path,
            trigger_snapshot_dir=Path(TRIGGER_DIR) / session_id,
            user_id=f"USER_{session_id}",
            threshold=TRIGGER_THRESHOLD,
        )

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

    # ── Event + score helpers (mirror v0) ─────────────────────────────────
    def add_event(self, event_type, detail=""):
        self.events.append({"time": round(time.time(), 2), "type": event_type, "detail": detail})
        if len(self.events) > 200:
            self.events = self.events[-200:]

    def bump_score(self, delta):
        self.suspicion_score = min(self.suspicion_score + delta, 100)


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
        self._session = None

        # MediaPipe solvers are created once and reused across frames.
        import mediapipe as mp
        self._mp = mp
        self._face_det = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.7
        )
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True, min_detection_confidence=0.5
        )

        self.room.on("track_subscribed", self._on_track_subscribed)

    async def start(self):
        await self.room.connect(self.livekit_url, self.token)
        print(
            f"[PROCTOR] Joined room {self.session_id} "
            f"(identity={self.room.local_participant.identity})"
        )
        for kind in (rtc.TrackKind.KIND_VIDEO, rtc.TrackKind.KIND_AUDIO):
            try:
                await self.room.add_subscription(self.candidate_identity, kind)
            except Exception as e:
                print(f"[PROCTOR] add_subscription({kind}) failed: {e}")

    def _on_track_subscribed(self, track, publication, participant):
        try:
            if participant is None or track is None:
                return
            if participant.identity != self.candidate_identity:
                # Agent voice track — ignore (not the candidate).
                return
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                track.add_listener(self._on_candidate_video)
        except Exception as e:
            print(f"[PROCTOR] track_subscribed error: {e}")

    def _on_candidate_video(self, event):
        try:
            if self._processing:
                return
            frame = getattr(event, "frame", None)
            if frame is None:
                return
            bgr = self._frame_to_bgr(frame)
            if bgr is None:
                return
            self._processing = True
            asyncio.create_task(self._process_video(bgr))
        except Exception as e:
            self._processing = False
            print(f"[PROCTOR] video error: {e}")

    async def _process_video(self, bgr):
        try:
            session = self._session
            if session is None:
                return

            ok, enc = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            if not ok:
                return
            jpeg = enc.tobytes()
            latest_frames[self.session_id] = jpeg

            result, trigger = await asyncio.to_thread(
                process_frame, bgr, session, self._face_det, self._face_mesh
            )
            if result is None:
                return
            sessions[self.session_id] = session
            self._maybe_publish_status(result)
            if trigger:
                await self._publish_trigger(trigger)
        except Exception as e:
            print(f"[PROCTOR] process error: {e}")
        finally:
            self._processing = False

    def _maybe_publish_status(self, result):
        now = time.time()
        if now - self._last_status_time < self._status_min_interval:
            return
        self._last_status_time = now
        asyncio.create_task(self._publish_status(result))

    async def _publish_status(self, result):
        try:
            payload = json.dumps({"type": "cheating_status", **result}).encode("utf-8")
            await self.room.local_participant.publish_data(payload, reliable=True)
        except Exception as e:
            print(f"[PROCTOR] publish status error: {e}")

    async def _publish_trigger(self, trigger):
        try:
            payload = json.dumps({"type": "trigger", **trigger}).encode("utf-8")
            await self.room.local_participant.publish_data(payload, reliable=True)
        except Exception as e:
            print(f"[PROCTOR] publish trigger error: {e}")

    @staticmethod
    def _frame_to_bgr(frame):
        w, h = frame.width, frame.height
        data = np.frombuffer(frame.data, dtype=np.uint8)
        i420_len = w * h * 3 // 2
        if len(data) == i420_len:
            yuv = data.reshape((h * 3 // 2, w))
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        rgba_len = w * h * 4
        if len(data) == rgba_len:
            rgba = data.reshape((h, w, 4))
            return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        bgra = data.reshape((h, w, 4))
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

    async def stop(self):
        try:
            self._face_det.close()
            self._face_mesh.close()
            await self.room.disconnect()
        except Exception as e:
            print(f"[PROCTOR] disconnect error: {e}")


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

    def _log(v_type, hr_text, ai_warning, snap_name=None):
        nonlocal trigger
        reached = session.tracker.log_violation(v_type, formatted_time, hr_text, ai_warning, bgr)
        if snap_name:
            snap_path = os.path.join(EVIDENCE_DIR, session.session_id, snap_name)
            os.makedirs(os.path.dirname(snap_path), exist_ok=True)
            cv2.imwrite(snap_path, bgr)
            session.add_event("screenshot", os.path.basename(snap_path))
        if reached:
            trigger = {
                "violation": v_type,
                "aiWarning": ai_warning,
                "hrMessage": hr_text,
                "timestamp": formatted_time,
                "suspicion_score": round(session.suspicion_score, 1),
            }

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
                            "Verification failed, candidate must be present",
                            f"identity_mismatch_{int(now)}.jpg",
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
            "Please adjust lighting to face the camera in normal light",
            f"light_violation_{int(now)}.jpg",
        )

    # ── People count ─────────────────────────────────────────────────────
    if face_count > 1:
        result["flags"].append("multiple_faces")
        session.bump_score(5)
        session.add_event("multi_face", f"{face_count} faces detected")
        if now - session.last_snap_time > 3.0:
            session.last_snap_time = now
            _log(
                "Multiple People Detected",
                f"Multiple people detected ({face_count} people)",
                "Only candidate is allowed in frame, clear other persons",
                f"violation_{int(now)}.jpg",
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
            session.bump_score(2)
            session.add_event("face_gone", f"{gone_dur:.1f}s")
            if now - session.last_missing_face_snap_time > 3.0:
                session.last_missing_face_snap_time = now
                _log(
                    "Missing Candidate",
                    "Candidate missing from frame",
                    "Please stay in front of the camera at all times",
                    f"missing_candidate_{int(now)}.jpg",
                )
    else:
        session.face_gone_start = None

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
                        "Please remove dark glasses, eyes must be clearly visible",
                        f"glasses_violation_{int(now)}.jpg",
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
                            "Please keep your eyes open and stay attentive",
                            f"eye_closed_{int(now)}.jpg",
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
                        "No Blink (>8s)",
                        "No natural eye blink detected for over 8 seconds",
                        "Please blink naturally to maintain liveliness check",
                        f"no_blink_{int(now)}.jpg",
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
                                "Please maintain eye contact with the screen center",
                                f"gaze_violation_{int(now)}.jpg",
                            )
                else:
                    session.iris_streak_direction = None
                    session.iris_streak_count = 0

    session.suspicion_score = min(session.suspicion_score, 100)
    result["suspicion_score"] = round(session.suspicion_score, 1)
    return result, trigger


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

    return web.json_response({
        "sessionId": session_id,
        "suspicion_score": round(session.suspicion_score, 1),
        "events": session.events,
        "total_events": len(session.events),
        "violations": list(session.tracker.violations.keys()),
    })


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


# ── Routes ───────────────────────────────────────────────────────────────────
app.router.add_post('/register-face', register_face)
app.router.add_get('/report/{sessionId}', get_report)
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
