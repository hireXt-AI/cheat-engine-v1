# import cv2
# import mediapipe as mp
# import face_recognition
# import time
# import os
# import numpy as np
# from pathlib import Path
# import threading
# import wave
# import subprocess
# import csv
# import uuid


# # PyAudio import with fallback handling
# try:
#     import pyaudio
#     HAS_PYAUDIO = True
# except ImportError:
#     HAS_PYAUDIO = False
#     print("⚠️ 'pyaudio' is not installed. Audio recording will be disabled.")
#     print("   To enable audio, install pyaudio (e.g. `pip install pyaudio`).")

# # ─────────────────────────────────────────────────────────────────────────────
# # Audio Recording Helper Thread
# # ─────────────────────────────────────────────────────────────────────────────

# class AudioRecorderThread(threading.Thread):
#     def __init__(self, output_wav_path: Path, sample_rate=44100, channels=1, chunk=1024):
#         super().__init__()
#         self.output_wav_path = output_wav_path
#         self.sample_rate = sample_rate
#         self.channels = channels
#         self.chunk = chunk
#         self.format = pyaudio.paInt16 if HAS_PYAUDIO else None
#         self.is_recording = False
#         self._frames = []

#     def run(self):
#         if not HAS_PYAUDIO:
#             return

#         p = pyaudio.PyAudio()
#         try:
#             stream = p.open(
#                 format=self.format,
#                 channels=self.channels,
#                 rate=self.sample_rate,
#                 input=True,
#                 frames_per_buffer=self.chunk
#             )
#         except Exception as e:
#             print(f"⚠️ Could not open microphone for audio recording: {e}")
#             p.terminate()
#             return

#         self.is_recording = True
#         self._frames = []

#         while self.is_recording:
#             try:
#                 data = stream.read(self.chunk, exception_on_overflow=False)
#                 self._frames.append(data)
#             except Exception:
#                 break

#         stream.stop_stream()
#         stream.close()
#         p.terminate()

#         # Save to WAV file
#         if self._frames:
#             wf = wave.open(str(self.output_wav_path), 'wb')
#             wf.setnchannels(self.channels)
#             wf.setsampwidth(p.get_sample_size(self.format))
#             wf.setframerate(self.sample_rate)
#             wf.writeframes(b''.join(self._frames))
#             wf.close()

#     def stop(self):
#         self.is_recording = False

# def combine_audio_video(video_path: Path, audio_path: Path, output_path: Path, actual_fps: float = 20.0):
#     """Muxes audio and video together using ffmpeg with robust fallbacks."""
#     if not audio_path.exists() or audio_path.stat().st_size == 0:
#         print("⚠️ Audio track empty or missing. Keeping video file as-is.")
#         return

#     cmd = [
#         "ffmpeg",
#         "-y",
#         "-i", str(video_path),
#         "-i", str(audio_path),
#         "-c:v", "copy",
#         "-c:a", "aac",
#         "-map", "0:v:0",
#         "-map", "1:a:0",
#         "-shortest",
#         str(output_path)
#     ]

#     try:
#         subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
#         if output_path.exists() and output_path.stat().st_size > 0:
#             video_path.unlink(missing_ok=True)
#             audio_path.unlink(missing_ok=True)
#             print(f"✅ Synchronized video + audio saved to: {output_path}")
#             return
#     except Exception as e:
#         print(f"⚠️ Primary FFmpeg muxing failed ({e}). Trying full re-encode...")

#     cmd_reencode = [
#         "ffmpeg",
#         "-y",
#         "-r", f"{actual_fps:.2f}",
#         "-i", str(video_path),
#         "-i", str(audio_path),
#         "-c:v", "libx264",
#         "-pix_fmt", "yuv420p",
#         "-c:a", "aac",
#         "-async", "1",
#         "-shortest",
#         str(output_path)
#     ]

#     try:
#         subprocess.run(cmd_reencode, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
#         if output_path.exists() and output_path.stat().st_size > 0:
#             video_path.unlink(missing_ok=True)
#             audio_path.unlink(missing_ok=True)
#             print(f"✅ Synchronized video + audio saved to: {output_path}")
#     except Exception as e:
#         print(f"❌ FFmpeg execution failed. Ensure FFmpeg is installed and added to PATH.")

# # ─────────────────────────────────────────────────────────────────────────────
# # Eye Aspect Ratio (EAR) & Blink Helper Functions
# # ─────────────────────────────────────────────────────────────────────────────

# def calculate_ear(landmarks, frame_width, frame_height):
#     """Calculates Eye Aspect Ratio (EAR) for both eyes to detect closures."""
#     def get_pt(idx):
#         return np.array([landmarks[idx].x * frame_width, landmarks[idx].y * frame_height])

#     def single_eye_ear(p_top1, p_bot1, p_top2, p_bot2, p_left, p_right):
#         v1 = np.linalg.norm(p_top1 - p_bot1)
#         v2 = np.linalg.norm(p_top2 - p_bot2)
#         h = np.linalg.norm(p_left - p_right)
#         return (v1 + v2) / (2.0 * h) if h > 0 else 0.3

#     l_ear = single_eye_ear(get_pt(159), get_pt(145), get_pt(158), get_pt(153), get_pt(33), get_pt(133))
#     r_ear = single_eye_ear(get_pt(386), get_pt(374), get_pt(385), get_pt(380), get_pt(362), get_pt(263))

#     return (l_ear + r_ear) / 2.0

# # ─────────────────────────────────────────────────────────────────────────────
# # Lighting & Brightness Monitoring Helper
# # ─────────────────────────────────────────────────────────────────────────────

# MIN_BRIGHTNESS_PCT = 30.0
# MAX_BRIGHTNESS_PCT = 80.0

# def analyze_lighting_conditions(frame, face_box=None):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     h, w = gray.shape

#     if face_box is None:
#         overall_avg = np.mean(gray)
#         overall_pct = (overall_avg / 255.0) * 100.0
#         return None, overall_pct

#     x, y, bw, bh = face_box
#     x1, y1 = max(0, x), max(0, y)
#     x2, y2 = min(w, x + bw), min(h, y + bh)

#     face_region = gray[y1:y2, x1:x2]
#     face_brightness_pct = ((np.mean(face_region) / 255.0) * 100.0) if face_region.size > 0 else None

#     bg_mask = np.ones((h, w), dtype=bool)
#     bg_mask[y1:y2, x1:x2] = False
#     bg_pixels = gray[bg_mask]
#     bg_brightness_pct = ((np.mean(bg_pixels) / 255.0) * 100.0) if bg_pixels.size > 0 else None

#     return face_brightness_pct, bg_brightness_pct

# # ─────────────────────────────────────────────────────────────────────────────
# # Black Glasses Detection Helper Function
# # ─────────────────────────────────────────────────────────────────────────────

# def detect_black_glasses(landmarks, frame, frame_width, frame_height):
#     """
#     Detection using adaptive face skin baseline vs eye region luminance,
#     dark pixel density, and contrast variation.
#     """
#     LEFT_EYE_INDICES = [33, 133, 159, 145, 158, 153, 144, 160]
#     RIGHT_EYE_INDICES = [362, 263, 386, 374, 385, 380, 373, 387]
#     SKIN_SAMPLING_INDICES = [10, 109, 338, 205, 425]  # Forehead + Cheeks

#     def get_coords(idx_list):
#         pts = []
#         for idx in idx_list:
#             pts.append([int(landmarks[idx].x * frame_width), int(landmarks[idx].y * frame_height)])
#         return np.array(pts, dtype=np.int32)

#     l_pts = get_coords(LEFT_EYE_INDICES)
#     r_pts = get_coords(RIGHT_EYE_INDICES)
#     skin_pts = get_coords(SKIN_SAMPLING_INDICES)

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     skin_values = []
#     for pt in skin_pts:
#         px = min(frame_width - 1, max(0, pt[0]))
#         py = min(frame_height - 1, max(0, pt[1]))
#         skin_values.append(gray[py, px])
    
#     skin_baseline = float(np.mean(skin_values)) if skin_values else 120.0
#     skin_baseline = max(skin_baseline, 30.0)

#     def analyze_eye_region(pts, expand=12):
#         x, y, w, h = cv2.boundingRect(pts)
#         x1, y1 = max(0, x - expand), max(0, y - expand)
#         x2, y2 = min(frame_width, x + w + expand), min(frame_height, y + h + expand)

#         region = gray[y1:y2, x1:x2]
#         if region.size == 0:
#             return 255.0, 0.0

#         mean_brightness = np.mean(region)
#         dark_thresh = max(40, skin_baseline * 0.45)
#         dark_pixels = np.sum(region < dark_thresh)
#         dark_ratio = float(dark_pixels) / float(region.size)

#         return mean_brightness, dark_ratio

#     l_bright, l_dark_ratio = analyze_eye_region(l_pts)
#     r_bright, r_dark_ratio = analyze_eye_region(r_pts)

#     avg_eye_brightness = (l_bright + r_bright) / 2.0
#     avg_dark_pixel_ratio = (l_dark_ratio + r_dark_ratio) / 2.0
#     brightness_ratio = avg_eye_brightness / skin_baseline

#     is_glasses = (avg_dark_pixel_ratio >= 0.40) and (brightness_ratio < 0.45)
#     return is_glasses

# # ─────────────────────────────────────────────────────────────────────────────
# # Iris / Gaze Direction Helper Function
# # ─────────────────────────────────────────────────────────────────────────────

# def detect_iris_direction(landmarks, frame_width, frame_height):
#     LEFT_IRIS_CENTER = 468
#     LEFT_EYE_INNER   = 133
#     LEFT_EYE_OUTER   = 33
#     LEFT_EYE_TOP     = 159
#     LEFT_EYE_BOTTOM  = 145

#     RIGHT_IRIS_CENTER = 473
#     RIGHT_EYE_INNER   = 362
#     RIGHT_EYE_OUTER   = 263
#     RIGHT_EYE_TOP     = 386
#     RIGHT_EYE_BOTTOM  = 374

#     def get_pt(idx):
#         return np.array([landmarks[idx].x * frame_width, landmarks[idx].y * frame_height])

#     l_iris = get_pt(LEFT_IRIS_CENTER)
#     l_inner, l_outer = get_pt(LEFT_EYE_INNER), get_pt(LEFT_EYE_OUTER)
#     l_top, l_bottom   = get_pt(LEFT_EYE_TOP), get_pt(LEFT_EYE_BOTTOM)

#     l_horiz_dist = np.linalg.norm(l_outer - l_inner)
#     l_vert_dist  = np.linalg.norm(l_bottom - l_top)

#     l_h_ratio = (np.linalg.norm(l_iris - l_inner) / l_horiz_dist) if l_horiz_dist > 0 else 0.5
#     l_v_ratio = (np.linalg.norm(l_iris - l_top) / l_vert_dist) if l_vert_dist > 0 else 0.5

#     r_iris = get_pt(RIGHT_IRIS_CENTER)
#     r_inner, r_outer = get_pt(RIGHT_EYE_INNER), get_pt(RIGHT_EYE_OUTER)
#     r_top, r_bottom   = get_pt(RIGHT_EYE_TOP), get_pt(RIGHT_EYE_BOTTOM)

#     r_horiz_dist = np.linalg.norm(r_outer - r_inner)
#     r_vert_dist  = np.linalg.norm(r_bottom - r_top)

#     r_h_ratio = (np.linalg.norm(r_iris - r_inner) / r_horiz_dist) if r_horiz_dist > 0 else 0.5
#     r_v_ratio = (np.linalg.norm(r_iris - r_top) / r_vert_dist) if r_vert_dist > 0 else 0.5

#     avg_h_ratio = (l_h_ratio + r_h_ratio) / 2.0
#     avg_v_ratio = (l_v_ratio + r_v_ratio) / 2.0

#     direction = None
#     if avg_h_ratio < 0.42:
#         direction = "IRIS LEFT"
#     elif avg_h_ratio > 0.58:
#         direction = "IRIS RIGHT"
#     elif avg_v_ratio < 0.35:
#         direction = "IRIS UP"
#     elif avg_v_ratio > 0.65:
#         direction = "IRIS DOWN"

#     right_iris_pixel = (int(r_iris[0]), int(r_iris[1]))
#     left_iris_pixel  = (int(l_iris[0]), int(l_iris[1]))

#     return direction, right_iris_pixel, left_iris_pixel

# # ─────────────────────────────────────────────────────────────────────────────
# # In-Memory Violation Tracker (Separate Tracking + 12 Occurrence Trigger Point)
# # ─────────────────────────────────────────────────────────────────────────────

# TRIGGER_CSV_HEADER = ["User_ID", "Violation", "Timestamps", "HR Message", "AI Message"]

# class ViolationTracker:
#     def __init__(self, csv_path: Path, trigger_snapshot_dir: Path, user_id: str, threshold: int = 12):
#         self.csv_path = Path(csv_path)
#         self.trigger_snapshot_dir = Path(trigger_snapshot_dir)
#         self.trigger_snapshot_dir.mkdir(parents=True, exist_ok=True)
#         self.user_id = user_id
#         self.threshold = threshold
#         self.violations = {}
#         self._ensure_csv_header()

#     def _ensure_csv_header(self):
#         if not self.csv_path.exists():
#             self.csv_path.parent.mkdir(parents=True, exist_ok=True)
#             with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
#                 writer = csv.writer(f)
#                 writer.writerow(TRIGGER_CSV_HEADER)

#     def log_violation(self, violation_type: str, timestamp_str: str, hr_text: str, ai_warning: str, frame=None) -> bool:
#         """
#         Buffers violation timestamps separately per violation type.
#         When 12 total occurrences are reached:
#           - Saves a Trigger Point snapshot into Trigger Point folder
#           - Writes all 12 timestamps + HR Message + AI Message to CSV
#           - Resets the count/timestamps for this violation type
#         """
#         if violation_type not in self.violations:
#             self.violations[violation_type] = {
#                 "timestamps": [],
#                 "hr_text": hr_text,
#                 "ai_warning": ai_warning
#             }

#         v = self.violations[violation_type]
#         v["timestamps"].append(timestamp_str)
#         v["hr_text"] = hr_text
#         v["ai_warning"] = ai_warning

#         count = len(v["timestamps"])

#         if count >= self.threshold:
#             # 12 total occurrences reached -> Trigger Point
#             timestamps_joined = ", ".join(v["timestamps"])

#             # Save Trigger Point snapshot
#             if frame is not None:
#                 clean_name = violation_type.lower().replace(" ", "_").replace("/", "_")
#                 snap_name = f"trigger_point_{clean_name}_{int(time.time())}.jpg"
#                 snap_path = self.trigger_snapshot_dir / snap_name
#                 cv2.imwrite(str(snap_path), frame)

#             # Write to CSV
#             with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
#                 writer = csv.writer(f)
#                 writer.writerow([self.user_id, violation_type, timestamps_joined, hr_text, ai_warning])

#             print(f"🚨 [TRIGGER POINT REACHED - 12 OCCURRENCES] | Type: '{violation_type}' | Timestamps: {timestamps_joined}")

#             # Reset memory buffer for this violation
#             v["timestamps"] = []
#             return True
#         else:
#             print(f"⚠️ [VIOLATION BUFFERED ({count}/{self.threshold})] | Type: '{violation_type}' @ {timestamp_str}")
#             return False

# # ─────────────────────────────────────────────────────────────────────────────
# # Candidate Selection & Encoding
# # ─────────────────────────────────────────────────────────────────────────────

# IMAGES_DIR = Path("images")
# SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# def list_candidates(images_dir: Path) -> list[Path]:
#     if not images_dir.exists():
#         return []
#     return sorted(
#         p for p in images_dir.iterdir()
#         if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
#     )

# def select_candidate() -> Path | None:
#     candidates = list_candidates(IMAGES_DIR)

#     if not candidates:
#         print(f"\n❌ No images found in '{IMAGES_DIR}/'.")
#         print(f"   Add candidate photos there and try again.")
#         return None

#     print("\n" + "═" * 50)
#     print("  📋 CANDIDATE SELECTION")
#     print("═" * 50)
#     for idx, path in enumerate(candidates, start=1):
#         name = path.stem.replace("_", " ").replace("-", " ").title()
#         print(f"  [{idx}]  {name}  ({path.name})")
#     print("  [q]  Quit")
#     print("═" * 50)

#     while True:
#         raw = input("Select candidate number: ").strip().lower()
#         if raw == "q":
#             return None
#         if raw.isdigit():
#             choice = int(raw)
#             if 1 <= choice <= len(candidates):
#                 selected = candidates[choice - 1]
#                 name = selected.stem.replace("_", " ").replace("-", " ").title()
#                 print(f"\n✅ Selected: {name} → {selected}")
#                 return selected
#         print(f"   ⚠️ Enter a number between 1 and {len(candidates)}, or 'q'.")

# def load_reference_encoding(reference_image_path: Path):
#     if not reference_image_path.exists():
#         print(f"Error: Reference image not found at '{reference_image_path}'.")
#         return None

#     reference_image = face_recognition.load_image_file(str(reference_image_path))
#     face_locations = face_recognition.face_locations(reference_image)

#     if len(face_locations) == 0:
#         print("Error: No face detected in the reference image.")
#         return None
#     if len(face_locations) > 1:
#         print("Error: Reference image has more than one face.")
#         return None

#     return face_recognition.face_encodings(reference_image, known_face_locations=face_locations)[0]

# # ─────────────────────────────────────────────────────────────────────────────
# # Main Interview Session
# # ─────────────────────────────────────────────────────────────────────────────

# def start_interview_recording(reference_image_path: Path, match_tolerance=0.6, verify_every_seconds=2.0):
#     candidate_name = reference_image_path.stem.replace("_", " ").replace("-", " ").title()
#     safe_name = reference_image_path.stem.replace(" ", "_")

#     print(f"\n🔍 Encoding reference photo for: {candidate_name} ...")
#     reference_encoding = load_reference_encoding(reference_image_path)
#     if reference_encoding is None:
#         print("Aborting: fix the reference image and try again.")
#         return
#     print(f"✅ Reference photo loaded successfully.")

#     # 1. Directories & Tracking setup
#     video_dir = Path("video_interview")
#     snapshot_dir = video_dir / "snapshots"
#     trigger_snapshot_dir = video_dir / "trigger_points_snapshots"

#     video_dir.mkdir(parents=True, exist_ok=True)
#     snapshot_dir.mkdir(parents=True, exist_ok=True)
#     trigger_snapshot_dir.mkdir(parents=True, exist_ok=True)

#     user_id = f"USER_{uuid.uuid4().hex[:6].upper()}"
#     trigger_csv_path = video_dir / f"trigger_points_{safe_name}_{user_id}.csv"
    
#     # 12-occurrence threshold for Trigger Points
#     tracker = ViolationTracker(
#         csv_path=trigger_csv_path,
#         trigger_snapshot_dir=trigger_snapshot_dir,
#         user_id=user_id,
#         threshold=12
#     )

#     # 2. Camera Setup
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return

#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     timestamp     = time.strftime('%Y%m%d_%H%M%S')
#     raw_video_path = video_dir / f"interview_{safe_name}_{timestamp}_raw.mp4"
#     audio_path     = video_dir / f"interview_{safe_name}_{timestamp}.wav"
#     final_video_path = video_dir / f"interview_{safe_name}_{timestamp}.mp4"

#     target_fps = 20.0
#     writer = cv2.VideoWriter(str(raw_video_path), cv2.VideoWriter_fourcc(*'mp4v'), target_fps, (width, height))

#     audio_recorder = AudioRecorderThread(output_wav_path=audio_path)
#     audio_recorder.start()

#     # 3. MediaPipe Setup
#     mp_face = mp.solutions.face_detection
#     mp_face_mesh = mp.solutions.face_mesh

#     # Snapshot Cooldown Timers
#     last_verify_time = 0
#     last_snap_time = 0
#     last_movement_snap_time = 0
#     last_glasses_snap_time = 0
#     last_light_snap_time = 0
#     last_eye_closure_snap_time = 0
#     last_no_blink_snap_time = 0
#     last_trigger_snap_time = 0
#     last_missing_face_snap_time = 0
#     last_mismatch_snap_time = 0

#     EAR_CLOSED_THRESHOLD = 0.18
#     eye_closed_start_time = None

#     NO_BLINK_WARN_SECONDS = 8.0
#     last_blink_timestamp = time.time()

#     glasses_consecutive_frames = 0
#     REQUIRED_GLASSES_FRAMES = 3

#     EYE_OPEN_FOR_GAZE_EAR = 0.23
#     IRIS_STREAK_TRIGGER = 6
#     iris_streak_direction = None
#     iris_streak_count = 0

#     identity_ok = None
#     font = cv2.FONT_HERSHEY_SIMPLEX

#     frame_count = 0
#     start_time = time.time()

#     print(f"\n🎥 Recording to: {final_video_path}")
#     print(f"   Candidate : {candidate_name}")
#     print(f"   User ID   : {user_id}")
#     print(f"   CSV Log   : {trigger_csv_path}")
#     print("   Press 'q' in the video window to stop.\n")

#     try:
#         with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_det, \
#              mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:

#             while cap.isOpened():
#                 success, frame = cap.read()
#                 if not success:
#                     break

#                 frame_count += 1
#                 current_timestamp = time.time()
#                 elapsed_sec = current_timestamp - start_time
#                 formatted_time = time.strftime("%H:%M:%S", time.localtime(current_timestamp))

#                 rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results    = face_det.process(rgb_frame)
#                 face_count = len(results.detections) if results.detections else 0

#                 face_box = None

#                 # ── Primary Verification Loop ─────────────────────────────
#                 if face_count > 0:
#                     det = results.detections[0]
#                     box = det.location_data.relative_bounding_box
#                     x, y = int(box.xmin * width), int(box.ymin * height)
#                     w_box, h_box = int(box.width * width), int(box.height * height)
#                     face_box = (x, y, w_box, h_box)

#                     if current_timestamp - last_verify_time > verify_every_seconds:
#                         last_verify_time = current_timestamp
#                         top, left     = max(0, y), max(0, x)
#                         bottom, right = min(height, y + h_box), min(width, x + w_box)

#                         try:
#                             encodings = face_recognition.face_encodings(
#                                 rgb_frame,
#                                 known_face_locations=[(top, right, bottom, left)]
#                             )
#                             if encodings:
#                                 distance = face_recognition.face_distance([reference_encoding], encodings[0])[0]
#                                 identity_ok = bool(distance <= match_tolerance)
#                                 if not identity_ok and (current_timestamp - last_mismatch_snap_time > 3.0):
#                                     v_type = "Identity Mismatch"
#                                     hr_msg = "Unrecognized person in candidate position"
#                                     ai_warn = "Verification failed, candidate must be present"
                                    
#                                     tracker.log_violation(v_type, formatted_time, hr_msg, ai_warn, frame)
                                    
#                                     # Standard snapshot
#                                     snap_path = snapshot_dir / f"identity_mismatch_{int(current_timestamp)}.jpg"
#                                     cv2.imwrite(str(snap_path), frame)
#                                     last_mismatch_snap_time = current_timestamp
#                         except Exception as e:
#                             print(f"⚠️ Verification skipped: {e}")

#                 # ── Bounding Boxes ─────────────────────────────────────────
#                 if results.detections:
#                     for idx, det_item in enumerate(results.detections):
#                         b = det_item.location_data.relative_bounding_box
#                         bx, by = int(b.xmin * width), int(b.ymin * height)
#                         bw, bh = int(b.width * width), int(b.height * height)
                        
#                         if idx == 0 and identity_ok is True:
#                             color = (0, 255, 0)
#                             label = f"Candidate: {candidate_name}"
#                         else:
#                             color = (0, 0, 255)
#                             label = "Unknown / Other Person"

#                         cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)
#                         cv2.putText(frame, label, (bx, max(20, by - 10)), font, 0.6, color, 2)

#                 # ── Light / Brightness Monitoring ─────────────────────────
#                 face_pct, bg_pct = analyze_lighting_conditions(frame, face_box)

#                 hud_y = 30
#                 if face_pct is not None:
#                     cv2.putText(frame, f"Face Light: {face_pct:.1f}%", (width - 230, hud_y), font, 0.6, (255, 255, 255), 2)
#                     hud_y += 25
#                 if bg_pct is not None:
#                     cv2.putText(frame, f"BG Light:   {bg_pct:.1f}%", (width - 230, hud_y), font, 0.6, (255, 255, 255), 2)

#                 lighting_warning = None
#                 if face_pct is not None:
#                     if face_pct < MIN_BRIGHTNESS_PCT:
#                         lighting_warning = "WARN: FRONT LIGHT LOW"
#                     elif face_pct > MAX_BRIGHTNESS_PCT:
#                         lighting_warning = "WARN: FRONT LIGHT TOO HIGH - USE NORMAL LIGHT"

#                 if not lighting_warning and bg_pct is not None:
#                     if bg_pct < MIN_BRIGHTNESS_PCT:
#                         lighting_warning = "WARN: BACKGROUND LIGHT IS LOW"
#                     elif bg_pct > MAX_BRIGHTNESS_PCT:
#                         lighting_warning = "WARN: BACKGROUND LIGHT HIGH - SIT IN NORMAL LIGHT"

#                 if lighting_warning:
#                     cv2.putText(frame, lighting_warning, (20, 210), font, 0.75, (0, 165, 255), 2)
#                     if current_timestamp - last_light_snap_time > 3.0:
#                         v_type = f"Lighting Violation ({lighting_warning.replace('WARN: ', '')})"
#                         tracker.log_violation(
#                             v_type, formatted_time,
#                             f"Lighting Violation: {lighting_warning}",
#                             "Please adjust lighting to face the camera in normal light",
#                             frame
#                         )
#                         # Standard snapshot
#                         snap_path = snapshot_dir / f"light_violation_{int(current_timestamp)}.jpg"
#                         cv2.imwrite(str(snap_path), frame)
#                         last_light_snap_time = current_timestamp

#                 # ── People Counter Logic ──────────────────────────────────
#                 if face_count > 1:
#                     cv2.putText(frame, f"WARNING: {face_count} PEOPLE!", (20, 50), font, 1, (0, 0, 255), 3)
#                     cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)

#                     if current_timestamp - last_snap_time > 3.0:
#                         v_type = "Multiple People Detected"
#                         tracker.log_violation(
#                             v_type, formatted_time,
#                             f"Multiple people detected ({face_count} people)",
#                             "Only candidate is allowed in frame, clear other persons",
#                             frame
#                         )
#                         # Standard snapshot
#                         snap_path = snapshot_dir / f"violation_{int(current_timestamp)}.jpg"
#                         cv2.imwrite(str(snap_path), frame)
#                         last_snap_time = current_timestamp

#                 elif face_count == 0:
#                     cv2.putText(frame, "WARNING: NO CANDIDATE", (20, 50), font, 1, (0, 165, 255), 3)
#                     identity_ok = None
#                     glasses_consecutive_frames = 0
#                     eye_closed_start_time = None
#                     last_blink_timestamp = current_timestamp
#                     iris_streak_direction = None
#                     iris_streak_count = 0

#                     if current_timestamp - last_missing_face_snap_time > 3.0:
#                         v_type = "Missing Candidate"
#                         tracker.log_violation(
#                             v_type, formatted_time,
#                             "Candidate missing from frame",
#                             "Please stay in front of the camera at all times",
#                             frame
#                         )
#                         # Standard snapshot
#                         snap_path = snapshot_dir / f"missing_candidate_{int(current_timestamp)}.jpg"
#                         cv2.imwrite(str(snap_path), frame)
#                         last_missing_face_snap_time = current_timestamp

#                 else:
#                     det = results.detections[0]

#                     # ── Head Movement Detection (Separated by direction) ─────
#                     keypoints = det.location_data.relative_keypoints
#                     rx, ry = int(keypoints[0].x * width), int(keypoints[0].y * height)
#                     lx, ly = int(keypoints[1].x * width), int(keypoints[1].y * height)
#                     nx, ny = int(keypoints[2].x * width), int(keypoints[2].y * height)
#                     mx, my = int(keypoints[3].x * width), int(keypoints[3].y * height)

#                     eye_center_x = (rx + lx) / 2.0
#                     eye_dist = abs(lx - rx)
#                     yaw_ratio = (nx - eye_center_x) / eye_dist if eye_dist > 0 else 0

#                     eye_center_y = (ry + ly) / 2.0
#                     face_length = my - eye_center_y
#                     pitch_ratio = (ny - eye_center_y) / face_length if face_length > 0 else 0.5

#                     movement_state = None
#                     if yaw_ratio < -0.20:
#                         movement_state = "LOOKING RIGHT"
#                     elif yaw_ratio > 0.20:
#                         movement_state = "LOOKING LEFT"
#                     elif pitch_ratio < 0.26:
#                         movement_state = "LOOKING UP"
#                     elif pitch_ratio > 0.72:
#                         movement_state = "LOOKING DOWN"

#                     if movement_state:
#                         cv2.putText(frame, f"WARNING: {movement_state}", (20, 90), font, 1, (0, 165, 255), 3)
#                         if current_timestamp - last_movement_snap_time > 3.0:
#                             v_type = movement_state.title()  # e.g., "Looking Right", "Looking Left"
#                             short_dir = movement_state.replace("LOOKING ", "").lower()
                            
#                             tracker.log_violation(
#                                 v_type, formatted_time,
#                                 f"Head turned {short_dir}",
#                                 f"Please stay focused on screen, do not look {short_dir}",
#                                 frame
#                             )
#                             # Standard snapshot
#                             snap_path = snapshot_dir / f"movement_{short_dir}_{int(current_timestamp)}.jpg"
#                             cv2.imwrite(str(snap_path), frame)
#                             last_movement_snap_time = current_timestamp

#                     # ── Black Glasses, Iris & Eye Tracking ──────────
#                     mesh_results = face_mesh.process(rgb_frame)
#                     if mesh_results.multi_face_landmarks:
#                         face_landmarks = mesh_results.multi_face_landmarks[0]

#                         # ── Black Glasses Check ──────────
#                         is_wearing_glasses = detect_black_glasses(face_landmarks.landmark, frame, width, height)
#                         if is_wearing_glasses:
#                             glasses_consecutive_frames += 1
#                         else:
#                             glasses_consecutive_frames = max(0, glasses_consecutive_frames - 1)

#                         if glasses_consecutive_frames >= REQUIRED_GLASSES_FRAMES:
#                             cv2.putText(frame, "WARN: EYE NOT VISIBLE", (20, 170), font, 1, (0, 0, 255), 3)
#                             if current_timestamp - last_glasses_snap_time > 3.0:
#                                 v_type = "Eye Obstruction"
#                                 tracker.log_violation(
#                                     v_type, formatted_time,
#                                     "Dark sunglasses or covered eyes detected",
#                                     "Please remove dark glasses, eyes must be clearly visible",
#                                     frame
#                                 )
#                                 # Standard snapshot
#                                 snap_path = snapshot_dir / f"glasses_violation_{int(current_timestamp)}.jpg"
#                                 cv2.imwrite(str(snap_path), frame)
#                                 last_glasses_snap_time = current_timestamp

#                         ear = calculate_ear(face_landmarks.landmark, width, height)

#                         # ── Eye Closed Warning (> 5 seconds) ─────────
#                         if ear < EAR_CLOSED_THRESHOLD:
#                             if eye_closed_start_time is None:
#                                 eye_closed_start_time = current_timestamp
#                                 last_blink_timestamp = current_timestamp
                            
#                             closed_duration = current_timestamp - eye_closed_start_time
                            
#                             if closed_duration >= 5.0:
#                                 cv2.putText(frame, "WARN: EYE CLOSED (>5s)", (20, 250), font, 0.9, (0, 0, 255), 3)

#                                 if current_timestamp - last_eye_closure_snap_time > 3.0:
#                                     v_type = "Eye Closure (>5s)"
#                                     tracker.log_violation(
#                                         v_type, formatted_time,
#                                         "Candidate eyes closed continuously for over 5 seconds",
#                                         "Please keep your eyes open and stay attentive",
#                                         frame
#                                     )
#                                     # Standard snapshot
#                                     snap_path = snapshot_dir / f"eye_closed_{int(current_timestamp)}.jpg"
#                                     cv2.imwrite(str(snap_path), frame)
#                                     last_eye_closure_snap_time = current_timestamp
#                         else:
#                             eye_closed_start_time = None

#                         # ── Liveliness (No-Blink Warning, 8s) ────────
#                         time_since_last_blink = current_timestamp - last_blink_timestamp
#                         if time_since_last_blink >= NO_BLINK_WARN_SECONDS:
#                             cv2.putText(frame, "WARN: PERSON NOT BLINKING (>8s)", (20, 290), font, 0.9, (0, 0, 255), 3)
#                             cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 4)
#                             if current_timestamp - last_no_blink_snap_time > 3.0:
#                                 v_type = "No Blink (>8s)"
#                                 tracker.log_violation(
#                                     v_type, formatted_time,
#                                     "No natural eye blink detected for over 8 seconds",
#                                     "Please blink naturally to maintain liveliness check",
#                                     frame
#                                 )
#                                 # Standard snapshot
#                                 snap_path = snapshot_dir / f"no_blink_{int(current_timestamp)}.jpg"
#                                 cv2.imwrite(str(snap_path), frame)
#                                 last_no_blink_snap_time = current_timestamp

#                         if ear < EAR_CLOSED_THRESHOLD:
#                             last_blink_timestamp = current_timestamp

#                         # ── Gaze & Iris Tracking (Separated by Iris Direction) ───────
#                         if ear >= EYE_OPEN_FOR_GAZE_EAR:
#                             iris_dir, r_iris_pt, l_iris_pt = detect_iris_direction(
#                                 face_landmarks.landmark, width, height
#                             )
#                             cv2.circle(frame, r_iris_pt, 3, (0, 255, 255), -1)
#                             cv2.circle(frame, l_iris_pt, 3, (0, 255, 255), -1)

#                             if iris_dir:
#                                 if iris_dir == iris_streak_direction:
#                                     iris_streak_count += 1
#                                 else:
#                                     iris_streak_direction = iris_dir
#                                     iris_streak_count = 1

#                                 if iris_streak_count >= IRIS_STREAK_TRIGGER:
#                                     cv2.putText(frame, f"GAZE WARN: {iris_dir}", (20, 130), font, 0.9, (0, 0, 255), 3)
#                                     if current_timestamp - last_trigger_snap_time > 3.0:
#                                         v_type = iris_dir.title()  # e.g., "Iris Left", "Iris Right"
#                                         tracker.log_violation(
#                                             v_type, formatted_time,
#                                             f"Candidate showed repeated eye movement ({iris_dir})",
#                                             "Please maintain eye contact with the screen center",
#                                             frame
#                                         )
#                                         # Standard snapshot
#                                         snap_path = snapshot_dir / f"gaze_violation_{int(current_timestamp)}.jpg"
#                                         cv2.imwrite(str(snap_path), frame)
#                                         last_trigger_snap_time = current_timestamp
#                             else:
#                                 iris_streak_direction = None
#                                 iris_streak_count = 0

#                 writer.write(frame)
#                 cv2.imshow("AI Proctoring & Recording System", frame)

#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     print("🛑 Stop key pressed. Closing recording...")
#                     break

#     finally:
#         cap.release()
#         writer.release()
#         cv2.destroyAllWindows()

#         if HAS_PYAUDIO and audio_recorder.is_alive():
#             audio_recorder.stop()
#             audio_recorder.join()

#         actual_fps = frame_count / elapsed_sec if elapsed_sec > 0 else target_fps
#         combine_audio_video(raw_video_path, audio_path, final_video_path, actual_fps)

# if __name__ == "__main__":
#     candidate_img = select_candidate()
#     if candidate_img:
#         start_interview_recording(candidate_img)





# import cv2
# import mediapipe as mp
# import face_recognition
# import time
# import os
# import numpy as np
# from pathlib import Path
# import threading
# import wave
# import subprocess
# import csv
# import uuid


# # PyAudio import with fallback handling
# try:
#     import pyaudio
#     HAS_PYAUDIO = True
# except ImportError:
#     HAS_PYAUDIO = False
#     print("⚠️ 'pyaudio' is not installed. Audio recording will be disabled.")
#     print("   To enable audio, install pyaudio (e.g. `pip install pyaudio`).")

# # ─────────────────────────────────────────────────────────────────────────────
# # Audio Recording Helper Thread
# # ─────────────────────────────────────────────────────────────────────────────

# class AudioRecorderThread(threading.Thread):
#     def __init__(self, output_wav_path: Path, sample_rate=44100, channels=1, chunk=1024):
#         super().__init__()
#         self.output_wav_path = output_wav_path
#         self.sample_rate = sample_rate
#         self.channels = channels
#         self.chunk = chunk
#         self.format = pyaudio.paInt16 if HAS_PYAUDIO else None
#         self.is_recording = False
#         self._frames = []

#     def run(self):
#         if not HAS_PYAUDIO:
#             return

#         p = pyaudio.PyAudio()
#         try:
#             stream = p.open(
#                 format=self.format,
#                 channels=self.channels,
#                 rate=self.sample_rate,
#                 input=True,
#                 frames_per_buffer=self.chunk
#             )
#         except Exception as e:
#             print(f"⚠️ Could not open microphone for audio recording: {e}")
#             p.terminate()
#             return

#         self.is_recording = True
#         self._frames = []

#         while self.is_recording:
#             try:
#                 data = stream.read(self.chunk, exception_on_overflow=False)
#                 self._frames.append(data)
#             except Exception:
#                 break

#         stream.stop_stream()
#         stream.close()
#         p.terminate()

#         # Save to WAV file
#         if self._frames:
#             wf = wave.open(str(self.output_wav_path), 'wb')
#             wf.setnchannels(self.channels)
#             wf.setsampwidth(p.get_sample_size(self.format))
#             wf.setframerate(self.sample_rate)
#             wf.writeframes(b''.join(self._frames))
#             wf.close()

#     def stop(self):
#         self.is_recording = False

# def combine_audio_video(video_path: Path, audio_path: Path, output_path: Path, actual_fps: float = 20.0):
#     """Muxes audio and video together using ffmpeg with robust fallbacks."""
#     if not audio_path.exists() or audio_path.stat().st_size == 0:
#         print("⚠️ Audio track empty or missing. Keeping video file as-is.")
#         return

#     cmd = [
#         "ffmpeg",
#         "-y",
#         "-i", str(video_path),
#         "-i", str(audio_path),
#         "-c:v", "copy",
#         "-c:a", "aac",
#         "-map", "0:v:0",
#         "-map", "1:a:0",
#         "-shortest",
#         str(output_path)
#     ]

#     try:
#         subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
#         if output_path.exists() and output_path.stat().st_size > 0:
#             video_path.unlink(missing_ok=True)
#             audio_path.unlink(missing_ok=True)
#             print(f"✅ Synchronized video + audio saved to: {output_path}")
#             return
#     except Exception as e:
#         print(f"⚠️ Primary FFmpeg muxing failed ({e}). Trying full re-encode...")

#     cmd_reencode = [
#         "ffmpeg",
#         "-y",
#         "-r", f"{actual_fps:.2f}",
#         "-i", str(video_path),
#         "-i", str(audio_path),
#         "-c:v", "libx264",
#         "-pix_fmt", "yuv420p",
#         "-c:a", "aac",
#         "-async", "1",
#         "-shortest",
#         str(output_path)
#     ]

#     try:
#         subprocess.run(cmd_reencode, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
#         if output_path.exists() and output_path.stat().st_size > 0:
#             video_path.unlink(missing_ok=True)
#             audio_path.unlink(missing_ok=True)
#             print(f"✅ Synchronized video + audio saved to: {output_path}")
#     except Exception as e:
#         print(f"❌ FFmpeg execution failed. Ensure FFmpeg is installed and added to PATH.")

# # ─────────────────────────────────────────────────────────────────────────────
# # Eye Aspect Ratio (EAR) & Blink Helper Functions
# # ─────────────────────────────────────────────────────────────────────────────

# def calculate_ear(landmarks, frame_width, frame_height):
#     """Calculates Eye Aspect Ratio (EAR) for both eyes to detect closures."""
#     def get_pt(idx):
#         return np.array([landmarks[idx].x * frame_width, landmarks[idx].y * frame_height])

#     def single_eye_ear(p_top1, p_bot1, p_top2, p_bot2, p_left, p_right):
#         v1 = np.linalg.norm(p_top1 - p_bot1)
#         v2 = np.linalg.norm(p_top2 - p_bot2)
#         h = np.linalg.norm(p_left - p_right)
#         return (v1 + v2) / (2.0 * h) if h > 0 else 0.3

#     l_ear = single_eye_ear(get_pt(159), get_pt(145), get_pt(158), get_pt(153), get_pt(33), get_pt(133))
#     r_ear = single_eye_ear(get_pt(386), get_pt(374), get_pt(385), get_pt(380), get_pt(362), get_pt(263))

#     return (l_ear + r_ear) / 2.0

# # ─────────────────────────────────────────────────────────────────────────────
# # Lighting & Brightness Monitoring Helper
# # ─────────────────────────────────────────────────────────────────────────────

# MIN_BRIGHTNESS_PCT = 35.0
# MAX_BRIGHTNESS_PCT = 75.0

# def analyze_lighting_conditions(frame, face_box=None):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     h, w = gray.shape

#     if face_box is None:
#         overall_avg = np.mean(gray)
#         overall_pct = (overall_avg / 255.0) * 100.0
#         return None, overall_pct

#     x, y, bw, bh = face_box
#     x1, y1 = max(0, x), max(0, y)
#     x2, y2 = min(w, x + bw), min(h, y + bh)

#     face_region = gray[y1:y2, x1:x2]
#     face_brightness_pct = ((np.mean(face_region) / 255.0) * 100.0) if face_region.size > 0 else None

#     bg_mask = np.ones((h, w), dtype=bool)
#     bg_mask[y1:y2, x1:x2] = False
#     bg_pixels = gray[bg_mask]
#     bg_brightness_pct = ((np.mean(bg_pixels) / 255.0) * 100.0) if bg_pixels.size > 0 else None

#     return face_brightness_pct, bg_brightness_pct

# # ─────────────────────────────────────────────────────────────────────────────
# # Black Glasses Detection Helper Function
# # ─────────────────────────────────────────────────────────────────────────────

# def detect_black_glasses(landmarks, frame, frame_width, frame_height):
#     """
#     Detection using adaptive face skin baseline vs eye region luminance,
#     dark pixel density, and contrast variation.
#     """
#     LEFT_EYE_INDICES = [33, 133, 159, 145, 158, 153, 144, 160]
#     RIGHT_EYE_INDICES = [362, 263, 386, 374, 385, 380, 373, 387]
#     SKIN_SAMPLING_INDICES = [10, 109, 338, 205, 425]  # Forehead + Cheeks

#     def get_coords(idx_list):
#         pts = []
#         for idx in idx_list:
#             pts.append([int(landmarks[idx].x * frame_width), int(landmarks[idx].y * frame_height)])
#         return np.array(pts, dtype=np.int32)

#     l_pts = get_coords(LEFT_EYE_INDICES)
#     r_pts = get_coords(RIGHT_EYE_INDICES)
#     skin_pts = get_coords(SKIN_SAMPLING_INDICES)

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     skin_values = []
#     for pt in skin_pts:
#         px = min(frame_width - 1, max(0, pt[0]))
#         py = min(frame_height - 1, max(0, pt[1]))
#         skin_values.append(gray[py, px])
    
#     skin_baseline = float(np.mean(skin_values)) if skin_values else 120.0
#     skin_baseline = max(skin_baseline, 30.0)

#     def analyze_eye_region(pts, expand=12):
#         x, y, w, h = cv2.boundingRect(pts)
#         x1, y1 = max(0, x - expand), max(0, y - expand)
#         x2, y2 = min(frame_width, x + w + expand), min(frame_height, y + h + expand)

#         region = gray[y1:y2, x1:x2]
#         if region.size == 0:
#             return 255.0, 0.0

#         mean_brightness = np.mean(region)
#         dark_thresh = max(40, skin_baseline * 0.45)
#         dark_pixels = np.sum(region < dark_thresh)
#         dark_ratio = float(dark_pixels) / float(region.size)

#         return mean_brightness, dark_ratio

#     l_bright, l_dark_ratio = analyze_eye_region(l_pts)
#     r_bright, r_dark_ratio = analyze_eye_region(r_pts)

#     avg_eye_brightness = (l_bright + r_bright) / 2.0
#     avg_dark_pixel_ratio = (l_dark_ratio + r_dark_ratio) / 2.0
#     brightness_ratio = avg_eye_brightness / skin_baseline

#     is_glasses = (avg_dark_pixel_ratio >= 0.40) and (brightness_ratio < 0.45)
#     return is_glasses

# # ─────────────────────────────────────────────────────────────────────────────
# # Iris / Gaze Direction Helper Function
# # ─────────────────────────────────────────────────────────────────────────────

# def detect_iris_direction(landmarks, frame_width, frame_height):
#     LEFT_IRIS_CENTER = 468
#     LEFT_EYE_INNER   = 133
#     LEFT_EYE_OUTER   = 33
#     LEFT_EYE_TOP     = 159
#     LEFT_EYE_BOTTOM  = 145

#     RIGHT_IRIS_CENTER = 473
#     RIGHT_EYE_INNER   = 362
#     RIGHT_EYE_OUTER   = 263
#     RIGHT_EYE_TOP     = 386
#     RIGHT_EYE_BOTTOM  = 374

#     def get_pt(idx):
#         return np.array([landmarks[idx].x * frame_width, landmarks[idx].y * frame_height])

#     l_iris = get_pt(LEFT_IRIS_CENTER)
#     l_inner, l_outer = get_pt(LEFT_EYE_INNER), get_pt(LEFT_EYE_OUTER)
#     l_top, l_bottom   = get_pt(LEFT_EYE_TOP), get_pt(LEFT_EYE_BOTTOM)

#     l_horiz_dist = np.linalg.norm(l_outer - l_inner)
#     l_vert_dist  = np.linalg.norm(l_bottom - l_top)

#     l_h_ratio = (np.linalg.norm(l_iris - l_inner) / l_horiz_dist) if l_horiz_dist > 0 else 0.5
#     l_v_ratio = (np.linalg.norm(l_iris - l_top) / l_vert_dist) if l_vert_dist > 0 else 0.5

#     r_iris = get_pt(RIGHT_IRIS_CENTER)
#     r_inner, r_outer = get_pt(RIGHT_EYE_INNER), get_pt(RIGHT_EYE_OUTER)
#     r_top, r_bottom   = get_pt(RIGHT_EYE_TOP), get_pt(RIGHT_EYE_BOTTOM)

#     r_horiz_dist = np.linalg.norm(r_outer - r_inner)
#     r_vert_dist  = np.linalg.norm(r_bottom - r_top)

#     r_h_ratio = (np.linalg.norm(r_iris - r_inner) / r_horiz_dist) if r_horiz_dist > 0 else 0.5
#     r_v_ratio = (np.linalg.norm(r_iris - r_top) / r_vert_dist) if r_vert_dist > 0 else 0.5

#     avg_h_ratio = (l_h_ratio + r_h_ratio) / 2.0
#     avg_v_ratio = (l_v_ratio + r_v_ratio) / 2.0

#     direction = None
#     if avg_h_ratio < 0.42:
#         direction = "IRIS LEFT"
#     elif avg_h_ratio > 0.58:
#         direction = "IRIS RIGHT"
#     elif avg_v_ratio < 0.35:
#         direction = "IRIS UP"
#     elif avg_v_ratio > 0.65:
#         direction = "IRIS DOWN"

#     right_iris_pixel = (int(r_iris[0]), int(r_iris[1]))
#     left_iris_pixel  = (int(l_iris[0]), int(l_iris[1]))

#     return direction, right_iris_pixel, left_iris_pixel

# # ─────────────────────────────────────────────────────────────────────────────
# # In-Memory Violation Tracker & Threshold CSV Logging
# # ─────────────────────────────────────────────────────────────────────────────

# TRIGGER_CSV_HEADER = ["User_ID", "Violation", "Timestamps", "HR Message", "AI Message"]

# class ViolationTracker:
#     def __init__(self, csv_path: Path, user_id: str, threshold: int = 6):
#         self.csv_path = Path(csv_path)
#         self.user_id = user_id
#         self.threshold = threshold
#         self.violations = {}
#         self._ensure_csv_header()

#     def _ensure_csv_header(self):
#         if not self.csv_path.exists():
#             self.csv_path.parent.mkdir(parents=True, exist_ok=True)
#             with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
#                 writer = csv.writer(f)
#                 writer.writerow(TRIGGER_CSV_HEADER)

#     def log_violation(self, violation_type: str, timestamp_str: str, hr_text: str, ai_warning: str):
#         """Buffers violation timestamps in memory. Writes to CSV only when occurrences >= threshold (6+)."""
#         if violation_type not in self.violations:
#             self.violations[violation_type] = {
#                 "timestamps": [],
#                 "hr_text": hr_text,
#                 "ai_warning": ai_warning
#             }

#         v = self.violations[violation_type]
#         v["timestamps"].append(timestamp_str)
#         v["hr_text"] = hr_text
#         v["ai_warning"] = ai_warning

#         count = len(v["timestamps"])
#         if count >= self.threshold:
#             timestamps_joined = ", ".join(v["timestamps"])
#             with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
#                 writer = csv.writer(f)
#                 writer.writerow([self.user_id, violation_type, timestamps_joined, hr_text, ai_warning])
#             print(f"🚨 [CSV TRIGGER LOGGED ({count} occurrences)] | Type: '{violation_type}' | Timestamps: {timestamps_joined}")
#         else:
#             print(f"⚠️ [VIOLATION BUFFERED ({count}/{self.threshold})] | Type: '{violation_type}' @ {timestamp_str}")

# # ─────────────────────────────────────────────────────────────────────────────
# # Candidate Selection & Encoding
# # ─────────────────────────────────────────────────────────────────────────────

# IMAGES_DIR = Path("images")
# SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# def list_candidates(images_dir: Path) -> list[Path]:
#     if not images_dir.exists():
#         return []
#     return sorted(
#         p for p in images_dir.iterdir()
#         if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
#     )

# def select_candidate() -> Path | None:
#     candidates = list_candidates(IMAGES_DIR)

#     if not candidates:
#         print(f"\n❌ No images found in '{IMAGES_DIR}/'.")
#         print(f"   Add candidate photos there and try again.")
#         return None

#     print("\n" + "═" * 50)
#     print("  📋 CANDIDATE SELECTION")
#     print("═" * 50)
#     for idx, path in enumerate(candidates, start=1):
#         name = path.stem.replace("_", " ").replace("-", " ").title()
#         print(f"  [{idx}]  {name}  ({path.name})")
#     print("  [q]  Quit")
#     print("═" * 50)

#     while True:
#         raw = input("Select candidate number: ").strip().lower()
#         if raw == "q":
#             return None
#         if raw.isdigit():
#             choice = int(raw)
#             if 1 <= choice <= len(candidates):
#                 selected = candidates[choice - 1]
#                 name = selected.stem.replace("_", " ").replace("-", " ").title()
#                 print(f"\n✅ Selected: {name} → {selected}")
#                 return selected
#         print(f"   ⚠️ Enter a number between 1 and {len(candidates)}, or 'q'.")

# def load_reference_encoding(reference_image_path: Path):
#     if not reference_image_path.exists():
#         print(f"Error: Reference image not found at '{reference_image_path}'.")
#         return None

#     reference_image = face_recognition.load_image_file(str(reference_image_path))
#     face_locations = face_recognition.face_locations(reference_image)

#     if len(face_locations) == 0:
#         print("Error: No face detected in the reference image.")
#         return None
#     if len(face_locations) > 1:
#         print("Error: Reference image has more than one face.")
#         return None

#     return face_recognition.face_encodings(reference_image, known_face_locations=face_locations)[0]

# # ─────────────────────────────────────────────────────────────────────────────
# # Main Interview Session
# # ─────────────────────────────────────────────────────────────────────────────

# def start_interview_recording(reference_image_path: Path, match_tolerance=0.6, verify_every_seconds=2.0):
#     candidate_name = reference_image_path.stem.replace("_", " ").replace("-", " ").title()
#     safe_name = reference_image_path.stem.replace(" ", "_")

#     print(f"\n🔍 Encoding reference photo for: {candidate_name} ...")
#     reference_encoding = load_reference_encoding(reference_image_path)
#     if reference_encoding is None:
#         print("Aborting: fix the reference image and try again.")
#         return
#     print(f"✅ Reference photo loaded successfully.")

#     # 1. Directories & Tracking setup
#     video_dir = Path("video_interview")
#     snapshot_dir = video_dir / "snapshots"
#     video_dir.mkdir(parents=True, exist_ok=True)
#     snapshot_dir.mkdir(parents=True, exist_ok=True)

#     user_id = f"USER_{uuid.uuid4().hex[:6].upper()}"
#     trigger_csv_path = video_dir / f"trigger_points_{safe_name}_{user_id}.csv"
#     tracker = ViolationTracker(csv_path=trigger_csv_path, user_id=user_id, threshold=6)

#     # 2. Camera Setup
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return

#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     timestamp     = time.strftime('%Y%m%d_%H%M%S')
#     raw_video_path = video_dir / f"interview_{safe_name}_{timestamp}_raw.mp4"
#     audio_path     = video_dir / f"interview_{safe_name}_{timestamp}.wav"
#     final_video_path = video_dir / f"interview_{safe_name}_{timestamp}.mp4"

#     target_fps = 20.0
#     writer = cv2.VideoWriter(str(raw_video_path), cv2.VideoWriter_fourcc(*'mp4v'), target_fps, (width, height))

#     audio_recorder = AudioRecorderThread(output_wav_path=audio_path)
#     audio_recorder.start()

#     # 3. MediaPipe Setup
#     mp_face = mp.solutions.face_detection
#     mp_face_mesh = mp.solutions.face_mesh

#     # Snapshot Cooldown Timers
#     last_verify_time = 0
#     last_snap_time = 0
#     last_movement_snap_time = 0
#     last_glasses_snap_time = 0
#     last_light_snap_time = 0
#     last_eye_closure_snap_time = 0
#     last_no_blink_snap_time = 0
#     last_trigger_snap_time = 0
#     last_missing_face_snap_time = 0
#     last_mismatch_snap_time = 0

#     EAR_CLOSED_THRESHOLD = 0.18
#     eye_closed_start_time = None

#     NO_BLINK_WARN_SECONDS = 15.0
#     last_blink_timestamp = time.time()

#     glasses_consecutive_frames = 0
#     REQUIRED_GLASSES_FRAMES = 3

#     EYE_OPEN_FOR_GAZE_EAR = 0.23
#     IRIS_STREAK_TRIGGER = 6
#     iris_streak_direction = None
#     iris_streak_count = 0

#     identity_ok = None
#     font = cv2.FONT_HERSHEY_SIMPLEX

#     frame_count = 0
#     start_time = time.time()

#     print(f"\n🎥 Recording to: {final_video_path}")
#     print(f"   Candidate : {candidate_name}")
#     print(f"   User ID   : {user_id}")
#     print(f"   CSV Log   : {trigger_csv_path}")
#     print("   Press 'q' in the video window to stop.\n")

#     try:
#         with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_det, \
#              mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:

#             while cap.isOpened():
#                 success, frame = cap.read()
#                 if not success:
#                     break

#                 frame_count += 1
#                 current_timestamp = time.time()
#                 elapsed_sec = current_timestamp - start_time
#                 formatted_time = time.strftime("%H:%M:%S", time.localtime(current_timestamp))

#                 rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results    = face_det.process(rgb_frame)
#                 face_count = len(results.detections) if results.detections else 0

#                 face_box = None

#                 # ── Primary Verification Loop ─────────────────────────────
#                 if face_count > 0:
#                     det = results.detections[0]
#                     box = det.location_data.relative_bounding_box
#                     x, y = int(box.xmin * width), int(box.ymin * height)
#                     w_box, h_box = int(box.width * width), int(box.height * height)
#                     face_box = (x, y, w_box, h_box)

#                     if current_timestamp - last_verify_time > verify_every_seconds:
#                         last_verify_time = current_timestamp
#                         top, left     = max(0, y), max(0, x)
#                         bottom, right = min(height, y + h_box), min(width, x + w_box)

#                         try:
#                             encodings = face_recognition.face_encodings(
#                                 rgb_frame,
#                                 known_face_locations=[(top, right, bottom, left)]
#                             )
#                             if encodings:
#                                 distance = face_recognition.face_distance([reference_encoding], encodings[0])[0]
#                                 identity_ok = bool(distance <= match_tolerance)
#                                 if not identity_ok and (current_timestamp - last_mismatch_snap_time > 3.0):
#                                     hr_msg = "Unrecognized person in candidate position"
#                                     ai_warn = "Verification failed, candidate must be present"
#                                     tracker.log_violation("Identity Mismatch", formatted_time, hr_msg, ai_warn)
                                    
#                                     snap_path = snapshot_dir / f"identity_mismatch_{int(current_timestamp)}.jpg"
#                                     cv2.imwrite(str(snap_path), frame)
#                                     last_mismatch_snap_time = current_timestamp
#                         except Exception as e:
#                             print(f"⚠️ Verification skipped: {e}")

#                 # ── Bounding Boxes ─────────────────────────────────────────
#                 if results.detections:
#                     for idx, det_item in enumerate(results.detections):
#                         b = det_item.location_data.relative_bounding_box
#                         bx, by = int(b.xmin * width), int(b.ymin * height)
#                         bw, bh = int(b.width * width), int(b.height * height)
                        
#                         if idx == 0 and identity_ok is True:
#                             color = (0, 255, 0)
#                             label = f"Candidate: {candidate_name}"
#                         else:
#                             color = (0, 0, 255)
#                             label = "Unknown / Other Person"

#                         cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)
#                         cv2.putText(frame, label, (bx, max(20, by - 10)), font, 0.6, color, 2)

#                 # ── Light / Brightness Monitoring ─────────────────────────
#                 face_pct, bg_pct = analyze_lighting_conditions(frame, face_box)

#                 hud_y = 30
#                 if face_pct is not None:
#                     cv2.putText(frame, f"Face Light: {face_pct:.1f}%", (width - 230, hud_y), font, 0.6, (255, 255, 255), 2)
#                     hud_y += 25
#                 if bg_pct is not None:
#                     cv2.putText(frame, f"BG Light:   {bg_pct:.1f}%", (width - 230, hud_y), font, 0.6, (255, 255, 255), 2)

#                 lighting_warning = None
#                 if face_pct is not None:
#                     if face_pct < MIN_BRIGHTNESS_PCT:
#                         lighting_warning = "WARN: FRONT LIGHT LOW"
#                     elif face_pct > MAX_BRIGHTNESS_PCT:
#                         lighting_warning = "WARN: FRONT LIGHT TOO HIGH - USE NORMAL LIGHT"

#                 if not lighting_warning and bg_pct is not None:
#                     if bg_pct < MIN_BRIGHTNESS_PCT:
#                         lighting_warning = "WARN: BACKGROUND LIGHT IS LOW"
#                     elif bg_pct > MAX_BRIGHTNESS_PCT:
#                         lighting_warning = "WARN: BACKGROUND LIGHT HIGH - SIT IN NORMAL LIGHT"

#                 if lighting_warning:
#                     cv2.putText(frame, lighting_warning, (20, 210), font, 0.75, (0, 165, 255), 2)
#                     if current_timestamp - last_light_snap_time > 3.0:
#                         tracker.log_violation(
#                             "Lighting Violation", formatted_time,
#                             f"Lighting Violation: {lighting_warning}",
#                             "Please adjust lighting to face the camera in normal light"
#                         )
#                         snap_path = snapshot_dir / f"light_violation_{int(current_timestamp)}.jpg"
#                         cv2.imwrite(str(snap_path), frame)
#                         last_light_snap_time = current_timestamp

#                 # ── People Counter Logic ──────────────────────────────────
#                 if face_count > 1:
#                     cv2.putText(frame, f"WARNING: {face_count} PEOPLE!", (20, 50), font, 1, (0, 0, 255), 3)
#                     cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)

#                     if current_timestamp - last_snap_time > 3.0:
#                         tracker.log_violation(
#                             "Multiple People", formatted_time,
#                             f"Multiple people detected ({face_count} people)",
#                             "Only candidate is allowed in frame, clear other persons"
#                         )
#                         snap_path = snapshot_dir / f"violation_{int(current_timestamp)}.jpg"
#                         cv2.imwrite(str(snap_path), frame)
#                         last_snap_time = current_timestamp

#                 elif face_count == 0:
#                     cv2.putText(frame, "WARNING: NO CANDIDATE", (20, 50), font, 1, (0, 165, 255), 3)
#                     identity_ok = None
#                     glasses_consecutive_frames = 0
#                     eye_closed_start_time = None
#                     last_blink_timestamp = current_timestamp
#                     iris_streak_direction = None
#                     iris_streak_count = 0

#                     if current_timestamp - last_missing_face_snap_time > 3.0:
#                         tracker.log_violation(
#                             "Missing Candidate", formatted_time,
#                             "Candidate missing from frame",
#                             "Please stay in front of the camera at all times"
#                         )
#                         snap_path = snapshot_dir / f"missing_candidate_{int(current_timestamp)}.jpg"
#                         cv2.imwrite(str(snap_path), frame)
#                         last_missing_face_snap_time = current_timestamp

#                 else:
#                     det = results.detections[0]

#                     # ── Head Movement Detection ──────────────────────────────
#                     keypoints = det.location_data.relative_keypoints
#                     rx, ry = int(keypoints[0].x * width), int(keypoints[0].y * height)
#                     lx, ly = int(keypoints[1].x * width), int(keypoints[1].y * height)
#                     nx, ny = int(keypoints[2].x * width), int(keypoints[2].y * height)
#                     mx, my = int(keypoints[3].x * width), int(keypoints[3].y * height)

#                     eye_center_x = (rx + lx) / 2.0
#                     eye_dist = abs(lx - rx)
#                     yaw_ratio = (nx - eye_center_x) / eye_dist if eye_dist > 0 else 0

#                     eye_center_y = (ry + ly) / 2.0
#                     face_length = my - eye_center_y
#                     pitch_ratio = (ny - eye_center_y) / face_length if face_length > 0 else 0.5

#                     movement_state = None
#                     if yaw_ratio < -0.18:
#                         movement_state = "LOOKING RIGHT"
#                     elif yaw_ratio > 0.18:
#                         movement_state = "LOOKING LEFT"
#                     elif pitch_ratio < 0.27:
#                         movement_state = "LOOKING UP"
#                     elif pitch_ratio > 0.72:
#                         movement_state = "LOOKING DOWN"

#                     if movement_state:
#                         cv2.putText(frame, f"WARNING: {movement_state}", (20, 90), font, 1, (0, 165, 255), 3)
#                         if current_timestamp - last_movement_snap_time > 3.0:
#                             short_dir = movement_state.replace("LOOKING ", "").lower()
#                             tracker.log_violation(
#                                 "Head Movement", formatted_time,
#                                 f"Head turned {short_dir}",
#                                 f"please focused on screen don't look {short_dir}"
#                             )
#                             snap_path = snapshot_dir / f"movement_violation_{int(current_timestamp)}.jpg"
#                             cv2.imwrite(str(snap_path), frame)
#                             last_movement_snap_time = current_timestamp

#                     # ── Black Glasses, Iris & Eye Tracking ──────────
#                     mesh_results = face_mesh.process(rgb_frame)
#                     if mesh_results.multi_face_landmarks:
#                         face_landmarks = mesh_results.multi_face_landmarks[0]

#                         # ── Black Glasses Check ──────────
#                         is_wearing_glasses = detect_black_glasses(face_landmarks.landmark, frame, width, height)
#                         if is_wearing_glasses:
#                             glasses_consecutive_frames += 1
#                         else:
#                             glasses_consecutive_frames = max(0, glasses_consecutive_frames - 1)

#                         if glasses_consecutive_frames >= REQUIRED_GLASSES_FRAMES:
#                             cv2.putText(frame, "WARN: EYE NOT VISIBLE", (20, 170), font, 1, (0, 0, 255), 3)
#                             if current_timestamp - last_glasses_snap_time > 3.0:
#                                 tracker.log_violation(
#                                     "Eye Obstruction", formatted_time,
#                                     "Dark sunglasses or covered eyes detected",
#                                     "Please remove dark glasses, eyes must be clearly visible"
#                                 )
#                                 snap_path = snapshot_dir / f"glasses_violation_{int(current_timestamp)}.jpg"
#                                 cv2.imwrite(str(snap_path), frame)
#                                 last_glasses_snap_time = current_timestamp

#                         ear = calculate_ear(face_landmarks.landmark, width, height)

#                         # ── Eye Closed Warning (> 5 seconds) ─────────
#                         if ear < EAR_CLOSED_THRESHOLD:
#                             if eye_closed_start_time is None:
#                                 eye_closed_start_time = current_timestamp
#                                 last_blink_timestamp = current_timestamp
                            
#                             closed_duration = current_timestamp - eye_closed_start_time
                            
#                             if closed_duration >= 5.0:
#                                 cv2.putText(frame, "WARN: EYE CLOSED (>5s)", (20, 250), font, 0.9, (0, 0, 255), 3)

#                                 if current_timestamp - last_eye_closure_snap_time > 3.0:
#                                     tracker.log_violation(
#                                         "Eye Closure", formatted_time,
#                                         "Candidate eyes closed continuously for over 5 seconds",
#                                         "Please keep your eyes open and stay attentive"
#                                     )
#                                     snap_path = snapshot_dir / f"eye_closed_{int(current_timestamp)}.jpg"
#                                     cv2.imwrite(str(snap_path), frame)
#                                     last_eye_closure_snap_time = current_timestamp
#                         else:
#                             eye_closed_start_time = None

#                         # ── Liveliness (No-Blink Warning, 8s) ────────
#                         time_since_last_blink = current_timestamp - last_blink_timestamp
#                         if time_since_last_blink >= NO_BLINK_WARN_SECONDS:
#                             cv2.putText(frame, "WARN: PERSON NOT BLINKING (>8s)", (20, 290), font, 0.9, (0, 0, 255), 3)
#                             cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 4)
#                             if current_timestamp - last_no_blink_snap_time > 3.0:
#                                 tracker.log_violation(
#                                     "No Blink", formatted_time,
#                                     "No natural eye blink detected for over 8 seconds",
#                                     "Please blink naturally to maintain liveliness check"
#                                 )
#                                 snap_path = snapshot_dir / f"no_blink_{int(current_timestamp)}.jpg"
#                                 cv2.imwrite(str(snap_path), frame)
#                                 last_no_blink_snap_time = current_timestamp

#                         if ear < EAR_CLOSED_THRESHOLD:
#                             last_blink_timestamp = current_timestamp

#                         # ── Gaze & Iris Tracking Integration ───────
#                         if ear >= EYE_OPEN_FOR_GAZE_EAR:
#                             iris_dir, r_iris_pt, l_iris_pt = detect_iris_direction(
#                                 face_landmarks.landmark, width, height
#                             )
#                             cv2.circle(frame, r_iris_pt, 3, (0, 255, 255), -1)
#                             cv2.circle(frame, l_iris_pt, 3, (0, 255, 255), -1)

#                             if iris_dir:
#                                 if iris_dir == iris_streak_direction:
#                                     iris_streak_count += 1
#                                 else:
#                                     iris_streak_direction = iris_dir
#                                     iris_streak_count = 1

#                                 if iris_streak_count >= IRIS_STREAK_TRIGGER:
#                                     cv2.putText(frame, f"GAZE WARN: {iris_dir}", (20, 130), font, 0.9, (0, 0, 255), 3)
#                                     if current_timestamp - last_trigger_snap_time > 3.0:
#                                         tracker.log_violation(
#                                             "Eye Movement", formatted_time,
#                                             f"Candidate showed repeated eye movement ({iris_dir})",
#                                             "Please maintain eye contact with the screen center"
#                                         )
#                                         snap_path = snapshot_dir / f"gaze_violation_{int(current_timestamp)}.jpg"
#                                         cv2.imwrite(str(snap_path), frame)
#                                         last_trigger_snap_time = current_timestamp
#                             else:
#                                 iris_streak_direction = None
#                                 iris_streak_count = 0

#                 writer.write(frame)
#                 cv2.imshow("AI Proctoring & Recording System", frame)

#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     print("🛑 Stop key pressed. Closing recording...")
#                     break

#     finally:
#         cap.release()
#         writer.release()
#         cv2.destroyAllWindows()

#         if HAS_PYAUDIO and audio_recorder.is_alive():
#             audio_recorder.stop()
#             audio_recorder.join()

#         actual_fps = frame_count / elapsed_sec if elapsed_sec > 0 else target_fps
#         combine_audio_video(raw_video_path, audio_path, final_video_path, actual_fps)

# if __name__ == "__main__":
#     candidate_img = select_candidate()
#     if candidate_img:
#         start_interview_recording(candidate_img)


# import cv2
# import mediapipe as mp
# import face_recognition
# import time
# import os
# import numpy as np
# from pathlib import Path
# import threading
# import wave
# import subprocess
# import csv
# import uuid


# # PyAudio import with fallback handling
# try:
#     import pyaudio
#     HAS_PYAUDIO = True
# except ImportError:
#     HAS_PYAUDIO = False
#     print("⚠️ 'pyaudio' is not installed. Audio recording will be disabled.")
#     print("   To enable audio, install pyaudio (e.g. `pip install pyaudio`).")

# # ─────────────────────────────────────────────────────────────────────────────
# # Audio Recording Helper Thread
# # ─────────────────────────────────────────────────────────────────────────────

# class AudioRecorderThread(threading.Thread):
#     def __init__(self, output_wav_path: Path, sample_rate=44100, channels=1, chunk=1024):
#         super().__init__()
#         self.output_wav_path = output_wav_path
#         self.sample_rate = sample_rate
#         self.channels = channels
#         self.chunk = chunk
#         self.format = pyaudio.paInt16 if HAS_PYAUDIO else None
#         self.is_recording = False
#         self._frames = []

#     def run(self):
#         if not HAS_PYAUDIO:
#             return

#         p = pyaudio.PyAudio()
#         try:
#             stream = p.open(
#                 format=self.format,
#                 channels=self.channels,
#                 rate=self.sample_rate,
#                 input=True,
#                 frames_per_buffer=self.chunk
#             )
#         except Exception as e:
#             print(f"⚠️ Could not open microphone for audio recording: {e}")
#             p.terminate()
#             return

#         self.is_recording = True
#         self._frames = []

#         while self.is_recording:
#             try:
#                 data = stream.read(self.chunk, exception_on_overflow=False)
#                 self._frames.append(data)
#             except Exception:
#                 break

#         stream.stop_stream()
#         stream.close()
#         p.terminate()

#         # Save to WAV file
#         if self._frames:
#             wf = wave.open(str(self.output_wav_path), 'wb')
#             wf.setnchannels(self.channels)
#             wf.setsampwidth(p.get_sample_size(self.format))
#             wf.setframerate(self.sample_rate)
#             wf.writeframes(b''.join(self._frames))
#             wf.close()

#     def stop(self):
#         self.is_recording = False

# def combine_audio_video(video_path: Path, audio_path: Path, output_path: Path, actual_fps: float = 20.0):
#     """Muxes audio and video together using ffmpeg with robust fallbacks."""
#     if not audio_path.exists() or audio_path.stat().st_size == 0:
#         print("⚠️ Audio track empty or missing. Keeping video file as-is.")
#         return

#     cmd = [
#         "ffmpeg",
#         "-y",
#         "-i", str(video_path),
#         "-i", str(audio_path),
#         "-c:v", "copy",
#         "-c:a", "aac",
#         "-map", "0:v:0",
#         "-map", "1:a:0",
#         "-shortest",
#         str(output_path)
#     ]

#     try:
#         subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
#         if output_path.exists() and output_path.stat().st_size > 0:
#             video_path.unlink(missing_ok=True)
#             audio_path.unlink(missing_ok=True)
#             print(f"✅ Synchronized video + audio saved to: {output_path}")
#             return
#     except Exception as e:
#         print(f"⚠️ Primary FFmpeg muxing failed ({e}). Trying full re-encode...")

#     cmd_reencode = [
#         "ffmpeg",
#         "-y",
#         "-r", f"{actual_fps:.2f}",
#         "-i", str(video_path),
#         "-i", str(audio_path),
#         "-c:v", "libx264",
#         "-pix_fmt", "yuv420p",
#         "-c:a", "aac",
#         "-async", "1",
#         "-shortest",
#         str(output_path)
#     ]

#     try:
#         subprocess.run(cmd_reencode, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
#         if output_path.exists() and output_path.stat().st_size > 0:
#             video_path.unlink(missing_ok=True)
#             audio_path.unlink(missing_ok=True)
#             print(f"✅ Synchronized video + audio saved to: {output_path}")
#     except Exception as e:
#         print(f"❌ FFmpeg execution failed. Ensure FFmpeg is installed and added to PATH.")

# # ─────────────────────────────────────────────────────────────────────────────
# # Eye Aspect Ratio (EAR) & Blink Helper Functions
# # ─────────────────────────────────────────────────────────────────────────────

# def calculate_ear(landmarks, frame_width, frame_height):
#     """Calculates Eye Aspect Ratio (EAR) for both eyes to detect closures."""
#     def get_pt(idx):
#         return np.array([landmarks[idx].x * frame_width, landmarks[idx].y * frame_height])

#     def single_eye_ear(p_top1, p_bot1, p_top2, p_bot2, p_left, p_right):
#         v1 = np.linalg.norm(p_top1 - p_bot1)
#         v2 = np.linalg.norm(p_top2 - p_bot2)
#         h = np.linalg.norm(p_left - p_right)
#         return (v1 + v2) / (2.0 * h) if h > 0 else 0.3

#     l_ear = single_eye_ear(get_pt(159), get_pt(145), get_pt(158), get_pt(153), get_pt(33), get_pt(133))
#     r_ear = single_eye_ear(get_pt(386), get_pt(374), get_pt(385), get_pt(380), get_pt(362), get_pt(263))

#     return (l_ear + r_ear) / 2.0

# # ─────────────────────────────────────────────────────────────────────────────
# # Lighting & Brightness Monitoring Helper
# # ─────────────────────────────────────────────────────────────────────────────

# MIN_BRIGHTNESS_PCT = 40.0
# MAX_BRIGHTNESS_PCT = 70.0

# def analyze_lighting_conditions(frame, face_box=None):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     h, w = gray.shape

#     if face_box is None:
#         overall_avg = np.mean(gray)
#         overall_pct = (overall_avg / 255.0) * 100.0
#         return None, overall_pct

#     x, y, bw, bh = face_box
#     x1, y1 = max(0, x), max(0, y)
#     x2, y2 = min(w, x + bw), min(h, y + bh)

#     face_region = gray[y1:y2, x1:x2]
#     face_brightness_pct = ((np.mean(face_region) / 255.0) * 100.0) if face_region.size > 0 else None

#     bg_mask = np.ones((h, w), dtype=bool)
#     bg_mask[y1:y2, x1:x2] = False
#     bg_pixels = gray[bg_mask]
#     bg_brightness_pct = ((np.mean(bg_pixels) / 255.0) * 100.0) if bg_pixels.size > 0 else None

#     return face_brightness_pct, bg_brightness_pct

# # ─────────────────────────────────────────────────────────────────────────────
# # Enhanced Black Glasses Detection Helper Function (Feature 1)
# # ─────────────────────────────────────────────────────────────────────────────

# def detect_black_glasses(landmarks, frame, frame_width, frame_height):
#     """
#     Improved detection using adaptive face skin baseline vs eye region luminance,
#     dark pixel density, and contrast variation.
#     """
#     LEFT_EYE_INDICES = [33, 133, 159, 145, 158, 153, 144, 160]
#     RIGHT_EYE_INDICES = [362, 263, 386, 374, 385, 380, 373, 387]
#     SKIN_SAMPLING_INDICES = [10, 109, 338, 205, 425]  # Forehead + Cheeks

#     def get_coords(idx_list):
#         pts = []
#         for idx in idx_list:
#             pts.append([int(landmarks[idx].x * frame_width), int(landmarks[idx].y * frame_height)])
#         return np.array(pts, dtype=np.int32)

#     l_pts = get_coords(LEFT_EYE_INDICES)
#     r_pts = get_coords(RIGHT_EYE_INDICES)
#     skin_pts = get_coords(SKIN_SAMPLING_INDICES)

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Calculate adaptive skin brightness baseline
#     skin_values = []
#     for pt in skin_pts:
#         px = min(frame_width - 1, max(0, pt[0]))
#         py = min(frame_height - 1, max(0, pt[1]))
#         skin_values.append(gray[py, px])
    
#     skin_baseline = float(np.mean(skin_values)) if skin_values else 120.0
#     skin_baseline = max(skin_baseline, 30.0)  # Avoid divide-by-zero in dark rooms

#     def analyze_eye_region(pts, expand=12):
#         x, y, w, h = cv2.boundingRect(pts)
#         x1, y1 = max(0, x - expand), max(0, y - expand)
#         x2, y2 = min(frame_width, x + w + expand), min(frame_height, y + h + expand)

#         region = gray[y1:y2, x1:x2]
#         if region.size == 0:
#             return 255.0, 0.0

#         mean_brightness = np.mean(region)
#         # Dynamic dark pixel threshold based on skin baseline
#         dark_thresh = max(40, skin_baseline * 0.45)
#         dark_pixels = np.sum(region < dark_thresh)
#         dark_ratio = float(dark_pixels) / float(region.size)

#         return mean_brightness, dark_ratio

#     l_bright, l_dark_ratio = analyze_eye_region(l_pts)
#     r_bright, r_dark_ratio = analyze_eye_region(r_pts)

#     avg_eye_brightness = (l_bright + r_bright) / 2.0
#     avg_dark_pixel_ratio = (l_dark_ratio + r_dark_ratio) / 2.0
#     brightness_ratio = avg_eye_brightness / skin_baseline

#     # Sunglasses / Black Glasses Condition:
#     # 1. High concentration of dark pixels in eye zone (>40%)
#     # 2. Eye zone significantly darker than surrounding facial skin (<45% of skin brightness)
#     is_glasses = (avg_dark_pixel_ratio >= 0.40) and (brightness_ratio < 0.45)

#     return is_glasses

# # ─────────────────────────────────────────────────────────────────────────────
# # Iris / Gaze Direction Helper Function
# # ─────────────────────────────────────────────────────────────────────────────

# def detect_iris_direction(landmarks, frame_width, frame_height):
#     LEFT_IRIS_CENTER = 468
#     LEFT_EYE_INNER   = 133
#     LEFT_EYE_OUTER   = 33
#     LEFT_EYE_TOP     = 159
#     LEFT_EYE_BOTTOM  = 145

#     RIGHT_IRIS_CENTER = 473
#     RIGHT_EYE_INNER   = 362
#     RIGHT_EYE_OUTER   = 263
#     RIGHT_EYE_TOP     = 386
#     RIGHT_EYE_BOTTOM  = 374

#     def get_pt(idx):
#         return np.array([landmarks[idx].x * frame_width, landmarks[idx].y * frame_height])

#     l_iris = get_pt(LEFT_IRIS_CENTER)
#     l_inner, l_outer = get_pt(LEFT_EYE_INNER), get_pt(LEFT_EYE_OUTER)
#     l_top, l_bottom   = get_pt(LEFT_EYE_TOP), get_pt(LEFT_EYE_BOTTOM)

#     l_horiz_dist = np.linalg.norm(l_outer - l_inner)
#     l_vert_dist  = np.linalg.norm(l_bottom - l_top)

#     l_h_ratio = (np.linalg.norm(l_iris - l_inner) / l_horiz_dist) if l_horiz_dist > 0 else 0.5
#     l_v_ratio = (np.linalg.norm(l_iris - l_top) / l_vert_dist) if l_vert_dist > 0 else 0.5

#     r_iris = get_pt(RIGHT_IRIS_CENTER)
#     r_inner, r_outer = get_pt(RIGHT_EYE_INNER), get_pt(RIGHT_EYE_OUTER)
#     r_top, r_bottom   = get_pt(RIGHT_EYE_TOP), get_pt(RIGHT_EYE_BOTTOM)

#     r_horiz_dist = np.linalg.norm(r_outer - r_inner)
#     r_vert_dist  = np.linalg.norm(r_bottom - r_top)

#     r_h_ratio = (np.linalg.norm(r_iris - r_inner) / r_horiz_dist) if r_horiz_dist > 0 else 0.5
#     r_v_ratio = (np.linalg.norm(r_iris - r_top) / r_vert_dist) if r_vert_dist > 0 else 0.5

#     avg_h_ratio = (l_h_ratio + r_h_ratio) / 2.0
#     avg_v_ratio = (l_v_ratio + r_v_ratio) / 2.0

#     direction = None
#     if avg_h_ratio < 0.42:
#         direction = "IRIS LEFT"
#     elif avg_h_ratio > 0.58:
#         direction = "IRIS RIGHT"
#     elif avg_v_ratio < 0.35:
#         direction = "IRIS UP"
#     elif avg_v_ratio > 0.65:
#         direction = "IRIS DOWN"

#     right_iris_pixel = (int(r_iris[0]), int(r_iris[1]))
#     left_iris_pixel  = (int(l_iris[0]), int(l_iris[1]))

#     return direction, right_iris_pixel, left_iris_pixel

# # ─────────────────────────────────────────────────────────────────────────────
# # Universal CSV Trigger Point Logging (Feature 2)
# # ─────────────────────────────────────────────────────────────────────────────

# TRIGGER_CSV_HEADER = ["User_ID", "Timestamp", "HR_Identification", "AI_Warning"]

# def ensure_trigger_csv_header(csv_path) -> None:
#     """Create the trigger-point CSV with header row if it doesn't exist yet."""
#     csv_path = Path(csv_path)
#     if not csv_path.exists():
#         csv_path.parent.mkdir(parents=True, exist_ok=True)
#         with open(csv_path, "w", newline="", encoding="utf-8") as f:
#             writer = csv.writer(f)
#             writer.writerow(TRIGGER_CSV_HEADER)

# def log_trigger_event(csv_path, user_id: str, timestamp_sec: str,
#                       hr_text: str, ai_warning: str) -> None:
#     """Records every trigger violation occurrence to the CSV file."""
#     csv_path = Path(csv_path)
#     ensure_trigger_csv_header(csv_path)
#     with open(csv_path, "a", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow([user_id, timestamp_sec, hr_text, ai_warning])
#     print(f"🚨 [CSV TRIGGER LOGGED @ {timestamp_sec}] | HR: '{hr_text}' | Warning: '{ai_warning}'")

# # ─────────────────────────────────────────────────────────────────────────────
# # Candidate Selection & Encoding
# # ─────────────────────────────────────────────────────────────────────────────

# IMAGES_DIR = Path("images")
# SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# def list_candidates(images_dir: Path) -> list[Path]:
#     if not images_dir.exists():
#         return []
#     return sorted(
#         p for p in images_dir.iterdir()
#         if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
#     )

# def select_candidate() -> Path | None:
#     candidates = list_candidates(IMAGES_DIR)

#     if not candidates:
#         print(f"\n❌ No images found in '{IMAGES_DIR}/'.")
#         print(f"   Add candidate photos there and try again.")
#         return None

#     print("\n" + "═" * 50)
#     print("  📋 CANDIDATE SELECTION")
#     print("═" * 50)
#     for idx, path in enumerate(candidates, start=1):
#         name = path.stem.replace("_", " ").replace("-", " ").title()
#         print(f"  [{idx}]  {name}  ({path.name})")
#     print("  [q]  Quit")
#     print("═" * 50)

#     while True:
#         raw = input("Select candidate number: ").strip().lower()
#         if raw == "q":
#             return None
#         if raw.isdigit():
#             choice = int(raw)
#             if 1 <= choice <= len(candidates):
#                 selected = candidates[choice - 1]
#                 name = selected.stem.replace("_", " ").replace("-", " ").title()
#                 print(f"\n✅ Selected: {name} → {selected}")
#                 return selected
#         print(f"   ⚠️ Enter a number between 1 and {len(candidates)}, or 'q'.")

# def load_reference_encoding(reference_image_path: Path):
#     if not reference_image_path.exists():
#         print(f"Error: Reference image not found at '{reference_image_path}'.")
#         return None

#     reference_image = face_recognition.load_image_file(str(reference_image_path))
#     face_locations = face_recognition.face_locations(reference_image)

#     if len(face_locations) == 0:
#         print("Error: No face detected in the reference image.")
#         return None
#     if len(face_locations) > 1:
#         print("Error: Reference image has more than one face.")
#         return None

#     return face_recognition.face_encodings(reference_image, known_face_locations=face_locations)[0]

# # ─────────────────────────────────────────────────────────────────────────────
# # Main Interview Session
# # ─────────────────────────────────────────────────────────────────────────────

# def start_interview_recording(reference_image_path: Path, match_tolerance=0.6, verify_every_seconds=2.0):
#     candidate_name = reference_image_path.stem.replace("_", " ").replace("-", " ").title()
#     safe_name = reference_image_path.stem.replace(" ", "_")

#     print(f"\n🔍 Encoding reference photo for: {candidate_name} ...")
#     reference_encoding = load_reference_encoding(reference_image_path)
#     if reference_encoding is None:
#         print("Aborting: fix the reference image and try again.")
#         return
#     print(f"✅ Reference photo loaded successfully.")

#     # 1. Directories setup
#     video_dir = Path("video_interview")
#     snapshot_dir = video_dir / "snapshots"
#     video_dir.mkdir(parents=True, exist_ok=True)
#     snapshot_dir.mkdir(parents=True, exist_ok=True)

#     # 1b. Unique ID & CSV path including Candidate Name and Unique ID
#     user_id = f"USER_{uuid.uuid4().hex[:6].upper()}"
#     trigger_csv_path = video_dir / f"trigger_points_{safe_name}_{user_id}.csv"
#     ensure_trigger_csv_header(trigger_csv_path)

#     # Register initial session start in CSV
#     log_trigger_event(
#         trigger_csv_path,
#         user_id,
#         "0.00s",
#         f"User session started (Candidate: {candidate_name})",
#         "No violation detected yet"
#     )

#     # 2. Camera Setup
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return

#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     timestamp     = time.strftime('%Y%m%d_%H%M%S')
#     raw_video_path = video_dir / f"interview_{safe_name}_{timestamp}_raw.mp4"
#     audio_path     = video_dir / f"interview_{safe_name}_{timestamp}.wav"
#     final_video_path = video_dir / f"interview_{safe_name}_{timestamp}.mp4"

#     target_fps = 20.0
#     writer = cv2.VideoWriter(str(raw_video_path), cv2.VideoWriter_fourcc(*'mp4v'), target_fps, (width, height))

#     # Initialize Audio Recorder Thread
#     audio_recorder = AudioRecorderThread(output_wav_path=audio_path)
#     audio_recorder.start()

#     # 3. MediaPipe Setup
#     mp_face = mp.solutions.face_detection
#     mp_face_mesh = mp.solutions.face_mesh

#     # Timers for Snapshot & Violation Cooldown (Prevents CSV spam while preserving continuous logs)
#     last_verify_time = 0
#     last_snap_time = 0
#     last_movement_snap_time = 0
#     last_glasses_snap_time = 0
#     last_light_snap_time = 0
#     last_eye_closure_snap_time = 0
#     last_no_blink_snap_time = 0
#     last_trigger_snap_time = 0
#     last_missing_face_snap_time = 0
#     last_mismatch_snap_time = 0

#     # Eye Closure Timers
#     EAR_CLOSED_THRESHOLD = 0.18
#     eye_closed_start_time = None

#     # Liveliness (Blink) Timers
#     NO_BLINK_WARN_SECONDS = 8.0
#     last_blink_timestamp = time.time()

#     # Black Glasses Smooth Counter
#     glasses_consecutive_frames = 0
#     REQUIRED_GLASSES_FRAMES = 3

#     # Gaze / Iris settings (> 5 occurrences threshold)
#     EYE_OPEN_FOR_GAZE_EAR = 0.23
#     IRIS_STREAK_TRIGGER = 6        # Triggered on > 5 occurrences (6th repeated frame)
#     iris_streak_direction = None
#     iris_streak_count = 0

#     identity_ok = None
#     font = cv2.FONT_HERSHEY_SIMPLEX

#     frame_count = 0
#     start_time = time.time()

#     print(f"\n🎥 Recording to: {final_video_path}")
#     print(f"   Candidate : {candidate_name}")
#     print(f"   User ID   : {user_id}")
#     print(f"   CSV Log   : {trigger_csv_path}")
#     print("   Press 'q' in the video window to stop.\n")

#     try:
#         with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_det, \
#              mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:

#             while cap.isOpened():
#                 success, frame = cap.read()
#                 if not success:
#                     break

#                 frame_count += 1
#                 current_timestamp = time.time()
#                 elapsed_sec = current_timestamp - start_time
#                 formatted_time = f"{elapsed_sec:.2f}s"

#                 rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results    = face_det.process(rgb_frame)
#                 face_count = len(results.detections) if results.detections else 0

#                 face_box = None

#                 # ── Primary Verification Loop ─────────────────────────────
#                 if face_count > 0:
#                     det = results.detections[0]
#                     box = det.location_data.relative_bounding_box
#                     x, y = int(box.xmin * width), int(box.ymin * height)
#                     w_box, h_box = int(box.width * width), int(box.height * height)
#                     face_box = (x, y, w_box, h_box)

#                     # Periodically run identity verification
#                     if current_timestamp - last_verify_time > verify_every_seconds:
#                         last_verify_time = current_timestamp
#                         top, left     = max(0, y), max(0, x)
#                         bottom, right = min(height, y + h_box), min(width, x + w_box)

#                         try:
#                             encodings = face_recognition.face_encodings(
#                                 rgb_frame,
#                                 known_face_locations=[(top, right, bottom, left)]
#                             )
#                             if encodings:
#                                 distance = face_recognition.face_distance([reference_encoding], encodings[0])[0]
#                                 identity_ok = bool(distance <= match_tolerance)
#                                 if not identity_ok and (current_timestamp - last_mismatch_snap_time > 3.0):
#                                     hr_msg = "Unrecognized person in candidate position"
#                                     ai_warn = "Verification failed, candidate must be present"
#                                     log_trigger_event(trigger_csv_path, user_id, formatted_time, hr_msg, ai_warn)
                                    
#                                     snap_path = snapshot_dir / f"identity_mismatch_{int(current_timestamp)}.jpg"
#                                     cv2.imwrite(str(snap_path), frame)
#                                     last_mismatch_snap_time = current_timestamp
#                         except Exception as e:
#                             print(f"⚠️ Verification skipped: {e}")

#                 # ── Bounding Boxes ─────────────────────────────────────────
#                 if results.detections:
#                     for idx, det_item in enumerate(results.detections):
#                         b = det_item.location_data.relative_bounding_box
#                         bx, by = int(b.xmin * width), int(b.ymin * height)
#                         bw, bh = int(b.width * width), int(b.height * height)
                        
#                         if idx == 0 and identity_ok is True:
#                             color = (0, 255, 0)
#                             label = f"Candidate: {candidate_name}"
#                         else:
#                             color = (0, 0, 255)
#                             label = "Unknown / Other Person"

#                         cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)
#                         cv2.putText(frame, label, (bx, max(20, by - 10)), font, 0.6, color, 2)

#                 # ── Light / Brightness Monitoring ─────────────────────────
#                 face_pct, bg_pct = analyze_lighting_conditions(frame, face_box)

#                 hud_y = 30
#                 if face_pct is not None:
#                     cv2.putText(frame, f"Face Light: {face_pct:.1f}%", (width - 230, hud_y), font, 0.6, (255, 255, 255), 2)
#                     hud_y += 25
#                 if bg_pct is not None:
#                     cv2.putText(frame, f"BG Light:   {bg_pct:.1f}%", (width - 230, hud_y), font, 0.6, (255, 255, 255), 2)

#                 lighting_warning = None
#                 if face_pct is not None:
#                     if face_pct < MIN_BRIGHTNESS_PCT:
#                         lighting_warning = "WARN: FRONT LIGHT LOW"
#                     elif face_pct > MAX_BRIGHTNESS_PCT:
#                         lighting_warning = "WARN: FRONT LIGHT TOO HIGH - USE NORMAL LIGHT"

#                 if not lighting_warning and bg_pct is not None:
#                     if bg_pct < MIN_BRIGHTNESS_PCT:
#                         lighting_warning = "WARN: BACKGROUND LIGHT IS LOW"
#                     elif bg_pct > MAX_BRIGHTNESS_PCT:
#                         lighting_warning = "WARN: BACKGROUND LIGHT HIGH - SIT IN NORMAL LIGHT"

#                 if lighting_warning:
#                     cv2.putText(frame, lighting_warning, (20, 210), font, 0.75, (0, 165, 255), 2)
#                     if current_timestamp - last_light_snap_time > 3.0:
#                         log_trigger_event(
#                             trigger_csv_path, user_id, formatted_time,
#                             f"Lighting Violation: {lighting_warning}",
#                             "Please adjust lighting to face the camera in normal light"
#                         )
#                         snap_path = snapshot_dir / f"light_violation_{int(current_timestamp)}.jpg"
#                         cv2.imwrite(str(snap_path), frame)
#                         last_light_snap_time = current_timestamp

#                 # ── People Counter Logic ──────────────────────────────────
#                 if face_count > 1:
#                     cv2.putText(frame, f"WARNING: {face_count} PEOPLE!", (20, 50), font, 1, (0, 0, 255), 3)
#                     cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)

#                     if current_timestamp - last_snap_time > 3.0:
#                         log_trigger_event(
#                             trigger_csv_path, user_id, formatted_time,
#                             f"Multiple people detected ({face_count} people)",
#                             "Only candidate is allowed in frame, clear other persons"
#                         )
#                         snap_path = snapshot_dir / f"violation_{int(current_timestamp)}.jpg"
#                         cv2.imwrite(str(snap_path), frame)
#                         last_snap_time = current_timestamp

#                 elif face_count == 0:
#                     cv2.putText(frame, "WARNING: NO CANDIDATE", (20, 50), font, 1, (0, 165, 255), 3)
#                     identity_ok = None
#                     glasses_consecutive_frames = 0
#                     eye_closed_start_time = None
#                     last_blink_timestamp = current_timestamp
#                     iris_streak_direction = None
#                     iris_streak_count = 0

#                     if current_timestamp - last_missing_face_snap_time > 3.0:
#                         log_trigger_event(
#                             trigger_csv_path, user_id, formatted_time,
#                             "Candidate missing from frame",
#                             "Please stay in front of the camera at all times"
#                         )
#                         snap_path = snapshot_dir / f"missing_candidate_{int(current_timestamp)}.jpg"
#                         cv2.imwrite(str(snap_path), frame)
#                         last_missing_face_snap_time = current_timestamp

#                 else:
#                     det = results.detections[0]

#                     # ── Head Movement Detection ──────────────────────────────
#                     keypoints = det.location_data.relative_keypoints
#                     rx, ry = int(keypoints[0].x * width), int(keypoints[0].y * height)
#                     lx, ly = int(keypoints[1].x * width), int(keypoints[1].y * height)
#                     nx, ny = int(keypoints[2].x * width), int(keypoints[2].y * height)
#                     mx, my = int(keypoints[3].x * width), int(keypoints[3].y * height)

#                     eye_center_x = (rx + lx) / 2.0
#                     eye_dist = abs(lx - rx)
#                     yaw_ratio = (nx - eye_center_x) / eye_dist if eye_dist > 0 else 0

#                     eye_center_y = (ry + ly) / 2.0
#                     face_length = my - eye_center_y
#                     pitch_ratio = (ny - eye_center_y) / face_length if face_length > 0 else 0.5

#                     movement_state = None
#                     if yaw_ratio < -0.10:
#                         movement_state = "LOOKING RIGHT"
#                     elif yaw_ratio > 0.10:
#                         movement_state = "LOOKING LEFT"
#                     elif pitch_ratio < 0.40:
#                         movement_state = "LOOKING UP"
#                     elif pitch_ratio > 0.65:
#                         movement_state = "LOOKING DOWN"

#                     if movement_state:
#                         cv2.putText(frame, f"WARNING: {movement_state}", (20, 90), font, 1, (0, 165, 255), 3)
#                         if current_timestamp - last_movement_snap_time > 3.0:
#                             short_dir = movement_state.replace("LOOKING ", "").lower()
#                             log_trigger_event(
#                                 trigger_csv_path, user_id, formatted_time,
#                                 f"Head turned {short_dir}",
#                                 f"please focused on screen don't look {short_dir}"
#                             )
#                             snap_path = snapshot_dir / f"movement_violation_{int(current_timestamp)}.jpg"
#                             cv2.imwrite(str(snap_path), frame)
#                             last_movement_snap_time = current_timestamp

#                     # ── Improved Black Glasses, Iris & Eye Tracking ──────────
#                     mesh_results = face_mesh.process(rgb_frame)
#                     if mesh_results.multi_face_landmarks:
#                         face_landmarks = mesh_results.multi_face_landmarks[0]

#                         # ── Feature 1: Improved Black Glasses Check ──────────
#                         is_wearing_glasses = detect_black_glasses(face_landmarks.landmark, frame, width, height)
#                         if is_wearing_glasses:
#                             glasses_consecutive_frames += 1
#                         else:
#                             glasses_consecutive_frames = max(0, glasses_consecutive_frames - 1)

#                         if glasses_consecutive_frames >= REQUIRED_GLASSES_FRAMES:
#                             cv2.putText(frame, "WARN: EYE NOT VISIBLE", (20, 170), font, 1, (0, 0, 255), 3)
#                             if current_timestamp - last_glasses_snap_time > 3.0:
#                                 log_trigger_event(
#                                     trigger_csv_path, user_id, formatted_time,
#                                     "Dark sunglasses or covered eyes detected",
#                                     "Please remove dark glasses, eyes must be clearly visible"
#                                 )
#                                 snap_path = snapshot_dir / f"glasses_violation_{int(current_timestamp)}.jpg"
#                                 cv2.imwrite(str(snap_path), frame)
#                                 last_glasses_snap_time = current_timestamp

#                         ear = calculate_ear(face_landmarks.landmark, width, height)

#                         # ── Eye Closed Warning (> 5 seconds) ─────────
#                         if ear < EAR_CLOSED_THRESHOLD:
#                             if eye_closed_start_time is None:
#                                 eye_closed_start_time = current_timestamp
#                                 last_blink_timestamp = current_timestamp
                            
#                             closed_duration = current_timestamp - eye_closed_start_time
                            
#                             if closed_duration >= 5.0:
#                                 cv2.putText(frame, "WARN: EYE CLOSED (>5s)", (20, 250), font, 0.9, (0, 0, 255), 3)

#                                 if current_timestamp - last_eye_closure_snap_time > 3.0:
#                                     log_trigger_event(
#                                         trigger_csv_path, user_id, formatted_time,
#                                         "Candidate eyes closed continuously for over 5 seconds",
#                                         "Please keep your eyes open and stay attentive"
#                                     )
#                                     snap_path = snapshot_dir / f"eye_closed_{int(current_timestamp)}.jpg"
#                                     cv2.imwrite(str(snap_path), frame)
#                                     last_eye_closure_snap_time = current_timestamp
#                         else:
#                             eye_closed_start_time = None

#                         # ── Liveliness (No-Blink Warning, 8s) ────────
#                         time_since_last_blink = current_timestamp - last_blink_timestamp
#                         if time_since_last_blink >= NO_BLINK_WARN_SECONDS:
#                             cv2.putText(frame, "WARN: PERSON NOT BLINKING (>8s)", (20, 290), font, 0.9, (0, 0, 255), 3)
#                             cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 4)
#                             if current_timestamp - last_no_blink_snap_time > 3.0:
#                                 log_trigger_event(
#                                     trigger_csv_path, user_id, formatted_time,
#                                     "No natural eye blink detected for over 8 seconds",
#                                     "Please blink naturally and look at the screen"
#                                 )
#                                 snap_path = snapshot_dir / f"no_blink_violation_{int(current_timestamp)}.jpg"
#                                 cv2.imwrite(str(snap_path), frame)
#                                 last_no_blink_snap_time = current_timestamp
#                         else:
#                             cv2.putText(frame, f"No Blink: {time_since_last_blink:.1f}s / {NO_BLINK_WARN_SECONDS:.0f}s", (20, 290), font, 0.6, (0, 255, 255), 2)

#                         # ── Feature 2: Iris Gaze Trigger (> 5 Repeated Occurrences) ──
#                         if ear >= EYE_OPEN_FOR_GAZE_EAR:
#                             iris_dir, r_iris_pt, l_iris_pt = detect_iris_direction(face_landmarks.landmark, width, height)
#                             cv2.circle(frame, r_iris_pt, 2, (0, 255, 0), -1)
#                             cv2.circle(frame, l_iris_pt, 2, (0, 255, 0), -1)

#                             if iris_dir is not None:
#                                 if iris_dir == iris_streak_direction:
#                                     iris_streak_count += 1
#                                 else:
#                                     iris_streak_direction = iris_dir
#                                     iris_streak_count = 1

#                                 # Exceeds threshold of 5 (i.e. on 6th occurrence)
#                                 if iris_streak_count >= IRIS_STREAK_TRIGGER:
#                                     short_dir = iris_dir.replace("IRIS ", "").lower()

#                                     hr_text = f"looking {short_dir} side repeadliy 6 time"
#                                     ai_warning = f"please focused on screen don't look {short_dir}"

#                                     cv2.putText(frame, ai_warning.upper(), (20, 130), font, 0.75, (0, 0, 255), 2)

#                                     if current_timestamp - last_trigger_snap_time > 3.0:
#                                         log_trigger_event(trigger_csv_path, user_id, formatted_time, hr_text, ai_warning)
#                                         snap_path = snapshot_dir / f"gaze_violation_{short_dir}_{int(current_timestamp)}.jpg"
#                                         cv2.imwrite(str(snap_path), frame)
#                                         last_trigger_snap_time = current_timestamp
#                             else:
#                                 iris_streak_direction = None
#                                 iris_streak_count = 0
#                         else:
#                             iris_streak_direction = None
#                             iris_streak_count = 0

#                 writer.write(frame)
#                 cv2.imshow("AI Interview Proctoring", frame)

#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     print("\n🛑 Stop requested by user.")
#                     break

#     finally:
#         cap.release()
#         writer.release()
#         cv2.destroyAllWindows()

#         print("🛑 Stopping audio recording...")
#         audio_recorder.stop()
#         audio_recorder.join()

#         actual_fps = frame_count / (time.time() - start_time) if (time.time() - start_time) > 0 else target_fps

#         print("🎬 Combining video and audio files...")
#         combine_audio_video(raw_video_path, audio_path, final_video_path, actual_fps=actual_fps)

#         print("\n" + "═" * 50)
#         print(f"✅ Session Completed!")
#         print(f"   Video: {final_video_path}")
#         print(f"   CSV:   {trigger_csv_path}")
#         print("═" * 50)

# # ─────────────────────────────────────────────────────────────────────────────
# # Entry Point
# # ─────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     selected_image = select_candidate()
#     if selected_image:
#         start_interview_recording(selected_image)





