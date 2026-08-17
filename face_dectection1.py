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
#         print(f"❌ FFmpeg execution failed. Ensure FFmpeg is installed and added to PATH.\n   Video: {video_path}\n   Audio: {audio_path}")

# # ─────────────────────────────────────────────────────────────────────────────
# # Eye Aspect Ratio (EAR) & Blink Helper Functions
# # ─────────────────────────────────────────────────────────────────────────────

# def calculate_ear(landmarks, frame_width, frame_height):
#     """
#     Calculates Eye Aspect Ratio (EAR) for both eyes to detect closures.
#     """
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
# # Black Glasses Detection Helper Function
# # ─────────────────────────────────────────────────────────────────────────────

# DARK_PIXEL_THRESHOLD = 80
# DARK_PIXEL_RATIO = 0.35
# CONTRAST_RATIO = 0.65
# MIN_EYE_BRIGHTNESS = 95

# def detect_black_glasses(landmarks, frame, frame_width, frame_height):
#     LEFT_EYE_INDICES = [33, 133, 159, 145, 158, 153, 144, 160]
#     RIGHT_EYE_INDICES = [362, 263, 386, 374, 385, 380, 373, 387]
#     CHEEK_INDICES = [205, 425]

#     def get_coords(idx_list):
#         pts = []
#         for idx in idx_list:
#             pts.append([int(landmarks[idx].x * frame_width), int(landmarks[idx].y * frame_height)])
#         return np.array(pts, dtype=np.int32)

#     l_pts = get_coords(LEFT_EYE_INDICES)
#     r_pts = get_coords(RIGHT_EYE_INDICES)
#     cheek_pts = get_coords(CHEEK_INDICES)

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     def analyze_eye_region(pts, expand=10):
#         x, y, w, h = cv2.boundingRect(pts)
#         x1, y1 = max(0, x - expand), max(0, y - expand)
#         x2, y2 = min(frame_width, x + w + expand), min(frame_height, y + h + expand)

#         region = gray[y1:y2, x1:x2]
#         if region.size == 0:
#             return 255.0, 0.0

#         mean_brightness = np.mean(region)
#         dark_pixels = np.sum(region < DARK_PIXEL_THRESHOLD)
#         dark_ratio = float(dark_pixels) / float(region.size)

#         return mean_brightness, dark_ratio

#     l_bright, l_dark_ratio = analyze_eye_region(l_pts)
#     r_bright, r_dark_ratio = analyze_eye_region(r_pts)

#     avg_eye_brightness = (l_bright + r_bright) / 2.0
#     avg_dark_pixel_ratio = (l_dark_ratio + r_dark_ratio) / 2.0

#     cheek_val_1 = gray[min(frame_height - 1, cheek_pts[0][1]), min(frame_width - 1, cheek_pts[0][0])]
#     cheek_val_2 = gray[min(frame_height - 1, cheek_pts[1][1]), min(frame_width - 1, cheek_pts[1][0])]
#     skin_baseline = max((float(cheek_val_1) + float(cheek_val_2)) / 2.0, 1.0)

#     calculated_contrast_ratio = avg_eye_brightness / skin_baseline

#     is_glasses = (
#         (avg_dark_pixel_ratio >= DARK_PIXEL_RATIO) and
#         (avg_eye_brightness < MIN_EYE_BRIGHTNESS) and
#         (calculated_contrast_ratio < CONTRAST_RATIO)
#     )

#     return is_glasses

# # ─────────────────────────────────────────────────────────────────────────────
# # Iris Detection Helper Function
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
#     if avg_h_ratio < 0.35:
#         direction = "IRIS LEFT"
#     elif avg_h_ratio > 0.65:
#         direction = "IRIS RIGHT"
#     elif avg_v_ratio < 0.30:
#         direction = "IRIS UP"
#     elif avg_v_ratio > 0.75:
#         direction = "IRIS DOWN"

#     right_iris_pixel = (int(r_iris[0]), int(r_iris[1]))
#     left_iris_pixel  = (int(l_iris[0]), int(l_iris[1]))

#     return direction, right_iris_pixel, left_iris_pixel

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

#     print(f"\n🔍 Encoding reference photo for: {candidate_name} ...")
#     reference_encoding = load_reference_encoding(reference_image_path)
#     if reference_encoding is None:
#         print("Aborting: fix the reference image and try again.")
#         return
#     print(f"✅ Reference photo loaded successfully.")

#     # 1. Directories
#     video_dir = Path("video_interview")
#     snapshot_dir = video_dir / "snapshots"
#     video_dir.mkdir(parents=True, exist_ok=True)
#     snapshot_dir.mkdir(parents=True, exist_ok=True)

#     # 2. Camera Setup
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return

#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     safe_name  = reference_image_path.stem
#     timestamp  = time.strftime('%Y%m%d_%H%M%S')
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

#     # State & Snapshot timers
#     last_snap_time = 0
#     last_verify_time = 0
#     last_movement_snap_time = 0
#     last_iris_snap_time = 0
#     last_glasses_snap_time = 0
#     last_light_snap_time = 0
#     last_eye_closure_snap_time = 0
#     last_no_blink_snap_time = 0

#     # Eye Closure Timers
#     EAR_CLOSED_THRESHOLD = 0.18
#     eye_closed_start_time = None  # Tracks continuous eye closure duration

#     # Liveliness (Blink) Timers
#     NO_BLINK_WARN_SECONDS = 8.0
#     last_blink_timestamp = time.time()

#     glasses_consecutive_frames = 0
#     REQUIRED_GLASSES_FRAMES = 3

#     identity_ok = None
#     font = cv2.FONT_HERSHEY_SIMPLEX

#     frame_count = 0
#     start_time = time.time()

#     print(f"\n🎥 Recording to: {final_video_path}")
#     print(f"   Candidate : {candidate_name}")
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

#                     # Periodically run identity verification on the main face
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
#                                 if not identity_ok:
#                                     print(f"🚨 IDENTITY MISMATCH (distance={distance:.3f})")
#                                     snap_path = snapshot_dir / f"identity_mismatch_{int(current_timestamp)}.jpg"
#                                     cv2.imwrite(str(snap_path), frame)
#                         except Exception as e:
#                             print(f"⚠️ Verification skipped: {e}")

#                 # ── Requirement 3: Color-Coded Bounding Boxes ─────────────
#                 if results.detections:
#                     for idx, det_item in enumerate(results.detections):
#                         b = det_item.location_data.relative_bounding_box
#                         bx, by = int(b.xmin * width), int(b.ymin * height)
#                         bw, bh = int(b.width * width), int(b.height * height)
                        
#                         # First face is candidate (if verified), secondary/unverified faces are red
#                         if idx == 0 and identity_ok is True:
#                             color = (0, 255, 0)  # Green for Candidate
#                             label = f"Candidate: {candidate_name}"
#                         else:
#                             color = (0, 0, 255)  # Red for Other / Unrecognized
#                             label = "Unknown / Other Person"

#                         cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)
#                         cv2.putText(frame, label, (bx, max(20, by - 10)), font, 0.6, color, 2)

#                 # ── 1. Light / Brightness Monitoring (40% - 70%) ─────────────
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
#                     if current_timestamp - last_light_snap_time > 5.0:
#                         snap_path = snapshot_dir / f"light_violation_{int(current_timestamp)}.jpg"
#                         cv2.imwrite(str(snap_path), frame)
#                         print(f"🚨 Lighting Warning ({lighting_warning}) Snapshot saved.")
#                         last_light_snap_time = current_timestamp

#                 # ── 2. People Counter Logic ──────────────────────────────────
#                 if face_count > 1:
#                     cv2.putText(frame, f"WARNING: {face_count} PEOPLE!", (20, 50), font, 1, (0, 0, 255), 3)
#                     cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)

#                     if current_timestamp - last_snap_time > 3.0:
#                         snap_path = snapshot_dir / f"violation_{int(current_timestamp)}.jpg"
#                         cv2.imwrite(str(snap_path), frame)
#                         print(f"🚨 Multiple people detected! Snapshot: {snap_path}")
#                         last_snap_time = current_timestamp

#                 elif face_count == 0:
#                     cv2.putText(frame, "WARNING: NO CANDIDATE", (20, 50), font, 1, (0, 165, 255), 3)
#                     identity_ok = None
#                     glasses_consecutive_frames = 0
#                     eye_closed_start_time = None
#                     last_blink_timestamp = current_timestamp

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
#                     if yaw_ratio < -0.15:
#                         movement_state = "LOOKING RIGHT"
#                     elif yaw_ratio > 0.15:
#                         movement_state = "LOOKING LEFT"
#                     elif pitch_ratio < 0.45:
#                         movement_state = "LOOKING UP"
#                     elif pitch_ratio > 0.60:
#                         movement_state = "LOOKING DOWN"

#                     if movement_state:
#                         cv2.putText(frame, f"WARNING: {movement_state}", (20, 90), font, 1, (0, 165, 255), 3)
#                         if current_timestamp - last_movement_snap_time > 3.0:
#                             snap_path = snapshot_dir / f"movement_violation_{int(current_timestamp)}.jpg"
#                             cv2.imwrite(str(snap_path), frame)
#                             print(f"🚨 Face Movement Detected ({movement_state})!")
#                             last_movement_snap_time = current_timestamp

#                     # ── Black Glasses, Iris & Eye Closure Tracking ────────────
#                     mesh_results = face_mesh.process(rgb_frame)
#                     if mesh_results.multi_face_landmarks:
#                         face_landmarks = mesh_results.multi_face_landmarks[0]

#                         # ── Requirement 2: Black Glasses Check ONLY ─────────
#                         is_wearing_glasses = detect_black_glasses(face_landmarks.landmark, frame, width, height)
#                         if is_wearing_glasses:
#                             glasses_consecutive_frames += 1
#                         else:
#                             glasses_consecutive_frames = max(0, glasses_consecutive_frames - 1)

#                         if glasses_consecutive_frames >= REQUIRED_GLASSES_FRAMES:
#                             cv2.putText(frame, "WARN: EYE NOT VISIBLE", (20, 170), font, 1, (0, 0, 255), 3)
#                             if current_timestamp - last_glasses_snap_time > 3.0:
#                                 snap_path = snapshot_dir / f"glasses_violation_{int(current_timestamp)}.jpg"
#                                 cv2.imwrite(str(snap_path), frame)
#                                 print(f"🚨 Black Glasses Detected!")
#                                 last_glasses_snap_time = current_timestamp

#                         # ── Requirement 1: Eye Closed Warning (> 5 seconds) ──
#                         ear = calculate_ear(face_landmarks.landmark, width, height)
#                         if ear < EAR_CLOSED_THRESHOLD:
#                             if eye_closed_start_time is None:
#                                 eye_closed_start_time = current_timestamp
#                                 last_blink_timestamp = current_timestamp  # Register blink moment
                            
#                             closed_duration = current_timestamp - eye_closed_start_time
                            
#                             if closed_duration >= 5.0:
#                                 cv2.putText(frame, "WARN: EYE CLOSED (>5s)", (20, 250), font, 0.9, (0, 0, 255), 3)

#                                 if current_timestamp - last_eye_closure_snap_time > 3.0:
#                                     snap_path = snapshot_dir / f"eye_closed_{int(current_timestamp)}.jpg"
#                                     cv2.imwrite(str(snap_path), frame)
#                                     print(f"🚨 Eye Closed for >5 Seconds Detected! Snapshot: {snap_path}")
#                                     last_eye_closure_snap_time = current_timestamp
#                         else:
#                             eye_closed_start_time = None  # Reset timer when eyes reopen

#                         # ── Requirement 4: Liveliness (No-Blink Warning, strict 8s) ──
#                         time_since_last_blink = current_timestamp - last_blink_timestamp
#                         if time_since_last_blink >= NO_BLINK_WARN_SECONDS:
#                             cv2.putText(frame, "WARN: PERSON NOT BLINKING (>8s)", (20, 290), font, 0.9, (0, 0, 255), 3)
#                             cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 4)
#                             if current_timestamp - last_no_blink_snap_time > 3.0:
#                                 snap_path = snapshot_dir / f"no_blink_violation_{int(current_timestamp)}.jpg"
#                                 cv2.imwrite(str(snap_path), frame)
#                                 print(f"🚨 Candidate does not blink his eye! No blink for >{NO_BLINK_WARN_SECONDS}s. Snapshot: {snap_path}")
#                                 last_no_blink_snap_time = current_timestamp
#                         else:
#                             cv2.putText(frame, f"No Blink: {time_since_last_blink:.1f}s / {NO_BLINK_WARN_SECONDS:.0f}s", (20, 290), font, 0.6, (0, 255, 255), 2)

#                         # Iris Direction Tracking
#                         iris_dir, r_iris_pt, l_iris_pt = detect_iris_direction(face_landmarks.landmark, width, height)
#                         cv2.circle(frame, r_iris_pt, 2, (0, 255, 0), -1)
#                         cv2.circle(frame, l_iris_pt, 2, (0, 255, 0), -1)

#                         if iris_dir:
#                             cv2.putText(frame, f"WARNING: {iris_dir}", (20, 130), font, 1, (0, 165, 255), 3)
#                             if current_timestamp - last_iris_snap_time > 3.0:
#                                 snap_path = snapshot_dir / f"iris_violation_{int(current_timestamp)}.jpg"
#                                 cv2.imwrite(str(snap_path), frame)
#                                 print(f"🚨 Iris Movement Detected ({iris_dir})!")
#                                 last_iris_snap_time = current_timestamp

#                     # ── Identity Overlay ──────────────────────────────────────
#                     if identity_ok is True:
#                         cv2.putText(frame, f"Verified: {candidate_name}", (20, 50), font, 0.9, (0, 255, 0), 2)
#                     elif identity_ok is False:
#                         cv2.putText(frame, "WARNING: UNRECOGNIZED PERSON", (20, 50), font, 1, (0, 0, 255), 3)
#                         cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)
#                     else:
#                         cv2.putText(frame, "Verifying...", (20, 50), font, 1, (0, 165, 255), 2)

#                 # Record & Display
#                 writer.write(frame)
#                 cv2.imshow('Interview Proctoring System', frame)

#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

#     except KeyboardInterrupt:
#         print("\nStopped via terminal (Ctrl+C).")
#     finally:
#         elapsed_time = time.time() - start_time
#         actual_fps = (frame_count / elapsed_time) if elapsed_time > 0 else target_fps

#         cap.release()
#         writer.release()
#         cv2.destroyAllWindows()

#         audio_recorder.stop()
#         audio_recorder.join()

#         print("\n🎬 Processing final recorded interview video and audio...")
#         combine_audio_video(raw_video_path, audio_path, final_video_path, actual_fps=actual_fps)
#         print(f"✅ Interview finished. Video saved to: {final_video_path}")

# # ─────────────────────────────────────────────────────────────────────────────
# # Entry Point
# # ─────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     chosen = select_candidate()
#     if chosen is not None:
#         start_interview_recording(reference_image_path=chosen)



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

#     # FFmpeg command to cleanly merge video and audio
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
#         result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
#         if output_path.exists() and output_path.stat().st_size > 0:
#             video_path.unlink(missing_ok=True)
#             audio_path.unlink(missing_ok=True)
#             print(f"✅ Synchronized video + audio saved to: {output_path}")
#             return
#     except Exception as e:
#         print(f"⚠️ Primary FFmpeg muxing failed ({e}). Trying full re-encode...")

#     # Secondary re-encode attempt if fast copy failed
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
#         print(f"❌ FFmpeg execution failed. Ensure FFmpeg is installed and added to PATH.\n   Video: {video_path}\n   Audio: {audio_path}")


# # ─────────────────────────────────────────────────────────────────────────────
# # Eye Aspect Ratio (EAR) & Blink Helper Functions
# # ─────────────────────────────────────────────────────────────────────────────

# def calculate_ear(landmarks, frame_width, frame_height):
#     """
#     Calculates Eye Aspect Ratio (EAR) for both eyes to detect closures & blinks.
#     EAR < 0.18 indicates closed eyes.
#     """
#     # MediaPipe FaceMesh Eye Landmarks
#     # Left eye: vertical (159, 145), (158, 153), horizontal (33, 133)
#     # Right eye: vertical (386, 374), (385, 380), horizontal (362, 263)

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
# # Black Glasses Detection Helper Function
# # ─────────────────────────────────────────────────────────────────────────────

# DARK_PIXEL_THRESHOLD = 80
# DARK_PIXEL_RATIO = 0.35
# CONTRAST_RATIO = 0.65
# MIN_EYE_BRIGHTNESS = 95


# def detect_black_glasses(landmarks, frame, frame_width, frame_height):
#     LEFT_EYE_INDICES = [33, 133, 159, 145, 158, 153, 144, 160]
#     RIGHT_EYE_INDICES = [362, 263, 386, 374, 385, 380, 373, 387]
#     CHEEK_INDICES = [205, 425]

#     def get_coords(idx_list):
#         pts = []
#         for idx in idx_list:
#             pts.append([int(landmarks[idx].x * frame_width), int(landmarks[idx].y * frame_height)])
#         return np.array(pts, dtype=np.int32)

#     l_pts = get_coords(LEFT_EYE_INDICES)
#     r_pts = get_coords(RIGHT_EYE_INDICES)
#     cheek_pts = get_coords(CHEEK_INDICES)

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     def analyze_eye_region(pts, expand=10):
#         x, y, w, h = cv2.boundingRect(pts)
#         x1, y1 = max(0, x - expand), max(0, y - expand)
#         x2, y2 = min(frame_width, x + w + expand), min(frame_height, y + h + expand)

#         region = gray[y1:y2, x1:x2]
#         if region.size == 0:
#             return 255.0, 0.0

#         mean_brightness = np.mean(region)
#         dark_pixels = np.sum(region < DARK_PIXEL_THRESHOLD)
#         dark_ratio = float(dark_pixels) / float(region.size)

#         return mean_brightness, dark_ratio

#     l_bright, l_dark_ratio = analyze_eye_region(l_pts)
#     r_bright, r_dark_ratio = analyze_eye_region(r_pts)

#     avg_eye_brightness = (l_bright + r_bright) / 2.0
#     avg_dark_pixel_ratio = (l_dark_ratio + r_dark_ratio) / 2.0

#     cheek_val_1 = gray[min(frame_height - 1, cheek_pts[0][1]), min(frame_width - 1, cheek_pts[0][0])]
#     cheek_val_2 = gray[min(frame_height - 1, cheek_pts[1][1]), min(frame_width - 1, cheek_pts[1][0])]
#     skin_baseline = max((float(cheek_val_1) + float(cheek_val_2)) / 2.0, 1.0)

#     calculated_contrast_ratio = avg_eye_brightness / skin_baseline

#     is_glasses = (
#         (avg_dark_pixel_ratio >= DARK_PIXEL_RATIO) and
#         (avg_eye_brightness < MIN_EYE_BRIGHTNESS) and
#         (calculated_contrast_ratio < CONTRAST_RATIO)
#     )

#     return is_glasses


# # ─────────────────────────────────────────────────────────────────────────────
# # Iris Detection Helper Function
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
#     if avg_h_ratio < 0.35:
#         direction = "IRIS LEFT"
#     elif avg_h_ratio > 0.65:
#         direction = "IRIS RIGHT"
#     elif avg_v_ratio < 0.30:
#         direction = "IRIS UP"
#     elif avg_v_ratio > 0.75:
#         direction = "IRIS DOWN"

#     right_iris_pixel = (int(r_iris[0]), int(r_iris[1]))
#     left_iris_pixel  = (int(l_iris[0]), int(l_iris[1]))

#     return direction, right_iris_pixel, left_iris_pixel


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

#     print(f"\n🔍 Encoding reference photo for: {candidate_name} ...")
#     reference_encoding = load_reference_encoding(reference_image_path)
#     if reference_encoding is None:
#         print("Aborting: fix the reference image and try again.")
#         return
#     print(f"✅ Reference photo loaded successfully.")

#     # 1. Directories
#     video_dir = Path("video_interview")
#     snapshot_dir = video_dir / "snapshots"
#     video_dir.mkdir(parents=True, exist_ok=True)
#     snapshot_dir.mkdir(parents=True, exist_ok=True)

#     # 2. Camera Setup
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return

#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     safe_name  = reference_image_path.stem
#     timestamp  = time.strftime('%Y%m%d_%H%M%S')
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

#     # State & Snapshot timers
#     last_snap_time = 0
#     last_verify_time = 0
#     last_movement_snap_time = 0
#     last_iris_snap_time = 0
#     last_glasses_snap_time = 0
#     last_light_snap_time = 0
#     last_eye_closure_snap_time = 0
#     last_no_blink_snap_time = 0

#     # Eye Blink & Closure Timers
#     last_blink_timestamp = time.time()
#     EAR_CLOSED_THRESHOLD = 0.18
#     NO_BLINK_WARN_SECONDS = 6.0

#     glasses_consecutive_frames = 0
#     REQUIRED_GLASSES_FRAMES = 3

#     identity_ok = None
#     font = cv2.FONT_HERSHEY_SIMPLEX

#     frame_count = 0
#     start_time = time.time()

#     print(f"\n🎥 Recording to: {final_video_path}")
#     print(f"   Candidate : {candidate_name}")
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
#                 rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results    = face_det.process(rgb_frame)
#                 face_count = len(results.detections) if results.detections else 0

#                 face_box = None
#                 if face_count > 0:
#                     det = results.detections[0]
#                     box = det.location_data.relative_bounding_box
#                     x, y = int(box.xmin * width), int(box.ymin * height)
#                     w_box, h_box = int(box.width * width), int(box.height * height)
#                     face_box = (x, y, w_box, h_box)

#                     for det_item in results.detections:
#                         b = det_item.location_data.relative_bounding_box
#                         bx, by = int(b.xmin * width), int(b.ymin * height)
#                         bw, bh = int(b.width * width), int(b.height * height)
#                         cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 255, 0), 2)

#                 # ── 1. Light / Brightness Monitoring (40% - 70%) ─────────────
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
#                     if current_timestamp - last_light_snap_time > 5.0:
#                         snap_path = snapshot_dir / f"light_violation_{int(current_timestamp)}.jpg"
#                         cv2.imwrite(str(snap_path), frame)
#                         print(f"🚨 Lighting Warning ({lighting_warning}) Snapshot saved.")
#                         last_light_snap_time = current_timestamp

#                 # ── 2. People Counter Logic ──────────────────────────────────
#                 if face_count > 1:
#                     cv2.putText(frame, f"WARNING: {face_count} PEOPLE!", (20, 50), font, 1, (0, 0, 255), 3)
#                     cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)

#                     if current_timestamp - last_snap_time > 3.0:
#                         snap_path = snapshot_dir / f"violation_{int(current_timestamp)}.jpg"
#                         cv2.imwrite(str(snap_path), frame)
#                         print(f"🚨 Multiple people detected! Snapshot: {snap_path}")
#                         last_snap_time = current_timestamp

#                 elif face_count == 0:
#                     cv2.putText(frame, "WARNING: NO CANDIDATE", (20, 50), font, 1, (0, 165, 255), 3)
#                     identity_ok = None
#                     glasses_consecutive_frames = 0

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
#                     if yaw_ratio < -0.15:
#                         movement_state = "LOOKING RIGHT"
#                     elif yaw_ratio > 0.15:
#                         movement_state = "LOOKING LEFT"
#                     elif pitch_ratio < 0.45:
#                         movement_state = "LOOKING UP"
#                     elif pitch_ratio > 0.60:
#                         movement_state = "LOOKING DOWN"

#                     if movement_state:
#                         cv2.putText(frame, f"WARNING: {movement_state}", (20, 90), font, 1, (0, 165, 255), 3)
#                         if current_timestamp - last_movement_snap_time > 3.0:
#                             snap_path = snapshot_dir / f"movement_violation_{int(current_timestamp)}.jpg"
#                             cv2.imwrite(str(snap_path), frame)
#                             print(f"🚨 Face Movement Detected ({movement_state})!")
#                             last_movement_snap_time = current_timestamp

#                     # ── Black Glasses, Iris & Eye Closure/Blink Tracking ────
#                     mesh_results = face_mesh.process(rgb_frame)
#                     if mesh_results.multi_face_landmarks:
#                         face_landmarks = mesh_results.multi_face_landmarks[0]

#                         # 1. Black Glasses Check
#                         is_wearing_glasses = detect_black_glasses(face_landmarks.landmark, frame, width, height)
#                         if is_wearing_glasses:
#                             glasses_consecutive_frames += 1
#                         else:
#                             glasses_consecutive_frames = max(0, glasses_consecutive_frames - 1)

#                         if glasses_consecutive_frames >= REQUIRED_GLASSES_FRAMES:
#                             cv2.putText(frame, "WARN: EYE NOT VISIBLE", (20, 170), font, 1, (0, 0, 255), 3)
#                             if current_timestamp - last_glasses_snap_time > 3.0:
#                                 snap_path = snapshot_dir / f"glasses_violation_{int(current_timestamp)}.jpg"
#                                 cv2.imwrite(str(snap_path), frame)
#                                 print(f"🚨 Black Glasses Detected!")
#                                 last_glasses_snap_time = current_timestamp

#                         # 2. Eye Aspect Ratio (Closure & Blink Detection)
#                         ear = calculate_ear(face_landmarks.landmark, width, height)
#                         if ear < EAR_CLOSED_THRESHOLD:
#                             cv2.putText(frame, "WARN: EYE CLOSED", (20, 250), font, 0.9, (0, 0, 255), 3)
#                             last_blink_timestamp = current_timestamp  # Register blink moment

#                             if current_timestamp - last_eye_closure_snap_time > 3.0:
#                                 snap_path = snapshot_dir / f"eye_closed_{int(current_timestamp)}.jpg"
#                                 cv2.imwrite(str(snap_path), frame)
#                                 print(f"🚨 Eye Closed Detected! Snapshot: {snap_path}")
#                                 last_eye_closure_snap_time = current_timestamp
#                         else:
#                             # Candidate has eyes open; check duration since last blink
#                             time_since_last_blink = current_timestamp - last_blink_timestamp
#                             if time_since_last_blink >= NO_BLINK_WARN_SECONDS:
#                                 cv2.putText(frame, "WARN: CANDIDATE NOT BLINKING EYE", (20, 280), font, 0.8, (0, 0, 255), 3)
#                                 if current_timestamp - last_no_blink_snap_time > 5.0:
#                                     snap_path = snapshot_dir / f"no_blink_violation_{int(current_timestamp)}.jpg"
#                                     cv2.imwrite(str(snap_path), frame)
#                                     print(f"🚨 Candidate Not Blinking Warning (> {NO_BLINK_WARN_SECONDS}s)!")
#                                     last_no_blink_snap_time = current_timestamp

#                         # 3. Iris Direction Tracking
#                         iris_dir, r_iris_pt, l_iris_pt = detect_iris_direction(face_landmarks.landmark, width, height)
#                         cv2.circle(frame, r_iris_pt, 2, (0, 255, 0), -1)
#                         cv2.circle(frame, l_iris_pt, 2, (0, 255, 0), -1)

#                         if iris_dir:
#                             cv2.putText(frame, f"WARNING: {iris_dir}", (20, 130), font, 1, (0, 165, 255), 3)
#                             if current_timestamp - last_iris_snap_time > 3.0:
#                                 snap_path = snapshot_dir / f"iris_violation_{int(current_timestamp)}.jpg"
#                                 cv2.imwrite(str(snap_path), frame)
#                                 print(f"🚨 Iris Movement Detected ({iris_dir})!")
#                                 last_iris_snap_time = current_timestamp

#                     # ── Identity Verification ────────────────────────────────
#                     if current_timestamp - last_verify_time > verify_every_seconds:
#                         last_verify_time = current_timestamp
#                         x, y, w_box, h_box = face_box
#                         top, left   = max(0, y),          max(0, x)
#                         bottom, right = min(height, y+h_box), min(width, x+w_box)

#                         try:
#                             encodings = face_recognition.face_encodings(
#                                 rgb_frame,
#                                 known_face_locations=[(top, right, bottom, left)]
#                             )
#                             if encodings:
#                                 distance   = face_recognition.face_distance([reference_encoding], encodings[0])[0]
#                                 identity_ok = bool(distance <= match_tolerance)
#                                 if not identity_ok:
#                                     print(f"🚨 IDENTITY MISMATCH (distance={distance:.3f})")
#                                     snap_path = snapshot_dir / f"identity_mismatch_{int(current_timestamp)}.jpg"
#                                     cv2.imwrite(str(snap_path), frame)
#                         except Exception as e:
#                             print(f"⚠️ Verification skipped: {e}")

#                     # ── Identity Overlay ──────────────────────────────────────
#                     if identity_ok is True:
#                         cv2.putText(frame, f"Verified: {candidate_name}", (20, 50), font, 0.9, (0, 255, 0), 2)
#                     elif identity_ok is False:
#                         cv2.putText(frame, "WARNING: UNRECOGNIZED PERSON", (20, 50), font, 1, (0, 0, 255), 3)
#                         cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)
#                     else:
#                         cv2.putText(frame, "Verifying...", (20, 50), font, 1, (0, 165, 255), 2)

#                 # Record & Display
#                 writer.write(frame)
#                 cv2.imshow('Interview Proctoring System', frame)

#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

#     except KeyboardInterrupt:
#         print("\nStopped via terminal (Ctrl+C).")
#     finally:
#         elapsed_time = time.time() - start_time
#         actual_fps = (frame_count / elapsed_time) if elapsed_time > 0 else target_fps

#         cap.release()
#         writer.release()
#         cv2.destroyAllWindows()

#         audio_recorder.stop()
#         audio_recorder.join()

#         print("\n🎬 Processing final recorded interview video and audio...")
#         combine_audio_video(raw_video_path, audio_path, final_video_path, actual_fps=actual_fps)
#         print(f"✅ Interview finished. Video saved to: {final_video_path}")


# # ─────────────────────────────────────────────────────────────────────────────
# # Entry Point
# # ─────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     chosen = select_candidate()
#     if chosen is not None:
#         start_interview_recording(reference_image_path=chosen)









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
#     """Muxes audio and video together using ffmpeg with sync alignment."""
#     if not audio_path.exists() or audio_path.stat().st_size == 0:
#         print("⚠️ Audio track empty or missing. Keeping video file as-is.")
#         return

#     cmd = [
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
#         subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
#         if output_path.exists() and output_path.stat().st_size > 0:
#             video_path.unlink(missing_ok=True)
#             audio_path.unlink(missing_ok=True)
#             print(f"✅ Synchronized video + audio saved to: {output_path}")
#     except Exception as e:
#         print(f"⚠️ Could not mux audio with video via ffmpeg ({e}). Video and audio remain separate.")


# # ─────────────────────────────────────────────────────────────────────────────
# # Lighting & Brightness Monitoring Helper
# # ─────────────────────────────────────────────────────────────────────────────

# # Brightness range settings (50% to 60%)
# MIN_BRIGHTNESS_PCT = 40.0
# MAX_BRIGHTNESS_PCT = 50.0


# def analyze_lighting_conditions(frame, face_box=None):
#     """
#     Calculates percentage brightness for both Face (Front Light) and Background regions.
#     Returns: (face_brightness_pct, bg_brightness_pct)
#     """
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     h, w = gray.shape

#     if face_box is None:
#         # If no face detected, analyze total frame as background
#         overall_avg = np.mean(gray)
#         overall_pct = (overall_avg / 255.0) * 100.0
#         return None, overall_pct

#     x, y, bw, bh = face_box
#     x1, y1 = max(0, x), max(0, y)
#     x2, y2 = min(w, x + bw), min(h, y + bh)

#     # 1. Extract Face Region
#     face_region = gray[y1:y2, x1:x2]
#     face_brightness_pct = ((np.mean(face_region) / 255.0) * 100.0) if face_region.size > 0 else None

#     # 2. Extract Background Region (Frame minus Face Box)
#     bg_mask = np.ones((h, w), dtype=bool)
#     bg_mask[y1:y2, x1:x2] = False
#     bg_pixels = gray[bg_mask]
#     bg_brightness_pct = ((np.mean(bg_pixels) / 255.0) * 100.0) if bg_pixels.size > 0 else None

#     return face_brightness_pct, bg_brightness_pct


# # ─────────────────────────────────────────────────────────────────────────────
# # Black Glasses Detection Helper Function
# # ─────────────────────────────────────────────────────────────────────────────

# DARK_PIXEL_THRESHOLD = 80
# DARK_PIXEL_RATIO = 0.35
# CONTRAST_RATIO = 0.65
# MIN_EYE_BRIGHTNESS = 95


# def detect_black_glasses(landmarks, frame, frame_width, frame_height):
#     """
#     Analyzes both eye regions using pixel dark-ratio and skin contrast thresholds.
#     """
#     LEFT_EYE_INDICES = [33, 133, 159, 145, 158, 153, 144, 160]
#     RIGHT_EYE_INDICES = [362, 263, 386, 374, 385, 380, 373, 387]
#     CHEEK_INDICES = [205, 425]

#     def get_coords(idx_list):
#         pts = []
#         for idx in idx_list:
#             pts.append([int(landmarks[idx].x * frame_width), int(landmarks[idx].y * frame_height)])
#         return np.array(pts, dtype=np.int32)

#     l_pts = get_coords(LEFT_EYE_INDICES)
#     r_pts = get_coords(RIGHT_EYE_INDICES)
#     cheek_pts = get_coords(CHEEK_INDICES)

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     def analyze_eye_region(pts, expand=10):
#         x, y, w, h = cv2.boundingRect(pts)
#         x1, y1 = max(0, x - expand), max(0, y - expand)
#         x2, y2 = min(frame_width, x + w + expand), min(frame_height, y + h + expand)

#         region = gray[y1:y2, x1:x2]
#         if region.size == 0:
#             return 255.0, 0.0

#         mean_brightness = np.mean(region)
#         dark_pixels = np.sum(region < DARK_PIXEL_THRESHOLD)
#         dark_ratio = float(dark_pixels) / float(region.size)

#         return mean_brightness, dark_ratio

#     l_bright, l_dark_ratio = analyze_eye_region(l_pts)
#     r_bright, r_dark_ratio = analyze_eye_region(r_pts)

#     avg_eye_brightness = (l_bright + r_bright) / 2.0
#     avg_dark_pixel_ratio = (l_dark_ratio + r_dark_ratio) / 2.0

#     cheek_val_1 = gray[min(frame_height - 1, cheek_pts[0][1]), min(frame_width - 1, cheek_pts[0][0])]
#     cheek_val_2 = gray[min(frame_height - 1, cheek_pts[1][1]), min(frame_width - 1, cheek_pts[1][0])]
#     skin_baseline = max((float(cheek_val_1) + float(cheek_val_2)) / 2.0, 1.0)

#     calculated_contrast_ratio = avg_eye_brightness / skin_baseline

#     is_glasses = (
#         (avg_dark_pixel_ratio >= DARK_PIXEL_RATIO) and
#         (avg_eye_brightness < MIN_EYE_BRIGHTNESS) and
#         (calculated_contrast_ratio < CONTRAST_RATIO)
#     )

#     return is_glasses


# # ─────────────────────────────────────────────────────────────────────────────
# # Iris Detection Helper Function
# # ─────────────────────────────────────────────────────────────────────────────

# def detect_iris_direction(landmarks, frame_width, frame_height):
#     """Detects iris position relative to eye corners and eyelids."""
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
#     if avg_h_ratio < 0.35:
#         direction = "IRIS LEFT"
#     elif avg_h_ratio > 0.65:
#         direction = "IRIS RIGHT"
#     elif avg_v_ratio < 0.30:
#         direction = "IRIS UP"
#     elif avg_v_ratio > 0.75:
#         direction = "IRIS DOWN"

#     right_iris_pixel = (int(r_iris[0]), int(r_iris[1]))
#     left_iris_pixel  = (int(l_iris[0]), int(l_iris[1]))

#     return direction, right_iris_pixel, left_iris_pixel


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

#     print(f"\n🔍 Encoding reference photo for: {candidate_name} ...")
#     reference_encoding = load_reference_encoding(reference_image_path)
#     if reference_encoding is None:
#         print("Aborting: fix the reference image and try again.")
#         return
#     print(f"✅ Reference photo loaded successfully.")

#     # 1. Directories
#     video_dir = Path("video_interview")
#     snapshot_dir = video_dir / "snapshots"
#     video_dir.mkdir(parents=True, exist_ok=True)
#     snapshot_dir.mkdir(parents=True, exist_ok=True)

#     # 2. Camera Setup
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return

#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     safe_name  = reference_image_path.stem
#     timestamp  = time.strftime('%Y%m%d_%H%M%S')
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

#     # State & Snapshot timers
#     last_snap_time = 0
#     last_verify_time = 0
#     last_movement_snap_time = 0
#     last_iris_snap_time = 0
#     last_glasses_snap_time = 0
#     last_light_snap_time = 0

#     glasses_consecutive_frames = 0
#     REQUIRED_GLASSES_FRAMES = 3

#     identity_ok = None
#     font = cv2.FONT_HERSHEY_SIMPLEX

#     frame_count = 0
#     start_time = time.time()

#     print(f"\n🎥 Recording to: {final_video_path}")
#     print(f"   Candidate : {candidate_name}")
#     print("   Press 'q' in the video window to stop.\n")

#     try:
#         with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_det, \
#              mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:

#             while cap.isOpened():
#                 success, frame = cap.read()
#                 if not success:
#                     break

#                 frame_count += 1
#                 rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results    = face_det.process(rgb_frame)
#                 face_count = len(results.detections) if results.detections else 0

#                 face_box = None
#                 if face_count > 0:
#                     det = results.detections[0]
#                     box = det.location_data.relative_bounding_box
#                     x, y = int(box.xmin * width), int(box.ymin * height)
#                     w_box, h_box = int(box.width * width), int(box.height * height)
#                     face_box = (x, y, w_box, h_box)

#                     for det_item in results.detections:
#                         b = det_item.location_data.relative_bounding_box
#                         bx, by = int(b.xmin * width), int(b.ymin * height)
#                         bw, bh = int(b.width * width), int(b.height * height)
#                         cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 255, 0), 2)

#                 # ── 1. Light / Brightness Monitoring (40% - 70%) ─────────────
#                 face_pct, bg_pct = analyze_lighting_conditions(frame, face_box)

#                 # HUD Info on top right
#                 hud_y = 30
#                 if face_pct is not None:
#                     cv2.putText(frame, f"Face Light: {face_pct:.1f}%", (width - 230, hud_y), font, 0.6, (255, 255, 255), 2)
#                     hud_y += 25
#                 if bg_pct is not None:
#                     cv2.putText(frame, f"BG Light:   {bg_pct:.1f}%", (width - 230, hud_y), font, 0.6, (255, 255, 255), 2)

#                 lighting_warning = None
                
#                 # Check Face / Front Light
#                 if face_pct is not None:
#                     if face_pct < MIN_BRIGHTNESS_PCT:
#                         lighting_warning = "WARN: FRONT LIGHT LOW"
#                     elif face_pct > MAX_BRIGHTNESS_PCT:
#                         lighting_warning = "WARN: FRONT LIGHT TOO HIGH - USE NORMAL LIGHT"

#                 # Check Background Light
#                 if not lighting_warning and bg_pct is not None:
#                     if bg_pct < MIN_BRIGHTNESS_PCT:
#                         lighting_warning = "WARN: BACKGROUND LIGHT IS LOW"
#                     elif bg_pct > MAX_BRIGHTNESS_PCT:
#                         lighting_warning = "WARN: BACKGROUND LIGHT HIGH - SIT IN NORMAL LIGHT"

#                 if lighting_warning:
#                     cv2.putText(frame, lighting_warning, (20, 210), font, 0.75, (0, 165, 255), 2)
#                     if time.time() - last_light_snap_time > 5.0:
#                         snap_path = snapshot_dir / f"light_violation_{int(time.time())}.jpg"
#                         cv2.imwrite(str(snap_path), frame)
#                         print(f"🚨 Lighting Warning ({lighting_warning}) Snapshot saved.")
#                         last_light_snap_time = time.time()

#                 # ── 2. People Counter Logic ──────────────────────────────────
#                 if face_count > 1:
#                     cv2.putText(frame, f"WARNING: {face_count} PEOPLE!", (20, 50), font, 1, (0, 0, 255), 3)
#                     cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)

#                     if time.time() - last_snap_time > 3.0:
#                         snap_path = snapshot_dir / f"violation_{int(time.time())}.jpg"
#                         cv2.imwrite(str(snap_path), frame)
#                         print(f"🚨 Multiple people detected! Snapshot: {snap_path}")
#                         last_snap_time = time.time()

#                 elif face_count == 0:
#                     cv2.putText(frame, "WARNING: NO CANDIDATE", (20, 50), font, 1, (0, 165, 255), 3)
#                     identity_ok = None
#                     glasses_consecutive_frames = 0

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
#                     if yaw_ratio < -0.15:
#                         movement_state = "LOOKING RIGHT"
#                     elif yaw_ratio > 0.15:
#                         movement_state = "LOOKING LEFT"
#                     elif pitch_ratio < 0.45:
#                         movement_state = "LOOKING UP"
#                     elif pitch_ratio > 0.60:
#                         movement_state = "LOOKING DOWN"

#                     if movement_state:
#                         cv2.putText(frame, f"WARNING: {movement_state}", (20, 90), font, 1, (0, 165, 255), 3)
#                         if time.time() - last_movement_snap_time > 3.0:
#                             snap_path = snapshot_dir / f"movement_violation_{int(time.time())}.jpg"
#                             cv2.imwrite(str(snap_path), frame)
#                             print(f"🚨 Face Movement Detected ({movement_state})!")
#                             last_movement_snap_time = time.time()

#                     # ── Black Glasses & Iris Tracking ─────────────────────────
#                     mesh_results = face_mesh.process(rgb_frame)
#                     if mesh_results.multi_face_landmarks:
#                         face_landmarks = mesh_results.multi_face_landmarks[0]

#                         is_wearing_glasses = detect_black_glasses(face_landmarks.landmark, frame, width, height)
#                         if is_wearing_glasses:
#                             glasses_consecutive_frames += 1
#                         else:
#                             glasses_consecutive_frames = max(0, glasses_consecutive_frames - 1)

#                         if glasses_consecutive_frames >= REQUIRED_GLASSES_FRAMES:
#                             cv2.putText(frame, "WARN: EYE NOT VISIBLE", (20, 170), font, 1, (0, 0, 255), 3)
#                             if time.time() - last_glasses_snap_time > 3.0:
#                                 snap_path = snapshot_dir / f"glasses_violation_{int(time.time())}.jpg"
#                                 cv2.imwrite(str(snap_path), frame)
#                                 print(f"🚨 Black Glasses Detected!")
#                                 last_glasses_snap_time = time.time()

#                         # Iris Tracking
#                         iris_dir, r_iris_pt, l_iris_pt = detect_iris_direction(face_landmarks.landmark, width, height)
#                         cv2.circle(frame, r_iris_pt, 2, (0, 255, 0), -1)
#                         cv2.circle(frame, l_iris_pt, 2, (0, 255, 0), -1)

#                         if iris_dir:
#                             cv2.putText(frame, f"WARNING: {iris_dir}", (20, 130), font, 1, (0, 165, 255), 3)
#                             if time.time() - last_iris_snap_time > 3.0:
#                                 snap_path = snapshot_dir / f"iris_violation_{int(time.time())}.jpg"
#                                 cv2.imwrite(str(snap_path), frame)
#                                 print(f"🚨 Iris Movement Detected ({iris_dir})!")
#                                 last_iris_snap_time = time.time()

#                     # ── Identity Verification ────────────────────────────────
#                     if time.time() - last_verify_time > verify_every_seconds:
#                         last_verify_time = time.time()
#                         x, y, w_box, h_box = face_box
#                         top, left   = max(0, y),          max(0, x)
#                         bottom, right = min(height, y+h_box), min(width, x+w_box)

#                         try:
#                             encodings = face_recognition.face_encodings(
#                                 rgb_frame,
#                                 known_face_locations=[(top, right, bottom, left)]
#                             )
#                             if encodings:
#                                 distance   = face_recognition.face_distance([reference_encoding], encodings[0])[0]
#                                 identity_ok = bool(distance <= match_tolerance)
#                                 if not identity_ok:
#                                     print(f"🚨 IDENTITY MISMATCH (distance={distance:.3f})")
#                                     snap_path = snapshot_dir / f"identity_mismatch_{int(time.time())}.jpg"
#                                     cv2.imwrite(str(snap_path), frame)
#                         except Exception as e:
#                             print(f"⚠️ Verification skipped: {e}")

#                     # ── Identity Overlay ──────────────────────────────────────
#                     if identity_ok is True:
#                         cv2.putText(frame, f"Verified: {candidate_name}", (20, 50), font, 0.9, (0, 255, 0), 2)
#                     elif identity_ok is False:
#                         cv2.putText(frame, "WARNING: UNRECOGNIZED PERSON", (20, 50), font, 1, (0, 0, 255), 3)
#                         cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)
#                     else:
#                         cv2.putText(frame, "Verifying...", (20, 50), font, 1, (0, 165, 255), 2)

#                 # Record & Display
#                 writer.write(frame)
#                 cv2.imshow('Interview Proctoring System', frame)

#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

#     except KeyboardInterrupt:
#         print("\nStopped via terminal (Ctrl+C).")
#     finally:
#         elapsed_time = time.time() - start_time
#         actual_fps = (frame_count / elapsed_time) if elapsed_time > 0 else target_fps

#         cap.release()
#         writer.release()
#         cv2.destroyAllWindows()

#         audio_recorder.stop()
#         audio_recorder.join()

#         print("\n🎬 Processing final recorded interview video and audio...")
#         combine_audio_video(raw_video_path, audio_path, final_video_path, actual_fps=actual_fps)
#         print(f"✅ Interview finished. Video saved to: {final_video_path}")


# # ─────────────────────────────────────────────────────────────────────────────
# # Entry Point
# # ─────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     chosen = select_candidate()
#     if chosen is not None:
#         start_interview_recording(reference_image_path=chosen)



# it is working good wed 

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
#     """Muxes audio and video together using ffmpeg with sync alignment."""
#     if not audio_path.exists() or audio_path.stat().st_size == 0:
#         print("⚠️ Audio track empty or missing. Keeping video file as-is.")
#         return

#     cmd = [
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
#         subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
#         if output_path.exists() and output_path.stat().st_size > 0:
#             video_path.unlink(missing_ok=True)
#             audio_path.unlink(missing_ok=True)
#             print(f"✅ Synchronized video + audio saved to: {output_path}")
#     except Exception as e:
#         print(f"⚠️ Could not mux audio with video via ffmpeg ({e}). Video and audio remain separate.")


# # ─────────────────────────────────────────────────────────────────────────────
# # Black Glasses Detection Helper Function
# # ─────────────────────────────────────────────────────────────────────────────

# # Thresholds for strict dark glasses / sunglasses detection
# DARK_PIXEL_THRESHOLD = 80
# DARK_PIXEL_RATIO = 0.35
# CONTRAST_RATIO = 0.65
# MIN_EYE_BRIGHTNESS = 95


# def detect_black_glasses(landmarks, frame, frame_width, frame_height):
#     """
#     Analyzes both eye regions using pixel dark-ratio and skin contrast thresholds
#     to strictly detect dark glasses/sunglasses while ignoring normal eyebrows & shadows.
#     """
#     # MediaPipe FaceMesh Indices surrounding the outer eye/lens areas
#     LEFT_EYE_INDICES = [33, 133, 159, 145, 158, 153, 144, 160]
#     RIGHT_EYE_INDICES = [362, 263, 386, 374, 385, 380, 373, 387]
#     CHEEK_INDICES = [205, 425]

#     def get_coords(idx_list):
#         pts = []
#         for idx in idx_list:
#             pts.append([int(landmarks[idx].x * frame_width), int(landmarks[idx].y * frame_height)])
#         return np.array(pts, dtype=np.int32)

#     l_pts = get_coords(LEFT_EYE_INDICES)
#     r_pts = get_coords(RIGHT_EYE_INDICES)
#     cheek_pts = get_coords(CHEEK_INDICES)

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     def analyze_eye_region(pts, expand=10):
#         x, y, w, h = cv2.boundingRect(pts)
#         x1, y1 = max(0, x - expand), max(0, y - expand)
#         x2, y2 = min(frame_width, x + w + expand), min(frame_height, y + h + expand)
        
#         region = gray[y1:y2, x1:x2]
#         if region.size == 0:
#             return 255.0, 0.0

#         # Calculate average brightness
#         mean_brightness = np.mean(region)
        
#         # Calculate ratio of dark pixels inside the region
#         dark_pixels = np.sum(region < DARK_PIXEL_THRESHOLD)
#         dark_ratio = float(dark_pixels) / float(region.size)

#         return mean_brightness, dark_ratio

#     l_bright, l_dark_ratio = analyze_eye_region(l_pts)
#     r_bright, r_dark_ratio = analyze_eye_region(r_pts)

#     avg_eye_brightness = (l_bright + r_bright) / 2.0
#     avg_dark_pixel_ratio = (l_dark_ratio + r_dark_ratio) / 2.0

#     # Extract skin baseline brightness from cheek landmarks
#     cheek_val_1 = gray[min(frame_height - 1, cheek_pts[0][1]), min(frame_width - 1, cheek_pts[0][0])]
#     cheek_val_2 = gray[min(frame_height - 1, cheek_pts[1][1]), min(frame_width - 1, cheek_pts[1][0])]
#     skin_baseline = max((float(cheek_val_1) + float(cheek_val_2)) / 2.0, 1.0)

#     calculated_contrast_ratio = avg_eye_brightness / skin_baseline

#     # Strict multi-condition check:
#     # 1. High proportion of dark pixels in eye regions (>= DARK_PIXEL_RATIO)
#     # 2. Overall eye region mean brightness is low (< MIN_EYE_BRIGHTNESS)
#     # 3. Eye area is darker than surrounding skin by CONTRAST_RATIO threshold
#     is_glasses = (
#         (avg_dark_pixel_ratio >= DARK_PIXEL_RATIO) and
#         (avg_eye_brightness < MIN_EYE_BRIGHTNESS) and
#         (calculated_contrast_ratio < CONTRAST_RATIO)
#     )

#     return is_glasses


# # ─────────────────────────────────────────────────────────────────────────────
# # Iris Detection Helper Function
# # ─────────────────────────────────────────────────────────────────────────────

# def detect_iris_direction(landmarks, frame_width, frame_height):
#     """
#     Detects iris position relative to eye corners and eyelids.
#     Returns: (direction_label, right_iris_point, left_iris_point)
#     """
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

#     # Left Eye
#     l_iris = get_pt(LEFT_IRIS_CENTER)
#     l_inner, l_outer = get_pt(LEFT_EYE_INNER), get_pt(LEFT_EYE_OUTER)
#     l_top, l_bottom   = get_pt(LEFT_EYE_TOP), get_pt(LEFT_EYE_BOTTOM)

#     l_horiz_dist = np.linalg.norm(l_outer - l_inner)
#     l_vert_dist  = np.linalg.norm(l_bottom - l_top)

#     l_h_ratio = (np.linalg.norm(l_iris - l_inner) / l_horiz_dist) if l_horiz_dist > 0 else 0.5
#     l_v_ratio = (np.linalg.norm(l_iris - l_top) / l_vert_dist) if l_vert_dist > 0 else 0.5

#     # Right Eye
#     r_iris = get_pt(RIGHT_IRIS_CENTER)
#     r_inner, r_outer = get_pt(RIGHT_EYE_INNER), get_pt(RIGHT_EYE_OUTER)
#     r_top, r_bottom   = get_pt(RIGHT_EYE_TOP), get_pt(RIGHT_EYE_BOTTOM)

#     r_horiz_dist = np.linalg.norm(r_outer - r_inner)
#     r_vert_dist  = np.linalg.norm(r_bottom - r_top)

#     r_h_ratio = (np.linalg.norm(r_iris - r_inner) / r_horiz_dist) if r_horiz_dist > 0 else 0.5
#     r_v_ratio = (np.linalg.norm(r_iris - r_top) / r_vert_dist) if r_vert_dist > 0 else 0.5

#     # Average
#     avg_h_ratio = (l_h_ratio + r_h_ratio) / 2.0
#     avg_v_ratio = (l_v_ratio + r_v_ratio) / 2.0

#     direction = None
#     if avg_h_ratio < 0.35:
#         direction = "IRIS LEFT"
#     elif avg_h_ratio > 0.65:
#         direction = "IRIS RIGHT"
#     elif avg_v_ratio < 0.30:
#         direction = "IRIS UP"
#     elif avg_v_ratio > 0.75:
#         direction = "IRIS DOWN"

#     right_iris_pixel = (int(r_iris[0]), int(r_iris[1]))
#     left_iris_pixel  = (int(l_iris[0]), int(l_iris[1]))

#     return direction, right_iris_pixel, left_iris_pixel


# # ─────────────────────────────────────────────────────────────────────────────
# # Candidate Selection
# # ─────────────────────────────────────────────────────────────────────────────

# IMAGES_DIR = Path("images")
# SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# def list_candidates(images_dir: Path) -> list[Path]:
#     """Return all image files found in the images/ folder, sorted by name."""
#     if not images_dir.exists():
#         return []
#     return sorted(
#         p for p in images_dir.iterdir()
#         if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
#     )

# def select_candidate() -> Path | None:
#     candidates = list_candidates(IMAGES_DIR)

#     if not candidates:
#         print(f"\n❌  No images found in '{IMAGES_DIR}/'.")
#         print(f"    Add candidate photos there (e.g. images/john_doe.jpg) and try again.")
#         return None

#     print("\n" + "═" * 50)
#     print("  📋  CANDIDATE SELECTION")
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
#                 print(f"\n✅  Selected: {name}  →  {selected}")
#                 return selected
#         print(f"   ⚠️  Enter a number between 1 and {len(candidates)}, or 'q'.")


# # ─────────────────────────────────────────────────────────────────────────────
# # Reference Encoding
# # ─────────────────────────────────────────────────────────────────────────────

# def load_reference_encoding(reference_image_path: Path):
#     if not reference_image_path.exists():
#         print(f"Error: Reference image not found at '{reference_image_path}'.")
#         return None

#     reference_image = face_recognition.load_image_file(str(reference_image_path))
#     face_locations = face_recognition.face_locations(reference_image)

#     if len(face_locations) == 0:
#         print("Error: No face detected in the reference image. Use a clear, front-facing photo.")
#         return None
#     if len(face_locations) > 1:
#         print("Error: Reference image has more than one face. Use a photo with only the candidate.")
#         return None

#     return face_recognition.face_encodings(reference_image, known_face_locations=face_locations)[0]


# # ─────────────────────────────────────────────────────────────────────────────
# # Main Interview Session
# # ─────────────────────────────────────────────────────────────────────────────

# def start_interview_recording(reference_image_path: Path, match_tolerance=0.6, verify_every_seconds=2.0):
#     candidate_name = reference_image_path.stem.replace("_", " ").replace("-", " ").title()

#     print(f"\n🔍  Encoding reference photo for: {candidate_name} ...")
#     reference_encoding = load_reference_encoding(reference_image_path)
#     if reference_encoding is None:
#         print("Aborting: fix the reference image and try again.")
#         return
#     print(f"✅  Reference photo loaded successfully.")

#     # 1. Setup Directories
#     video_dir = Path("video_interview")
#     snapshot_dir = video_dir / "snapshots"
#     video_dir.mkdir(parents=True, exist_ok=True)
#     snapshot_dir.mkdir(parents=True, exist_ok=True)

#     # 2. Initialize Camera & Video Writer
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return

#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     safe_name  = reference_image_path.stem
#     timestamp  = time.strftime('%Y%m%d_%H%M%S')
#     raw_video_path = video_dir / f"interview_{safe_name}_{timestamp}_raw.mp4"
#     audio_path     = video_dir / f"interview_{safe_name}_{timestamp}.wav"
#     final_video_path = video_dir / f"interview_{safe_name}_{timestamp}.mp4"

#     target_fps = 20.0
#     writer = cv2.VideoWriter(str(raw_video_path), cv2.VideoWriter_fourcc(*'mp4v'), target_fps, (width, height))

#     # Initialize Audio Recorder Thread
#     audio_recorder = AudioRecorderThread(output_wav_path=audio_path)
#     audio_recorder.start()

#     # 3. Setup MediaPipe Solutions
#     mp_face = mp.solutions.face_detection
#     mp_face_mesh = mp.solutions.face_mesh

#     # State tracking variables
#     last_snap_time = 0
#     last_verify_time = 0
#     last_movement_snap_time = 0
#     last_iris_snap_time = 0
#     last_glasses_snap_time = 0

#     # Persistence count for glasses detection (3 consecutive frames required)
#     glasses_consecutive_frames = 0
#     REQUIRED_GLASSES_FRAMES = 3

#     identity_ok = None
#     font = cv2.FONT_HERSHEY_SIMPLEX

#     frame_count = 0
#     start_time = time.time()

#     print(f"\n🎥  Recording to: {final_video_path}")
#     print(f"    Candidate : {candidate_name}")
#     print(f"    Tolerance : {match_tolerance}  |  Re-verify every: {verify_every_seconds}s")
#     print("    Press 'q' in the video window to stop.\n")

#     try:
#         with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_det, \
#              mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:

#             while cap.isOpened():
#                 success, frame = cap.read()
#                 if not success:
#                     break

#                 frame_count += 1

#                 # Detect faces
#                 rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results    = face_det.process(rgb_frame)
#                 face_count = len(results.detections) if results.detections else 0

#                 # Draw bounding boxes
#                 if face_count > 0:
#                     for det in results.detections:
#                         box = det.location_data.relative_bounding_box
#                         x, y = int(box.xmin * width), int(box.ymin * height)
#                         w, h = int(box.width  * width), int(box.height * height)
#                         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

#                 # ── Business logic ───────────────────────────────────────────

#                 if face_count > 1:
#                     label = f"WARNING: {face_count} PEOPLE!"
#                     cv2.putText(frame, label, (20, 50), font, 1, (0, 0, 255), 3)
#                     cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)

#                     if time.time() - last_snap_time > 3.0:
#                         snap_path = snapshot_dir / f"violation_{int(time.time())}.jpg"
#                         cv2.imwrite(str(snap_path), frame)
#                         print(f"🚨 Multiple people detected! Snapshot: {snap_path}")
#                         last_snap_time = time.time()

#                 elif face_count == 0:
#                     cv2.putText(frame, "WARNING: NO CANDIDATE", (20, 50), font, 1, (0, 165, 255), 3)
#                     identity_ok = None
#                     glasses_consecutive_frames = 0

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
#                     if yaw_ratio < -0.15:
#                         movement_state = "LOOKING RIGHT"
#                     elif yaw_ratio > 0.15:
#                         movement_state = "LOOKING LEFT"
#                     elif pitch_ratio < 0.45:
#                         movement_state = "LOOKING UP"
#                     elif pitch_ratio > 0.60:
#                         movement_state = "LOOKING DOWN"

#                     if movement_state:
#                         cv2.putText(frame, f"WARNING: {movement_state}", (20, 90), font, 1, (0, 165, 255), 3)

#                         if time.time() - last_movement_snap_time > 3.0:
#                             snap_path = snapshot_dir / f"movement_violation_{int(time.time())}.jpg"
#                             cv2.imwrite(str(snap_path), frame)
#                             print(f"🚨 Face Movement Detected ({movement_state})! Snapshot: {snap_path}")
#                             last_movement_snap_time = time.time()

#                     # ── Black Glasses & Iris Tracking ─────────────────────────
#                     mesh_results = face_mesh.process(rgb_frame)
#                     if mesh_results.multi_face_landmarks:
#                         face_landmarks = mesh_results.multi_face_landmarks[0]

#                         # Black Glasses Check using improved threshold logic
#                         is_wearing_glasses = detect_black_glasses(face_landmarks.landmark, frame, width, height)
#                         if is_wearing_glasses:
#                             glasses_consecutive_frames += 1
#                         else:
#                             glasses_consecutive_frames = max(0, glasses_consecutive_frames - 1)

#                         if glasses_consecutive_frames >= REQUIRED_GLASSES_FRAMES:
#                             cv2.putText(frame, "WARN: EYE NOT VISIBLE", (20, 170), font, 1, (0, 0, 255), 3)

#                             if time.time() - last_glasses_snap_time > 3.0:
#                                 snap_path = snapshot_dir / f"glasses_violation_{int(time.time())}.jpg"
#                                 cv2.imwrite(str(snap_path), frame)
#                                 print(f"🚨 Black Glasses Detected (Eyes Hidden)! Snapshot: {snap_path}")
#                                 last_glasses_snap_time = time.time()

#                         # Iris Tracking
#                         iris_dir, r_iris_pt, l_iris_pt = detect_iris_direction(face_landmarks.landmark, width, height)

#                         cv2.circle(frame, r_iris_pt, 2, (0, 255, 0), -1)
#                         cv2.circle(frame, l_iris_pt, 2, (0, 255, 0), -1)

#                         if iris_dir:
#                             cv2.putText(frame, f"WARNING: {iris_dir}", (20, 130), font, 1, (0, 165, 255), 3)

#                             if time.time() - last_iris_snap_time > 3.0:
#                                 snap_path = snapshot_dir / f"iris_violation_{int(time.time())}.jpg"
#                                 cv2.imwrite(str(snap_path), frame)
#                                 print(f"🚨 Iris Movement Detected ({iris_dir})! Snapshot: {snap_path}")
#                                 last_iris_snap_time = time.time()

#                     # ── Identity Verification ────────────────────────────────
#                     if time.time() - last_verify_time > verify_every_seconds:
#                         last_verify_time = time.time()

#                         box = det.location_data.relative_bounding_box
#                         x, y = int(box.xmin * width), int(box.ymin * height)
#                         w, h = int(box.width  * width), int(box.height * height)
#                         top, left   = max(0, y),          max(0, x)
#                         bottom, right = min(height, y+h), min(width, x+w)

#                         try:
#                             encodings = face_recognition.face_encodings(
#                                 rgb_frame,
#                                 known_face_locations=[(top, right, bottom, left)]
#                             )
#                             if encodings:
#                                 distance   = face_recognition.face_distance([reference_encoding], encodings[0])[0]
#                                 identity_ok = bool(distance <= match_tolerance)
#                                 if not identity_ok:
#                                     print(
#                                         f"🚨 IDENTITY MISMATCH — person in frame does not match "
#                                         f"'{candidate_name}' "
#                                         f"(distance={distance:.3f}, tolerance={match_tolerance})"
#                                     )
#                                     snap_path = snapshot_dir / f"identity_mismatch_{int(time.time())}.jpg"
#                                     cv2.imwrite(str(snap_path), frame)
#                                     print(f"   Snapshot saved: {snap_path}")
#                         except Exception as e:
#                             print(f"⚠️  Verification skipped: {e}")

#                     # ── Identity Overlay ──────────────────────────────────────
#                     if identity_ok is True:
#                         cv2.putText(frame, f"Verified: {candidate_name}", (20, 50), font, 0.9, (0, 255, 0), 2)
#                     elif identity_ok is False:
#                         cv2.putText(frame, "WARNING: UNRECOGNIZED PERSON", (20, 50), font, 1, (0, 0, 255), 3)
#                         cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)
#                     else:
#                         cv2.putText(frame, "Verifying...", (20, 50), font, 1, (0, 165, 255), 2)

#                 # Record & display
#                 writer.write(frame)
#                 cv2.imshow('Interview Proctoring System', frame)

#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

#     except KeyboardInterrupt:
#         print("\nStopped via terminal (Ctrl+C).")
#     finally:
#         elapsed_time = time.time() - start_time
#         actual_fps = (frame_count / elapsed_time) if elapsed_time > 0 else target_fps

#         cap.release()
#         writer.release()
#         cv2.destroyAllWindows()

#         # Stop audio recorder thread
#         audio_recorder.stop()
#         audio_recorder.join()

#         print("\n🎬 Processing final recorded interview video and audio...")
#         combine_audio_video(raw_video_path, audio_path, final_video_path, actual_fps=actual_fps)
#         print(f"✅  Interview finished. Video saved to: {final_video_path}")


# # ─────────────────────────────────────────────────────────────────────────────
# # Entry Point
# # ─────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     chosen = select_candidate()
#     if chosen is not None:
#         start_interview_recording(reference_image_path=chosen)



# import cv2
# import mediapipe as mp
# import face_recognition
# import time
# import os
# import numpy as np
# from pathlib import Path

# # ─────────────────────────────────────────────────────────────────────────────
# # Iris Detection Helper Function
# # ─────────────────────────────────────────────────────────────────────────────

# def detect_iris_direction(landmarks, frame_width, frame_height):
#     """
#     Detects iris position relative to eye corners and eyelids.
#     Returns: (direction_label, right_iris_point, left_iris_point)
#     """
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

#     # Left Eye
#     l_iris = get_pt(LEFT_IRIS_CENTER)
#     l_inner, l_outer = get_pt(LEFT_EYE_INNER), get_pt(LEFT_EYE_OUTER)
#     l_top, l_bottom   = get_pt(LEFT_EYE_TOP), get_pt(LEFT_EYE_BOTTOM)

#     l_horiz_dist = np.linalg.norm(l_outer - l_inner)
#     l_vert_dist  = np.linalg.norm(l_bottom - l_top)

#     l_h_ratio = (np.linalg.norm(l_iris - l_inner) / l_horiz_dist) if l_horiz_dist > 0 else 0.5
#     l_v_ratio = (np.linalg.norm(l_iris - l_top) / l_vert_dist) if l_vert_dist > 0 else 0.5

#     # Right Eye
#     r_iris = get_pt(RIGHT_IRIS_CENTER)
#     r_inner, r_outer = get_pt(RIGHT_EYE_INNER), get_pt(RIGHT_EYE_OUTER)
#     r_top, r_bottom   = get_pt(RIGHT_EYE_TOP), get_pt(RIGHT_EYE_BOTTOM)

#     r_horiz_dist = np.linalg.norm(r_outer - r_inner)
#     r_vert_dist  = np.linalg.norm(r_bottom - r_top)

#     r_h_ratio = (np.linalg.norm(r_iris - r_inner) / r_horiz_dist) if r_horiz_dist > 0 else 0.5
#     r_v_ratio = (np.linalg.norm(r_iris - r_top) / r_vert_dist) if r_vert_dist > 0 else 0.5

#     # Average
#     avg_h_ratio = (l_h_ratio + r_h_ratio) / 2.0
#     avg_v_ratio = (l_v_ratio + r_v_ratio) / 2.0

#     direction = None
#     if avg_h_ratio < 0.35:
#         direction = "IRIS LEFT"
#     elif avg_h_ratio > 0.65:
#         direction = "IRIS RIGHT"
#     elif avg_v_ratio < 0.30:
#         direction = "IRIS UP"
#     elif avg_v_ratio > 0.75:
#         direction = "IRIS DOWN"

#     right_iris_pixel = (int(r_iris[0]), int(r_iris[1]))
#     left_iris_pixel  = (int(l_iris[0]), int(l_iris[1]))

#     return direction, right_iris_pixel, left_iris_pixel


# # ─────────────────────────────────────────────────────────────────────────────
# # Black Glasses Detection Helper Function
# # ─────────────────────────────────────────────────────────────────────────────

# def detect_black_glasses(frame, landmarks, frame_width, frame_height):
#     """
#     Detects if the user is wearing dark/black sunglasses by comparing 
#     eye region brightness to forehead skin tone.
#     """
#     def get_crop(indices):
#         xs = [int(landmarks[i].x * frame_width) for i in indices]
#         ys = [int(landmarks[i].y * frame_height) for i in indices]
#         x1, x2 = max(0, min(xs) - 5), min(frame_width, max(xs) + 5)
#         y1, y2 = max(0, min(ys) - 5), min(frame_height, max(ys) + 5)
#         return frame[y1:y2, x1:x2]

#     # Crop eye regions and forehead skin area
#     left_eye_crop = get_crop([33, 133, 159, 145])
#     right_eye_crop = get_crop([362, 263, 386, 374])
#     skin_crop = get_crop([10, 151, 9, 8])  # Forehead region

#     if left_eye_crop.size == 0 or right_eye_crop.size == 0 or skin_crop.size == 0:
#         return False

#     # Average brightness in grayscale
#     left_val = np.mean(cv2.cvtColor(left_eye_crop, cv2.COLOR_BGR2GRAY))
#     right_val = np.mean(cv2.cvtColor(right_eye_crop, cv2.COLOR_BGR2GRAY))
#     skin_val = np.mean(cv2.cvtColor(skin_crop, cv2.COLOR_BGR2GRAY))

#     eye_val = (left_val + right_val) / 2.0

#     # Threshold: eyes extremely dark OR eye brightness is significantly lower than forehead skin
#     if eye_val < 45 or (skin_val > 55 and (eye_val / skin_val) < 0.48):
#         return True

#     return False


# # ─────────────────────────────────────────────────────────────────────────────
# # Candidate selection
# # ─────────────────────────────────────────────────────────────────────────────

# IMAGES_DIR = Path("images")
# SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# def list_candidates(images_dir: Path) -> list[Path]:
#     """Return all image files found in the images/ folder, sorted by name."""
#     if not images_dir.exists():
#         return []
#     return sorted(
#         p for p in images_dir.iterdir()
#         if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
#     )

# def select_candidate() -> Path | None:
#     """
#     Scans the images/ folder, lists every candidate photo by number, and lets
#     the user pick one by typing a number (or 'q' to quit).
#     Returns the chosen Path, or None if the user quits / folder is empty.
#     """
#     candidates = list_candidates(IMAGES_DIR)

#     if not candidates:
#         print(f"\n❌  No images found in '{IMAGES_DIR}/'.")
#         print(f"    Add candidate photos there (e.g. images/john_doe.jpg) and try again.")
#         return None

#     print("\n" + "═" * 50)
#     print("   📋  CANDIDATE SELECTION")
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
#                 print(f"\n✅  Selected: {name}  →  {selected}")
#                 return selected
#         print(f"   ⚠️  Enter a number between 1 and {len(candidates)}, or 'q'.")


# # ─────────────────────────────────────────────────────────────────────────────
# # Reference encoding
# # ─────────────────────────────────────────────────────────────────────────────

# def load_reference_encoding(reference_image_path: Path):
#     """
#     Loads the candidate's sample photo and returns its 128-d face encoding.
#     Returns None on any error (missing file, no face, multiple faces).
#     """
#     if not reference_image_path.exists():
#         print(f"Error: Reference image not found at '{reference_image_path}'.")
#         return None

#     reference_image = face_recognition.load_image_file(str(reference_image_path))
#     face_locations = face_recognition.face_locations(reference_image)

#     if len(face_locations) == 0:
#         print("Error: No face detected in the reference image. Use a clear, front-facing photo.")
#         return None
#     if len(face_locations) > 1:
#         print("Error: Reference image has more than one face. Use a photo with only the candidate.")
#         return None

#     return face_recognition.face_encodings(reference_image, known_face_locations=face_locations)[0]


# # ─────────────────────────────────────────────────────────────────────────────
# # Main interview session
# # ─────────────────────────────────────────────────────────────────────────────

# def start_interview_recording(reference_image_path: Path, match_tolerance=0.6, verify_every_seconds=2.0):
#     candidate_name = reference_image_path.stem.replace("_", " ").replace("-", " ").title()

#     print(f"\n🔍  Encoding reference photo for: {candidate_name} ...")
#     reference_encoding = load_reference_encoding(reference_image_path)
#     if reference_encoding is None:
#         print("Aborting: fix the reference image and try again.")
#         return
#     print(f"✅  Reference photo loaded successfully.")

#     # 1. Setup Directories
#     video_dir = Path("video_interview")
#     snapshot_dir = video_dir / "snapshots"
#     video_dir.mkdir(parents=True, exist_ok=True)
#     snapshot_dir.mkdir(parents=True, exist_ok=True)

#     # 2. Initialize Camera & Video Writer
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return

#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     safe_name  = reference_image_path.stem
#     video_path = video_dir / f"interview_{safe_name}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
#     writer     = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

#     # 3. Setup MediaPipe Solutions
#     mp_face = mp.solutions.face_detection
#     mp_face_mesh = mp.solutions.face_mesh

#     # State tracking variables for snapshots
#     last_snap_time = 0
#     last_verify_time = 0
#     last_movement_snap_time = 0  # Head movement snapshots
#     last_iris_snap_time = 0      # Iris movement snapshots
#     last_glasses_snap_time = 0   # Black glasses snapshots

#     identity_ok = None
#     font = cv2.FONT_HERSHEY_SIMPLEX

#     print(f"\n🎥  Recording to: {video_path}")
#     print(f"    Candidate : {candidate_name}")
#     print(f"    Tolerance : {match_tolerance}  |  Re-verify every: {verify_every_seconds}s")
#     print("    Press 'q' in the video window to stop.\n")

#     try:
#         with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_det, \
#              mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:

#             while cap.isOpened():
#                 success, frame = cap.read()
#                 if not success:
#                     break

#                 # Detect faces
#                 rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results    = face_det.process(rgb_frame)
#                 face_count = len(results.detections) if results.detections else 0

#                 # Draw bounding boxes
#                 if face_count > 0:
#                     for det in results.detections:
#                         box = det.location_data.relative_bounding_box
#                         x, y = int(box.xmin * width), int(box.ymin * height)
#                         w, h = int(box.width  * width), int(box.height * height)
#                         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

#                 # ── Business logic ───────────────────────────────────────────

#                 if face_count > 1:
#                     label = f"WARNING: {face_count} PEOPLE!"
#                     cv2.putText(frame, label, (20, 50), font, 1, (0, 0, 255), 3)
#                     cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)

#                     if time.time() - last_snap_time > 3.0:
#                         snap_path = snapshot_dir / f"violation_{int(time.time())}.jpg"
#                         cv2.imwrite(str(snap_path), frame)
#                         print(f"🚨 Multiple people detected! Snapshot: {snap_path}")
#                         last_snap_time = time.time()

#                 elif face_count == 0:
#                     cv2.putText(frame, "WARNING: NO CANDIDATE", (20, 50), font, 1, (0, 165, 255), 3)
#                     identity_ok = None

#                 else:
#                     det = results.detections[0]

#                     # ── Head Movement Detection Logic ────────────────────────
#                     keypoints = det.location_data.relative_keypoints

#                     # MediaPipe keypoint indices: 0:Right Eye, 1:Left Eye, 2:Nose, 3:Mouth
#                     rx, ry = int(keypoints[0].x * width), int(keypoints[0].y * height)
#                     lx, ly = int(keypoints[1].x * width), int(keypoints[1].y * height)
#                     nx, ny = int(keypoints[2].x * width), int(keypoints[2].y * height)
#                     mx, my = int(keypoints[3].x * width), int(keypoints[3].y * height)

#                     # Calculate Yaw (Left/Right)
#                     eye_center_x = (rx + lx) / 2.0
#                     eye_dist = abs(lx - rx)
#                     yaw_ratio = (nx - eye_center_x) / eye_dist if eye_dist > 0 else 0

#                     # Calculate Pitch (Up/Down)
#                     eye_center_y = (ry + ly) / 2.0
#                     face_length = my - eye_center_y
#                     pitch_ratio = (ny - eye_center_y) / face_length if face_length > 0 else 0.5

#                     movement_state = None
#                     if yaw_ratio < -0.15:
#                         movement_state = "LOOKING RIGHT"
#                     elif yaw_ratio > 0.15:
#                         movement_state = "LOOKING LEFT"
#                     elif pitch_ratio < 0.45:
#                         movement_state = "LOOKING UP"
#                     elif pitch_ratio > 0.60:
#                         movement_state = "LOOKING DOWN"

#                     if movement_state:
#                         cv2.putText(frame, f"WARNING: {movement_state}", (20, 90), font, 1, (0, 165, 255), 3)

#                         # Take snapshot with a 3-second cooldown
#                         if time.time() - last_movement_snap_time > 3.0:
#                             snap_path = snapshot_dir / f"movement_violation_{int(time.time())}.jpg"
#                             cv2.imwrite(str(snap_path), frame)
#                             print(f"🚨 Face Movement Detected ({movement_state})! Snapshot: {snap_path}")
#                             last_movement_snap_time = time.time()

#                     # ── Face Mesh Analytics (Iris & Black Glasses) ──────────
#                     mesh_results = face_mesh.process(rgb_frame)
#                     if mesh_results.multi_face_landmarks:
#                         face_landmarks = mesh_results.multi_face_landmarks[0]

#                         # 1. Iris Tracking
#                         iris_dir, r_iris_pt, l_iris_pt = detect_iris_direction(face_landmarks.landmark, width, height)
#                         cv2.circle(frame, r_iris_pt, 2, (0, 255, 0), -1)
#                         cv2.circle(frame, l_iris_pt, 2, (0, 255, 0), -1)

#                         if iris_dir:
#                             cv2.putText(frame, f"WARNING: {iris_dir}", (20, 130), font, 1, (0, 165, 255), 3)

#                             if time.time() - last_iris_snap_time > 3.0:
#                                 snap_path = snapshot_dir / f"iris_violation_{int(time.time())}.jpg"
#                                 cv2.imwrite(str(snap_path), frame)
#                                 print(f"🚨 Iris Movement Detected ({iris_dir})! Snapshot: {snap_path}")
#                                 last_iris_snap_time = time.time()

#                         # 2. Black Glasses Detection
#                         has_black_glasses = detect_black_glasses(frame, face_landmarks.landmark, width, height)
#                         if has_black_glasses:
#                             cv2.putText(frame, "WARNING: BLACK GLASSES DETECTED", (20, 170), font, 1, (0, 0, 255), 3)

#                             if time.time() - last_glasses_snap_time > 3.0:
#                                 snap_path = snapshot_dir / f"glasses_violation_{int(time.time())}.jpg"
#                                 cv2.imwrite(str(snap_path), frame)
#                                 print(f"🚨 Black Glasses Detected! Snapshot: {snap_path}")
#                                 last_glasses_snap_time = time.time()

#                     # ── Identity verification (throttled) ────────────────────
#                     if time.time() - last_verify_time > verify_every_seconds:
#                         last_verify_time = time.time()

#                         box = det.location_data.relative_bounding_box
#                         x, y = int(box.xmin * width), int(box.ymin * height)
#                         w, h = int(box.width  * width), int(box.height * height)
#                         top, left   = max(0, y),          max(0, x)
#                         bottom, right = min(height, y+h), min(width, x+w)

#                         try:
#                             encodings = face_recognition.face_encodings(
#                                 rgb_frame,
#                                 known_face_locations=[(top, right, bottom, left)]
#                             )
#                             if encodings:
#                                 distance   = face_recognition.face_distance([reference_encoding], encodings[0])[0]
#                                 identity_ok = bool(distance <= match_tolerance)
#                                 if not identity_ok:
#                                     print(
#                                         f"🚨 IDENTITY MISMATCH — person in frame does not match "
#                                         f"'{candidate_name}' "
#                                         f"(distance={distance:.3f}, tolerance={match_tolerance})"
#                                     )
#                                     snap_path = snapshot_dir / f"identity_mismatch_{int(time.time())}.jpg"
#                                     cv2.imwrite(str(snap_path), frame)
#                                     print(f"   Snapshot saved: {snap_path}")
#                         except Exception as e:
#                             print(f"⚠️  Verification skipped: {e}")

#                     # ── Overlay based on last check result ───────────────────
#                     if identity_ok is True:
#                         cv2.putText(frame, f"Verified: {candidate_name}", (20, 50), font, 0.9, (0, 255, 0), 2)
#                     elif identity_ok is False:
#                         cv2.putText(frame, "WARNING: UNRECOGNIZED PERSON", (20, 50), font, 1, (0, 0, 255), 3)
#                         cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)
#                     else:
#                         cv2.putText(frame, "Verifying...", (20, 50), font, 1, (0, 165, 255), 2)

#                 # Record & display
#                 writer.write(frame)
#                 cv2.imshow('Interview Proctoring System', frame)

#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

#     except KeyboardInterrupt:
#         print("\nStopped via terminal (Ctrl+C).")
#     finally:
#         cap.release()
#         writer.release()
#         cv2.destroyAllWindows()
#         print(f"\n✅  Interview finished. Video saved to: {video_path}")


# # ─────────────────────────────────────────────────────────────────────────────
# # Entry point
# # ─────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     chosen = select_candidate()
#     if chosen is not None:
#         start_interview_recording(reference_image_path=chosen)



# day one night code 


# import cv2
# import mediapipe as mp
# import face_recognition
# import time
# import os
# import numpy as np
# from pathlib import Path

# # ─────────────────────────────────────────────────────────────────────────────
# # Iris Detection Helper Function
# # ─────────────────────────────────────────────────────────────────────────────

# def detect_iris_direction(landmarks, frame_width, frame_height):
#     """
#     Detects iris position relative to eye corners and eyelids.
#     Returns: (direction_label, right_iris_point, left_iris_point)
#     """
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

#     # Left Eye
#     l_iris = get_pt(LEFT_IRIS_CENTER)
#     l_inner, l_outer = get_pt(LEFT_EYE_INNER), get_pt(LEFT_EYE_OUTER)
#     l_top, l_bottom   = get_pt(LEFT_EYE_TOP), get_pt(LEFT_EYE_BOTTOM)

#     l_horiz_dist = np.linalg.norm(l_outer - l_inner)
#     l_vert_dist  = np.linalg.norm(l_bottom - l_top)

#     l_h_ratio = (np.linalg.norm(l_iris - l_inner) / l_horiz_dist) if l_horiz_dist > 0 else 0.5
#     l_v_ratio = (np.linalg.norm(l_iris - l_top) / l_vert_dist) if l_vert_dist > 0 else 0.5

#     # Right Eye
#     r_iris = get_pt(RIGHT_IRIS_CENTER)
#     r_inner, r_outer = get_pt(RIGHT_EYE_INNER), get_pt(RIGHT_EYE_OUTER)
#     r_top, r_bottom   = get_pt(RIGHT_EYE_TOP), get_pt(RIGHT_EYE_BOTTOM)

#     r_horiz_dist = np.linalg.norm(r_outer - r_inner)
#     r_vert_dist  = np.linalg.norm(r_bottom - r_top)

#     r_h_ratio = (np.linalg.norm(r_iris - r_inner) / r_horiz_dist) if r_horiz_dist > 0 else 0.5
#     r_v_ratio = (np.linalg.norm(r_iris - r_top) / r_vert_dist) if r_vert_dist > 0 else 0.5

#     # Average
#     avg_h_ratio = (l_h_ratio + r_h_ratio) / 2.0
#     avg_v_ratio = (l_v_ratio + r_v_ratio) / 2.0

#     direction = None
#     if avg_h_ratio < 0.35:
#         direction = "IRIS LEFT"
#     elif avg_h_ratio > 0.65:
#         direction = "IRIS RIGHT"
#     elif avg_v_ratio < 0.30:
#         direction = "IRIS UP"
#     elif avg_v_ratio > 0.75:
#         direction = "IRIS DOWN"

#     right_iris_pixel = (int(r_iris[0]), int(r_iris[1]))
#     left_iris_pixel  = (int(l_iris[0]), int(l_iris[1]))

#     return direction, right_iris_pixel, left_iris_pixel


# # ─────────────────────────────────────────────────────────────────────────────
# # Candidate selection
# # ─────────────────────────────────────────────────────────────────────────────

# IMAGES_DIR = Path("images")
# SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# def list_candidates(images_dir: Path) -> list[Path]:
#     """Return all image files found in the images/ folder, sorted by name."""
#     if not images_dir.exists():
#         return []
#     return sorted(
#         p for p in images_dir.iterdir()
#         if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
#     )

# def select_candidate() -> Path | None:
#     """
#     Scans the images/ folder, lists every candidate photo by number, and lets
#     the user pick one by typing a number (or 'q' to quit).
#     Returns the chosen Path, or None if the user quits / folder is empty.
#     """
#     candidates = list_candidates(IMAGES_DIR)

#     if not candidates:
#         print(f"\n❌  No images found in '{IMAGES_DIR}/'.")
#         print(f"    Add candidate photos there (e.g. images/john_doe.jpg) and try again.")
#         return None

#     print("\n" + "═" * 50)
#     print("   📋  CANDIDATE SELECTION")
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
#                 print(f"\n✅  Selected: {name}  →  {selected}")
#                 return selected
#         print(f"   ⚠️  Enter a number between 1 and {len(candidates)}, or 'q'.")


# # ─────────────────────────────────────────────────────────────────────────────
# # Reference encoding
# # ─────────────────────────────────────────────────────────────────────────────

# def load_reference_encoding(reference_image_path: Path):
#     """
#     Loads the candidate's sample photo and returns its 128-d face encoding.
#     Returns None on any error (missing file, no face, multiple faces).
#     """
#     if not reference_image_path.exists():
#         print(f"Error: Reference image not found at '{reference_image_path}'.")
#         return None

#     reference_image = face_recognition.load_image_file(str(reference_image_path))
#     face_locations = face_recognition.face_locations(reference_image)

#     if len(face_locations) == 0:
#         print("Error: No face detected in the reference image. Use a clear, front-facing photo.")
#         return None
#     if len(face_locations) > 1:
#         print("Error: Reference image has more than one face. Use a photo with only the candidate.")
#         return None

#     return face_recognition.face_encodings(reference_image, known_face_locations=face_locations)[0]


# # ─────────────────────────────────────────────────────────────────────────────
# # Main interview session
# # ─────────────────────────────────────────────────────────────────────────────

# def start_interview_recording(reference_image_path: Path, match_tolerance=0.6, verify_every_seconds=2.0):
#     candidate_name = reference_image_path.stem.replace("_", " ").replace("-", " ").title()

#     print(f"\n🔍  Encoding reference photo for: {candidate_name} ...")
#     reference_encoding = load_reference_encoding(reference_image_path)
#     if reference_encoding is None:
#         print("Aborting: fix the reference image and try again.")
#         return
#     print(f"✅  Reference photo loaded successfully.")

#     # 1. Setup Directories
#     video_dir = Path("video_interview")
#     snapshot_dir = video_dir / "snapshots"
#     video_dir.mkdir(parents=True, exist_ok=True)
#     snapshot_dir.mkdir(parents=True, exist_ok=True)

#     # 2. Initialize Camera & Video Writer
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return

#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     safe_name  = reference_image_path.stem
#     video_path = video_dir / f"interview_{safe_name}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
#     writer     = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

#     # 3. Setup MediaPipe Solutions
#     mp_face = mp.solutions.face_detection
#     mp_face_mesh = mp.solutions.face_mesh

#     # State tracking variables for snapshots
#     last_snap_time = 0
#     last_verify_time = 0
#     last_movement_snap_time = 0  # Head movement snapshots
#     last_iris_snap_time = 0      # Iris movement snapshots

#     identity_ok = None
#     font = cv2.FONT_HERSHEY_SIMPLEX

#     print(f"\n🎥  Recording to: {video_path}")
#     print(f"    Candidate : {candidate_name}")
#     print(f"    Tolerance : {match_tolerance}  |  Re-verify every: {verify_every_seconds}s")
#     print("    Press 'q' in the video window to stop.\n")

#     try:
#         with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_det, \
#              mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:

#             while cap.isOpened():
#                 success, frame = cap.read()
#                 if not success:
#                     break

#                 # Detect faces
#                 rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results    = face_det.process(rgb_frame)
#                 face_count = len(results.detections) if results.detections else 0

#                 # Draw bounding boxes
#                 if face_count > 0:
#                     for det in results.detections:
#                         box = det.location_data.relative_bounding_box
#                         x, y = int(box.xmin * width), int(box.ymin * height)
#                         w, h = int(box.width  * width), int(box.height * height)
#                         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

#                 # ── Business logic ───────────────────────────────────────────

#                 if face_count > 1:
#                     label = f"WARNING: {face_count} PEOPLE!"
#                     cv2.putText(frame, label, (20, 50), font, 1, (0, 0, 255), 3)
#                     cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)

#                     if time.time() - last_snap_time > 3.0:
#                         snap_path = snapshot_dir / f"violation_{int(time.time())}.jpg"
#                         cv2.imwrite(str(snap_path), frame)
#                         print(f"🚨 Multiple people detected! Snapshot: {snap_path}")
#                         last_snap_time = time.time()

#                 elif face_count == 0:
#                     cv2.putText(frame, "WARNING: NO CANDIDATE", (20, 50), font, 1, (0, 165, 255), 3)
#                     identity_ok = None

#                 else:
#                     det = results.detections[0]

#                     # ── Head Movement Detection Logic ────────────────────────
#                     keypoints = det.location_data.relative_keypoints

#                     # MediaPipe keypoint indices: 0:Right Eye, 1:Left Eye, 2:Nose, 3:Mouth
#                     rx, ry = int(keypoints[0].x * width), int(keypoints[0].y * height)
#                     lx, ly = int(keypoints[1].x * width), int(keypoints[1].y * height)
#                     nx, ny = int(keypoints[2].x * width), int(keypoints[2].y * height)
#                     mx, my = int(keypoints[3].x * width), int(keypoints[3].y * height)

#                     # Calculate Yaw (Left/Right)
#                     eye_center_x = (rx + lx) / 2.0
#                     eye_dist = abs(lx - rx)
#                     yaw_ratio = (nx - eye_center_x) / eye_dist if eye_dist > 0 else 0

#                     # Calculate Pitch (Up/Down)
#                     eye_center_y = (ry + ly) / 2.0
#                     face_length = my - eye_center_y
#                     pitch_ratio = (ny - eye_center_y) / face_length if face_length > 0 else 0.5

#                     movement_state = None
#                     if yaw_ratio < -0.15:
#                         movement_state = "LOOKING RIGHT"
#                     elif yaw_ratio > 0.15:
#                         movement_state = "LOOKING LEFT"
#                     elif pitch_ratio < 0.45:
#                         movement_state = "LOOKING UP"
#                     elif pitch_ratio > 0.60:
#                         movement_state = "LOOKING DOWN"

#                     if movement_state:
#                         cv2.putText(frame, f"WARNING: {movement_state}", (20, 90), font, 1, (0, 165, 255), 3)

#                         # Take snapshot with a 3-second cooldown
#                         if time.time() - last_movement_snap_time > 3.0:
#                             snap_path = snapshot_dir / f"movement_violation_{int(time.time())}.jpg"
#                             cv2.imwrite(str(snap_path), frame)
#                             print(f"🚨 Face Movement Detected ({movement_state})! Snapshot: {snap_path}")
#                             last_movement_snap_time = time.time()

#                     # ── Iris Tracking Logic ──────────────────────────────────
#                     mesh_results = face_mesh.process(rgb_frame)
#                     if mesh_results.multi_face_landmarks:
#                         face_landmarks = mesh_results.multi_face_landmarks[0]
#                         iris_dir, r_iris_pt, l_iris_pt = detect_iris_direction(face_landmarks.landmark, width, height)

#                         # Draw green dots on both irises
#                         cv2.circle(frame, r_iris_pt, 2, (0, 255, 0), -1)
#                         cv2.circle(frame, l_iris_pt, 2, (0, 255, 0), -1)

#                         if iris_dir:
#                             cv2.putText(frame, f"WARNING: {iris_dir}", (20, 130), font, 1, (0, 165, 255), 3)

#                             # Take snapshot with a 3-second cooldown
#                             if time.time() - last_iris_snap_time > 3.0:
#                                 snap_path = snapshot_dir / f"iris_violation_{int(time.time())}.jpg"
#                                 cv2.imwrite(str(snap_path), frame)
#                                 print(f"🚨 Iris Movement Detected ({iris_dir})! Snapshot: {snap_path}")
#                                 last_iris_snap_time = time.time()

#                     # ── Identity verification (throttled) ────────────────────
#                     if time.time() - last_verify_time > verify_every_seconds:
#                         last_verify_time = time.time()

#                         box = det.location_data.relative_bounding_box
#                         x, y = int(box.xmin * width), int(box.ymin * height)
#                         w, h = int(box.width  * width), int(box.height * height)
#                         top, left   = max(0, y),          max(0, x)
#                         bottom, right = min(height, y+h), min(width, x+w)

#                         try:
#                             encodings = face_recognition.face_encodings(
#                                 rgb_frame,
#                                 known_face_locations=[(top, right, bottom, left)]
#                             )
#                             if encodings:
#                                 distance   = face_recognition.face_distance([reference_encoding], encodings[0])[0]
#                                 identity_ok = bool(distance <= match_tolerance)
#                                 if not identity_ok:
#                                     print(
#                                         f"🚨 IDENTITY MISMATCH — person in frame does not match "
#                                         f"'{candidate_name}' "
#                                         f"(distance={distance:.3f}, tolerance={match_tolerance})"
#                                     )
#                                     snap_path = snapshot_dir / f"identity_mismatch_{int(time.time())}.jpg"
#                                     cv2.imwrite(str(snap_path), frame)
#                                     print(f"   Snapshot saved: {snap_path}")
#                         except Exception as e:
#                             print(f"⚠️  Verification skipped: {e}")

#                     # ── Overlay based on last check result ───────────────────
#                     if identity_ok is True:
#                         cv2.putText(frame, f"Verified: {candidate_name}", (20, 50), font, 0.9, (0, 255, 0), 2)
#                     elif identity_ok is False:
#                         cv2.putText(frame, "WARNING: UNRECOGNIZED PERSON", (20, 50), font, 1, (0, 0, 255), 3)
#                         cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)
#                     else:
#                         cv2.putText(frame, "Verifying...", (20, 50), font, 1, (0, 165, 255), 2)

#                 # Record & display
#                 writer.write(frame)
#                 cv2.imshow('Interview Proctoring System', frame)

#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

#     except KeyboardInterrupt:
#         print("\nStopped via terminal (Ctrl+C).")
#     finally:
#         cap.release()
#         writer.release()
#         cv2.destroyAllWindows()
#         print(f"\n✅  Interview finished. Video saved to: {video_path}")


# # ─────────────────────────────────────────────────────────────────────────────
# # Entry point
# # ─────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     chosen = select_candidate()
#     if chosen is not None:
#         start_interview_recording(reference_image_path=chosen)








# import cv2
# import mediapipe as mp
# import face_recognition
# import numpy as np
# import time
# import os
# from pathlib import Path

# # ─────────────────────────────────────────────────────────────────────────────
# # Iris Detector Function
# # ─────────────────────────────────────────────────────────────────────────────

# def detect_iris_direction(landmarks, frame_width, frame_height):
#     """
#     Detects iris position relative to eye corners and eyelids using MediaPipe 468+ landmarks.
#     Returns: (direction_label, right_iris_pixel, left_iris_pixel)
#     """
#     LEFT_IRIS_CENTER  = 468
#     LEFT_EYE_INNER    = 133
#     LEFT_EYE_OUTER    = 33
#     LEFT_EYE_TOP      = 159
#     LEFT_EYE_BOTTOM   = 145

#     RIGHT_IRIS_CENTER = 473
#     RIGHT_EYE_INNER   = 362
#     RIGHT_EYE_OUTER   = 263
#     RIGHT_EYE_TOP     = 386
#     RIGHT_EYE_BOTTOM  = 374

#     def get_pt(idx):
#         return np.array([landmarks[idx].x * frame_width, landmarks[idx].y * frame_height])

#     # Left Eye
#     l_iris = get_pt(LEFT_IRIS_CENTER)
#     l_inner, l_outer = get_pt(LEFT_EYE_INNER), get_pt(LEFT_EYE_OUTER)
#     l_top, l_bottom   = get_pt(LEFT_EYE_TOP), get_pt(LEFT_EYE_BOTTOM)

#     l_horiz_dist = np.linalg.norm(l_outer - l_inner)
#     l_vert_dist  = np.linalg.norm(l_bottom - l_top)

#     l_h_ratio = (np.linalg.norm(l_iris - l_inner) / l_horiz_dist) if l_horiz_dist > 0 else 0.5
#     l_v_ratio = (np.linalg.norm(l_iris - l_top) / l_vert_dist) if l_vert_dist > 0 else 0.5

#     # Right Eye
#     r_iris = get_pt(RIGHT_IRIS_CENTER)
#     r_inner, r_outer = get_pt(RIGHT_EYE_INNER), get_pt(RIGHT_EYE_OUTER)
#     r_top, r_bottom   = get_pt(RIGHT_EYE_TOP), get_pt(RIGHT_EYE_BOTTOM)

#     r_horiz_dist = np.linalg.norm(r_outer - r_inner)
#     r_vert_dist  = np.linalg.norm(r_bottom - r_top)

#     r_h_ratio = (np.linalg.norm(r_iris - r_inner) / r_horiz_dist) if r_horiz_dist > 0 else 0.5
#     r_v_ratio = (np.linalg.norm(r_iris - r_top) / r_vert_dist) if r_vert_dist > 0 else 0.5

#     # Average ratios
#     avg_h_ratio = (l_h_ratio + r_h_ratio) / 2.0
#     avg_v_ratio = (l_v_ratio + r_v_ratio) / 2.0
#     direction = None

#     if avg_h_ratio < 0.42:
#         direction = "IRIS LEFT"

#     elif avg_h_ratio > 0.58:
#         direction = "IRIS RIGHT"

#     elif avg_v_ratio < 0.40:
#         direction = "IRIS UP"

#     elif avg_v_ratio > 0.65:
#         direction = "IRIS DOWN"

#     right_iris_pixel = (int(r_iris[0]), int(r_iris[1]))
#     left_iris_pixel  = (int(l_iris[0]), int(l_iris[1]))

#     return direction, right_iris_pixel, left_iris_pixel


# # ─────────────────────────────────────────────────────────────────────────────
# # Candidate selection
# # ─────────────────────────────────────────────────────────────────────────────

# IMAGES_DIR = Path("images")
# SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# def list_candidates(images_dir: Path) -> list[Path]:
#     """Return all image files found in the images/ folder, sorted by name."""
#     if not images_dir.exists():
#         return []
#     return sorted(
#         p for p in images_dir.iterdir()
#         if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
#     )

# def select_candidate() -> Path | None:
#     """
#     Scans the images/ folder, lists every candidate photo by number, and lets
#     the user pick one by typing a number (or 'q' to quit).
#     Returns the chosen Path, or None if the user quits / folder is empty.
#     """
#     candidates = list_candidates(IMAGES_DIR)

#     if not candidates:
#         print(f"\n❌  No images found in '{IMAGES_DIR}/'.")
#         print(f"    Add candidate photos there (e.g. images/john_doe.jpg) and try again.")
#         return None

#     print("\n" + "═" * 50)
#     print("  📋  CANDIDATE SELECTION")
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
#                 print(f"\n✅  Selected: {name}  →  {selected}")
#                 return selected
#         print(f"   ⚠️  Enter a number between 1 and {len(candidates)}, or 'q'.")


# # ─────────────────────────────────────────────────────────────────────────────
# # Reference encoding
# # ─────────────────────────────────────────────────────────────────────────────

# def load_reference_encoding(reference_image_path: Path):
#     """
#     Loads the candidate's sample photo and returns its 128-d face encoding.
#     Returns None on any error (missing file, no face, multiple faces).
#     """
#     if not reference_image_path.exists():
#         print(f"Error: Reference image not found at '{reference_image_path}'.")
#         return None

#     reference_image = face_recognition.load_image_file(str(reference_image_path))
#     face_locations = face_recognition.face_locations(reference_image)

#     if len(face_locations) == 0:
#         print("Error: No face detected in the reference image. Use a clear, front-facing photo.")
#         return None
#     if len(face_locations) > 1:
#         print("Error: Reference image has more than one face. Use a photo with only the candidate.")
#         return None

#     return face_recognition.face_encodings(reference_image, known_face_locations=face_locations)[0]


# # ─────────────────────────────────────────────────────────────────────────────
# # Main interview session
# # ─────────────────────────────────────────────────────────────────────────────

# def start_interview_recording(reference_image_path: Path, match_tolerance=0.6, verify_every_seconds=2.0):
#     candidate_name = reference_image_path.stem.replace("_", " ").replace("-", " ").title()

#     print(f"\n🔍  Encoding reference photo for: {candidate_name} ...")
#     reference_encoding = load_reference_encoding(reference_image_path)
#     if reference_encoding is None:
#         print("Aborting: fix the reference image and try again.")
#         return
#     print(f"✅  Reference photo loaded successfully.")

#     # 1. Setup Directories
#     video_dir = Path("video_interview")
#     snapshot_dir = video_dir / "snapshots"
#     video_dir.mkdir(parents=True, exist_ok=True)
#     snapshot_dir.mkdir(parents=True, exist_ok=True)

#     # 2. Initialize Camera & Video Writer
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return

#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     safe_name  = reference_image_path.stem
#     video_path = video_dir / f"interview_{safe_name}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
#     writer     = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

#     # 3. Setup MediaPipe Models
#     mp_face = mp.solutions.face_detection
#     mp_face_mesh = mp.solutions.face_mesh
    
#     # State tracking variables for snapshots
#     last_snap_time = 0
#     last_verify_time = 0
#     last_movement_snap_time = 0  # Head movement snapshots
#     last_iris_snap_time = 0      # Iris tracking snapshots
    
#     identity_ok = None
#     font = cv2.FONT_HERSHEY_SIMPLEX

#     print(f"\n🎥  Recording to: {video_path}")
#     print(f"    Candidate : {candidate_name}")
#     print(f"    Tolerance : {match_tolerance}  |  Re-verify every: {verify_every_seconds}s")
#     print("    Press 'q' in the video window to stop.\n")

#     try:
#         with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_det, \
#              mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
            
#             while cap.isOpened():
#                 success, frame = cap.read()
#                 if not success:
#                     break

#                 rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
#                 # Detect faces & landmarks
#                 results    = face_det.process(rgb_frame)
#                 mesh_results = face_mesh.process(rgb_frame)
                
#                 face_count = len(results.detections) if results.detections else 0

#                 # Draw bounding boxes
#                 if face_count > 0:
#                     for det in results.detections:
#                         box = det.location_data.relative_bounding_box
#                         x, y = int(box.xmin * width), int(box.ymin * height)
#                         w, h = int(box.width  * width), int(box.height * height)
#                         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

#                 # ── Business logic ───────────────────────────────────────────

#                 if face_count > 1:
#                     label = f"WARNING: {face_count} PEOPLE!"
#                     cv2.putText(frame, label, (20, 50), font, 1, (0, 0, 255), 3)
#                     cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)

#                     if time.time() - last_snap_time > 3.0:
#                         snap_path = snapshot_dir / f"violation_{int(time.time())}.jpg"
#                         cv2.imwrite(str(snap_path), frame)
#                         print(f"🚨 Multiple people detected! Snapshot: {snap_path}")
#                         last_snap_time = time.time()

#                 elif face_count == 0:
#                     cv2.putText(frame, "WARNING: NO CANDIDATE", (20, 50), font, 1, (0, 165, 255), 3)
#                     identity_ok = None

#                 else:
#                     det = results.detections[0]

#                     # ── Head Movement Detection Logic ────────────────────────
#                     keypoints = det.location_data.relative_keypoints
                    
#                     # MediaPipe keypoint indices: 0:Right Eye, 1:Left Eye, 2:Nose, 3:Mouth
#                     rx, ry = int(keypoints[0].x * width), int(keypoints[0].y * height)
#                     lx, ly = int(keypoints[1].x * width), int(keypoints[1].y * height)
#                     nx, ny = int(keypoints[2].x * width), int(keypoints[2].y * height)
#                     mx, my = int(keypoints[3].x * width), int(keypoints[3].y * height)

#                     # Calculate Yaw (Left/Right)
#                     eye_center_x = (rx + lx) / 2.0
#                     eye_dist = abs(lx - rx)
#                     yaw_ratio = (nx - eye_center_x) / eye_dist if eye_dist > 0 else 0

#                     # Calculate Pitch (Up/Down)
#                     eye_center_y = (ry + ly) / 2.0
#                     face_length = my - eye_center_y
#                     pitch_ratio = (ny - eye_center_y) / face_length if face_length > 0 else 0.5

#                     if yaw_ratio < -0.07:
#                         movement_state = "LOOKING RIGHT"
#                     elif yaw_ratio > 0.07:
#                         movement_state = "LOOKING LEFT"
#                     elif pitch_ratio < 0.48:
#                         movement_state = "LOOKING UP"
#                     elif pitch_ratio > 0.57:
#                         movement_state = "LOOKING DOWN"
#                     if movement_state:
#                         cv2.putText(frame, f"WARNING: {movement_state}", (20, 90), font, 1, (0, 165, 255), 3)
                        
#                         # Take snapshot with a 3-second cooldown
#                         if time.time() - last_movement_snap_time > 3.0:
#                             snap_path = snapshot_dir / f"movement_violation_{int(time.time())}.jpg"
#                             cv2.imwrite(str(snap_path), frame)
#                             print(f"🚨 Face Movement Detected ({movement_state})! Snapshot: {snap_path}")
#                             last_movement_snap_time = time.time()

#                     # ── Iris Direction Tracking Logic ─────────────────────────
#                     if mesh_results.multi_face_landmarks:
#                         face_landmarks = mesh_results.multi_face_landmarks[0]
#                         iris_direction, r_iris_pt, l_iris_pt = detect_iris_direction(
#                             face_landmarks.landmark, width, height
#                         )

#                         # Draw iris centers on video preview frame
#                         cv2.circle(frame, r_iris_pt, 2, (0, 255, 0), -1)
#                         cv2.circle(frame, l_iris_pt, 2, (0, 255, 0), -1)

#                         if iris_direction:
#                             cv2.putText(frame, f"WARNING: {iris_direction}", (20, 130), font, 1, (0, 165, 255), 3)
                            
#                             # Snapshot cooldown timer for iris violations
#                             if time.time() - last_iris_snap_time > 3.0:
#                                 snap_path = snapshot_dir / f"iris_violation_{int(time.time())}.jpg"
#                                 cv2.imwrite(str(snap_path), frame)
#                                 print(f"🚨 Iris Movement Detected ({iris_direction})! Snapshot: {snap_path}")
#                                 last_iris_snap_time = time.time()

#                     # ── Identity verification (throttled) ────────────────────
#                     if time.time() - last_verify_time > verify_every_seconds:
#                         last_verify_time = time.time()

#                         box = det.location_data.relative_bounding_box
#                         x, y = int(box.xmin * width), int(box.ymin * height)
#                         w, h = int(box.width  * width), int(box.height * height)
#                         top, left   = max(0, y),          max(0, x)
#                         bottom, right = min(height, y+h), min(width, x+w)

#                         try:
#                             encodings = face_recognition.face_encodings(
#                                 rgb_frame,
#                                 known_face_locations=[(top, right, bottom, left)]
#                             )
#                             if encodings:
#                                 distance   = face_recognition.face_distance([reference_encoding], encodings[0])[0]
#                                 identity_ok = bool(distance <= match_tolerance)
#                                 if not identity_ok:
#                                     print(
#                                         f"🚨 IDENTITY MISMATCH — person in frame does not match "
#                                         f"'{candidate_name}' "
#                                         f"(distance={distance:.3f}, tolerance={match_tolerance})"
#                                     )
#                                     snap_path = snapshot_dir / f"identity_mismatch_{int(time.time())}.jpg"
#                                     cv2.imwrite(str(snap_path), frame)
#                                     print(f"   Snapshot saved: {snap_path}")
#                         except Exception as e:
#                             print(f"⚠️  Verification skipped: {e}")

#                     # ── Overlay based on last check result ───────────────────
#                     if identity_ok is True:
#                         cv2.putText(frame, f"Verified: {candidate_name}", (20, 50), font, 0.9, (0, 255, 0), 2)
#                     elif identity_ok is False:
#                         cv2.putText(frame, "WARNING: UNRECOGNIZED PERSON", (20, 50), font, 1, (0, 0, 255), 3)
#                         cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)
#                     else:
#                         cv2.putText(frame, "Verifying...", (20, 50), font, 1, (0, 165, 255), 2)

#                 # Record & display
#                 writer.write(frame)
#                 cv2.imshow('Interview Proctoring System', frame)

#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

#     except KeyboardInterrupt:
#         print("\nStopped via terminal (Ctrl+C).")
#     finally:
#         cap.release()
#         writer.release()
#         cv2.destroyAllWindows()
#         print(f"\n✅  Interview finished. Video saved to: {video_path}")


# # ─────────────────────────────────────────────────────────────────────────────
# # Entry point
# # ─────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     chosen = select_candidate()
#     if chosen is not None:
#         start_interview_recording(reference_image_path=chosen)




# import cv2
# import mediapipe as mp
# import face_recognition
# import time
# import os
# from pathlib import Path

# # ─────────────────────────────────────────────────────────────────────────────
# # Candidate selection
# # ─────────────────────────────────────────────────────────────────────────────

# IMAGES_DIR = Path("images")
# SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# def list_candidates(images_dir: Path) -> list[Path]:
#     """Return all image files found in the images/ folder, sorted by name."""
#     if not images_dir.exists():
#         return []
#     return sorted(
#         p for p in images_dir.iterdir()
#         if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
#     )

# def select_candidate() -> Path | None:
#     """
#     Scans the images/ folder, lists every candidate photo by number, and lets
#     the user pick one by typing a number (or 'q' to quit).
#     Returns the chosen Path, or None if the user quits / folder is empty.
#     """
#     candidates = list_candidates(IMAGES_DIR)

#     if not candidates:
#         print(f"\n❌  No images found in '{IMAGES_DIR}/'.")
#         print(f"    Add candidate photos there (e.g. images/john_doe.jpg) and try again.")
#         return None

#     print("\n" + "═" * 50)
#     print("   📋  CANDIDATE SELECTION")
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
#                 print(f"\n✅  Selected: {name}  →  {selected}")
#                 return selected
#         print(f"   ⚠️  Enter a number between 1 and {len(candidates)}, or 'q'.")


# # ─────────────────────────────────────────────────────────────────────────────
# # Reference encoding
# # ─────────────────────────────────────────────────────────────────────────────

# def load_reference_encoding(reference_image_path: Path):
#     """
#     Loads the candidate's sample photo and returns its 128-d face encoding.
#     Returns None on any error (missing file, no face, multiple faces).
#     """
#     if not reference_image_path.exists():
#         print(f"Error: Reference image not found at '{reference_image_path}'.")
#         return None

#     reference_image = face_recognition.load_image_file(str(reference_image_path))
#     face_locations = face_recognition.face_locations(reference_image)

#     if len(face_locations) == 0:
#         print("Error: No face detected in the reference image. Use a clear, front-facing photo.")
#         return None
#     if len(face_locations) > 1:
#         print("Error: Reference image has more than one face. Use a photo with only the candidate.")
#         return None

#     return face_recognition.face_encodings(reference_image, known_face_locations=face_locations)[0]


# # ─────────────────────────────────────────────────────────────────────────────
# # Main interview session
# # ─────────────────────────────────────────────────────────────────────────────

# def start_interview_recording(reference_image_path: Path, match_tolerance=0.6, verify_every_seconds=2.0):
#     candidate_name = reference_image_path.stem.replace("_", " ").replace("-", " ").title()

#     print(f"\n🔍  Encoding reference photo for: {candidate_name} ...")
#     reference_encoding = load_reference_encoding(reference_image_path)
#     if reference_encoding is None:
#         print("Aborting: fix the reference image and try again.")
#         return
#     print(f"✅  Reference photo loaded successfully.")

#     # 1. Setup Directories
#     video_dir = Path("video_interview")
#     snapshot_dir = video_dir / "snapshots"
#     video_dir.mkdir(parents=True, exist_ok=True)
#     snapshot_dir.mkdir(parents=True, exist_ok=True)

#     # 2. Initialize Camera & Video Writer
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return

#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     safe_name  = reference_image_path.stem
#     video_path = video_dir / f"interview_{safe_name}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
#     writer     = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

#     # 3. Setup MediaPipe & State Variables
#     mp_face = mp.solutions.face_detection
    
#     # State tracking variables for snapshots
#     last_snap_time = 0
#     last_verify_time = 0
#     last_movement_snap_time = 0  # Added for head movement snapshots
    
#     identity_ok = None
#     font = cv2.FONT_HERSHEY_SIMPLEX

#     print(f"\n🎥  Recording to: {video_path}")
#     print(f"    Candidate : {candidate_name}")
#     print(f"    Tolerance : {match_tolerance}  |  Re-verify every: {verify_every_seconds}s")
#     print("    Press 'q' in the video window to stop.\n")

#     try:
#         with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_det:
#             while cap.isOpened():
#                 success, frame = cap.read()
#                 if not success:
#                     break

#                 # Detect faces
#                 rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results    = face_det.process(rgb_frame)
#                 face_count = len(results.detections) if results.detections else 0

#                 # Draw bounding boxes
#                 if face_count > 0:
#                     for det in results.detections:
#                         box = det.location_data.relative_bounding_box
#                         x, y = int(box.xmin * width), int(box.ymin * height)
#                         w, h = int(box.width  * width), int(box.height * height)
#                         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

#                 # ── Business logic ───────────────────────────────────────────

#                 if face_count > 1:
#                     label = f"WARNING: {face_count} PEOPLE!"
#                     cv2.putText(frame, label, (20, 50), font, 1, (0, 0, 255), 3)
#                     cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)

#                     if time.time() - last_snap_time > 3.0:
#                         snap_path = snapshot_dir / f"violation_{int(time.time())}.jpg"
#                         cv2.imwrite(str(snap_path), frame)
#                         print(f"🚨 Multiple people detected! Snapshot: {snap_path}")
#                         last_snap_time = time.time()

#                 elif face_count == 0:
#                     cv2.putText(frame, "WARNING: NO CANDIDATE", (20, 50), font, 1, (0, 165, 255), 3)
#                     identity_ok = None

#                 else:
#                     det = results.detections[0]

#                     # ── Head Movement Detection Logic ────────────────────────
#                     keypoints = det.location_data.relative_keypoints
                    
#                     # MediaPipe keypoint indices: 0:Right Eye, 1:Left Eye, 2:Nose, 3:Mouth
#                     rx, ry = int(keypoints[0].x * width), int(keypoints[0].y * height)
#                     lx, ly = int(keypoints[1].x * width), int(keypoints[1].y * height)
#                     nx, ny = int(keypoints[2].x * width), int(keypoints[2].y * height)
#                     mx, my = int(keypoints[3].x * width), int(keypoints[3].y * height)

#                     # Calculate Yaw (Left/Right)
#                     eye_center_x = (rx + lx) / 2.0
#                     eye_dist = abs(lx - rx)
#                     yaw_ratio = (nx - eye_center_x) / eye_dist if eye_dist > 0 else 0

#                     # Calculate Pitch (Up/Down)
#                     eye_center_y = (ry + ly) / 2.0
#                     face_length = my - eye_center_y
#                     pitch_ratio = (ny - eye_center_y) / face_length if face_length > 0 else 0.5

#                     movement_state = None
#                     # Tolerances for movement (adjust these decimals if needed for your specific camera setup)
#                     if yaw_ratio < -0.15:
#                         movement_state = "LOOKING RIGHT"
#                     elif yaw_ratio > 0.15:
#                         movement_state = "LOOKING LEFT"
#                     elif pitch_ratio < 0.45:
#                         movement_state = "LOOKING UP"
#                     elif pitch_ratio > 0.60:
#                         movement_state = "LOOKING DOWN"

#                     if movement_state:
#                         cv2.putText(frame, f"WARNING: {movement_state}", (20, 90), font, 1, (0, 165, 255), 3)
                        
#                         # Take snapshot with a 3-second cooldown
#                         if time.time() - last_movement_snap_time > 3.0:
#                             snap_path = snapshot_dir / f"movement_violation_{int(time.time())}.jpg"
#                             cv2.imwrite(str(snap_path), frame)
#                             print(f"🚨 Face Movement Detected ({movement_state})! Snapshot: {snap_path}")
#                             last_movement_snap_time = time.time()

#                     # ── Identity verification (throttled) ────────────────────
#                     if time.time() - last_verify_time > verify_every_seconds:
#                         last_verify_time = time.time()

#                         box = det.location_data.relative_bounding_box
#                         x, y = int(box.xmin * width), int(box.ymin * height)
#                         w, h = int(box.width  * width), int(box.height * height)
#                         top, left   = max(0, y),          max(0, x)
#                         bottom, right = min(height, y+h), min(width, x+w)

#                         try:
#                             encodings = face_recognition.face_encodings(
#                                 rgb_frame,
#                                 known_face_locations=[(top, right, bottom, left)]
#                             )
#                             if encodings:
#                                 distance   = face_recognition.face_distance([reference_encoding], encodings[0])[0]
#                                 identity_ok = bool(distance <= match_tolerance)
#                                 if not identity_ok:
#                                     print(
#                                         f"🚨 IDENTITY MISMATCH — person in frame does not match "
#                                         f"'{candidate_name}' "
#                                         f"(distance={distance:.3f}, tolerance={match_tolerance})"
#                                     )
#                                     snap_path = snapshot_dir / f"identity_mismatch_{int(time.time())}.jpg"
#                                     cv2.imwrite(str(snap_path), frame)
#                                     print(f"   Snapshot saved: {snap_path}")
#                         except Exception as e:
#                             print(f"⚠️  Verification skipped: {e}")

#                     # ── Overlay based on last check result ───────────────────
#                     if identity_ok is True:
#                         cv2.putText(frame, f"Verified: {candidate_name}", (20, 50), font, 0.9, (0, 255, 0), 2)
#                     elif identity_ok is False:
#                         cv2.putText(frame, "WARNING: UNRECOGNIZED PERSON", (20, 50), font, 1, (0, 0, 255), 3)
#                         cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)
#                     else:
#                         cv2.putText(frame, "Verifying...", (20, 50), font, 1, (0, 165, 255), 2)

#                 # Record & display
#                 writer.write(frame)
#                 cv2.imshow('Interview Proctoring System', frame)

#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

#     except KeyboardInterrupt:
#         print("\nStopped via terminal (Ctrl+C).")
#     finally:
#         cap.release()
#         writer.release()
#         cv2.destroyAllWindows()
#         print(f"\n✅  Interview finished. Video saved to: {video_path}")


# # ─────────────────────────────────────────────────────────────────────────────
# # Entry point
# # ─────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     chosen = select_candidate()
#     if chosen is not None:
#         start_interview_recording(reference_image_path=chosen)

# import cv2
# import mediapipe as mp
# from iris_detector import detect_iris_direction  # Import your new file

# mp_face_mesh = mp.solutions.face_mesh

# # Must set refine_landmarks=True to track the iris points
# with mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
#     cap = cv2.VideoCapture(0)
    
#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             break
            
#         h, w, _ = frame.shape
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(rgb_frame)
        
#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 # Pass the landmarks array into your imported function
#                 direction, r_iris, l_iris = detect_iris_direction(face_landmarks.landmark, w, h)
                
#                 if direction:
#                     print(f"Warning: {direction}")
                    
#                 # Optionally draw points on the eyes
#                 cv2.circle(frame, r_iris, 2, (0, 255, 0), -1)
#                 cv2.circle(frame, l_iris, 2, (0, 255, 0), -1)

#         cv2.imshow('Iris Test', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()





# import cv2
# import mediapipe as mp
# import face_recognition
# import time
# import os
# from pathlib import Path

# # ─────────────────────────────────────────────────────────────────────────────
# # Candidate selection
# # ─────────────────────────────────────────────────────────────────────────────

# IMAGES_DIR = Path("images")
# SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# def list_candidates(images_dir: Path) -> list[Path]:
#     """Return all image files found in the images/ folder, sorted by name."""
#     if not images_dir.exists():
#         return []
#     return sorted(
#         p for p in images_dir.iterdir()
#         if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
#     )

# def select_candidate() -> Path | None:
#     """
#     Scans the images/ folder, lists every candidate photo by number, and lets
#     the user pick one by typing a number (or 'q' to quit).
#     Returns the chosen Path, or None if the user quits / folder is empty.
#     """
#     candidates = list_candidates(IMAGES_DIR)

#     if not candidates:
#         print(f"\n❌  No images found in '{IMAGES_DIR}/'.")
#         print(f"    Add candidate photos there (e.g. images/john_doe.jpg) and try again.")
#         return None

#     print("\n" + "═" * 50)
#     print("   📋  CANDIDATE SELECTION")
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
#                 print(f"\n✅  Selected: {name}  →  {selected}")
#                 return selected
#         print(f"   ⚠️  Enter a number between 1 and {len(candidates)}, or 'q'.")


# # ─────────────────────────────────────────────────────────────────────────────
# # Reference encoding
# # ─────────────────────────────────────────────────────────────────────────────

# def load_reference_encoding(reference_image_path: Path):
#     """
#     Loads the candidate's sample photo and returns its 128-d face encoding.
#     Returns None on any error (missing file, no face, multiple faces).
#     """
#     if not reference_image_path.exists():
#         print(f"Error: Reference image not found at '{reference_image_path}'.")
#         return None

#     reference_image = face_recognition.load_image_file(str(reference_image_path))
#     face_locations = face_recognition.face_locations(reference_image)

#     if len(face_locations) == 0:
#         print("Error: No face detected in the reference image. Use a clear, front-facing photo.")
#         return None
#     if len(face_locations) > 1:
#         print("Error: Reference image has more than one face. Use a photo with only the candidate.")
#         return None

#     return face_recognition.face_encodings(reference_image, known_face_locations=face_locations)[0]


# # ─────────────────────────────────────────────────────────────────────────────
# # Main interview session
# # ─────────────────────────────────────────────────────────────────────────────

# def start_interview_recording(reference_image_path: Path, match_tolerance=0.6, verify_every_seconds=2.0):
#     candidate_name = reference_image_path.stem.replace("_", " ").replace("-", " ").title()

#     print(f"\n🔍  Encoding reference photo for: {candidate_name} ...")
#     reference_encoding = load_reference_encoding(reference_image_path)
#     if reference_encoding is None:
#         print("Aborting: fix the reference image and try again.")
#         return
#     print(f"✅  Reference photo loaded successfully.")

#     # 1. Setup Directories
#     video_dir = Path("video_interview")
#     snapshot_dir = video_dir / "snapshots"
#     video_dir.mkdir(parents=True, exist_ok=True)
#     snapshot_dir.mkdir(parents=True, exist_ok=True)

#     # 2. Initialize Camera & Video Writer
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return

#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     safe_name  = reference_image_path.stem
#     video_path = video_dir / f"interview_{safe_name}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
#     writer     = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

#     # 3. Setup MediaPipe & State Variables
#     mp_face = mp.solutions.face_detection
    
#     # State tracking variables for snapshots
#     last_snap_time = 0
#     last_verify_time = 0
#     last_movement_snap_time = 0  # Added for head movement snapshots
    
#     identity_ok = None
#     font = cv2.FONT_HERSHEY_SIMPLEX

#     print(f"\n🎥  Recording to: {video_path}")
#     print(f"    Candidate : {candidate_name}")
#     print(f"    Tolerance : {match_tolerance}  |  Re-verify every: {verify_every_seconds}s")
#     print("    Press 'q' in the video window to stop.\n")

#     try:
#         with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_det:
#             while cap.isOpened():
#                 success, frame = cap.read()
#                 if not success:
#                     break

#                 # Detect faces
#                 rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results    = face_det.process(rgb_frame)
#                 face_count = len(results.detections) if results.detections else 0

#                 # Draw bounding boxes
#                 if face_count > 0:
#                     for det in results.detections:
#                         box = det.location_data.relative_bounding_box
#                         x, y = int(box.xmin * width), int(box.ymin * height)
#                         w, h = int(box.width  * width), int(box.height * height)
#                         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

#                 # ── Business logic ───────────────────────────────────────────

#                 if face_count > 1:
#                     label = f"WARNING: {face_count} PEOPLE!"
#                     cv2.putText(frame, label, (20, 50), font, 1, (0, 0, 255), 3)
#                     cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)

#                     if time.time() - last_snap_time > 3.0:
#                         snap_path = snapshot_dir / f"violation_{int(time.time())}.jpg"
#                         cv2.imwrite(str(snap_path), frame)
#                         print(f"🚨 Multiple people detected! Snapshot: {snap_path}")
#                         last_snap_time = time.time()

#                 elif face_count == 0:
#                     cv2.putText(frame, "WARNING: NO CANDIDATE", (20, 50), font, 1, (0, 165, 255), 3)
#                     identity_ok = None

#                 else:
#                     det = results.detections[0]

#                     # ── Head Movement Detection Logic ────────────────────────
#                     keypoints = det.location_data.relative_keypoints
                    
#                     # MediaPipe keypoint indices: 0:Right Eye, 1:Left Eye, 2:Nose, 3:Mouth
#                     rx, ry = int(keypoints[0].x * width), int(keypoints[0].y * height)
#                     lx, ly = int(keypoints[1].x * width), int(keypoints[1].y * height)
#                     nx, ny = int(keypoints[2].x * width), int(keypoints[2].y * height)
#                     mx, my = int(keypoints[3].x * width), int(keypoints[3].y * height)

#                     # Calculate Yaw (Left/Right)
#                     eye_center_x = (rx + lx) / 2.0
#                     eye_dist = abs(lx - rx)
#                     yaw_ratio = (nx - eye_center_x) / eye_dist if eye_dist > 0 else 0

#                     # Calculate Pitch (Up/Down)
#                     eye_center_y = (ry + ly) / 2.0
#                     face_length = my - eye_center_y
#                     pitch_ratio = (ny - eye_center_y) / face_length if face_length > 0 else 0.5

#                     movement_state = None
#                     # Tolerances for movement (adjust these decimals if needed for your specific camera setup)
#                     if yaw_ratio < -0.15:
#                         movement_state = "LOOKING RIGHT"
#                     elif yaw_ratio > 0.15:
#                         movement_state = "LOOKING LEFT"
#                     elif pitch_ratio < 0.45:
#                         movement_state = "LOOKING UP"
#                     elif pitch_ratio > 0.60:
#                         movement_state = "LOOKING DOWN"

#                     if movement_state:
#                         cv2.putText(frame, f"WARNING: {movement_state}", (20, 90), font, 1, (0, 165, 255), 3)
                        
#                         # Take snapshot with a 3-second cooldown
#                         if time.time() - last_movement_snap_time > 3.0:
#                             snap_path = snapshot_dir / f"movement_violation_{int(time.time())}.jpg"
#                             cv2.imwrite(str(snap_path), frame)
#                             print(f"🚨 Face Movement Detected ({movement_state})! Snapshot: {snap_path}")
#                             last_movement_snap_time = time.time()

#                     # ── Identity verification (throttled) ────────────────────
#                     if time.time() - last_verify_time > verify_every_seconds:
#                         last_verify_time = time.time()

#                         box = det.location_data.relative_bounding_box
#                         x, y = int(box.xmin * width), int(box.ymin * height)
#                         w, h = int(box.width  * width), int(box.height * height)
#                         top, left   = max(0, y),          max(0, x)
#                         bottom, right = min(height, y+h), min(width, x+w)

#                         try:
#                             encodings = face_recognition.face_encodings(
#                                 rgb_frame,
#                                 known_face_locations=[(top, right, bottom, left)]
#                             )
#                             if encodings:
#                                 distance   = face_recognition.face_distance([reference_encoding], encodings[0])[0]
#                                 identity_ok = bool(distance <= match_tolerance)
#                                 if not identity_ok:
#                                     print(
#                                         f"🚨 IDENTITY MISMATCH — person in frame does not match "
#                                         f"'{candidate_name}' "
#                                         f"(distance={distance:.3f}, tolerance={match_tolerance})"
#                                     )
#                                     snap_path = snapshot_dir / f"identity_mismatch_{int(time.time())}.jpg"
#                                     cv2.imwrite(str(snap_path), frame)
#                                     print(f"   Snapshot saved: {snap_path}")
#                         except Exception as e:
#                             print(f"⚠️  Verification skipped: {e}")

#                     # ── Overlay based on last check result ───────────────────
#                     if identity_ok is True:
#                         cv2.putText(frame, f"Verified: {candidate_name}", (20, 50), font, 0.9, (0, 255, 0), 2)
#                     elif identity_ok is False:
#                         cv2.putText(frame, "WARNING: UNRECOGNIZED PERSON", (20, 50), font, 1, (0, 0, 255), 3)
#                         cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)
#                     else:
#                         cv2.putText(frame, "Verifying...", (20, 50), font, 1, (0, 165, 255), 2)

#                 # Record & display
#                 writer.write(frame)
#                 cv2.imshow('Interview Proctoring System', frame)

#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

#     except KeyboardInterrupt:
#         print("\nStopped via terminal (Ctrl+C).")
#     finally:
#         cap.release()
#         writer.release()
#         cv2.destroyAllWindows()
#         print(f"\n✅  Interview finished. Video saved to: {video_path}")


# # ─────────────────────────────────────────────────────────────────────────────
# # Entry point
# # ─────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     chosen = select_candidate()
#     if chosen is not None:
#         start_interview_recording(reference_image_path=chosen)





# import cv2
# import mediapipe as mp
# import face_recognition
# import time
# import os
# from pathlib import Path


# # ─────────────────────────────────────────────────────────────────────────────
# # Candidate selection
# # ─────────────────────────────────────────────────────────────────────────────

# IMAGES_DIR = Path("images")
# SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# def list_candidates(images_dir: Path) -> list[Path]:
#     """Return all image files found in the images/ folder, sorted by name."""
#     if not images_dir.exists():
#         return []
#     return sorted(
#         p for p in images_dir.iterdir()
#         if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
#     )


# def select_candidate() -> Path | None:
#     """
#     Scans the images/ folder, lists every candidate photo by number, and lets
#     the user pick one by typing a number (or 'q' to quit).
#     Returns the chosen Path, or None if the user quits / folder is empty.
#     """
#     candidates = list_candidates(IMAGES_DIR)

#     if not candidates:
#         print(f"\n❌  No images found in '{IMAGES_DIR}/'.")
#         print(f"    Add candidate photos there (e.g. images/john_doe.jpg) and try again.")
#         return None

#     print("\n" + "═" * 50)
#     print("   📋  CANDIDATE SELECTION")
#     print("═" * 50)
#     for idx, path in enumerate(candidates, start=1):
#         # Show the stem (filename without extension) as the candidate name
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
#                 print(f"\n✅  Selected: {name}  →  {selected}")
#                 return selected
#         print(f"   ⚠️  Enter a number between 1 and {len(candidates)}, or 'q'.")


# # ─────────────────────────────────────────────────────────────────────────────
# # Reference encoding
# # ─────────────────────────────────────────────────────────────────────────────

# def load_reference_encoding(reference_image_path: Path):
#     """
#     Loads the candidate's sample photo and returns its 128-d face encoding.
#     Returns None on any error (missing file, no face, multiple faces).
#     """
#     if not reference_image_path.exists():
#         print(f"Error: Reference image not found at '{reference_image_path}'.")
#         return None

#     reference_image = face_recognition.load_image_file(str(reference_image_path))
#     face_locations = face_recognition.face_locations(reference_image)

#     if len(face_locations) == 0:
#         print("Error: No face detected in the reference image. Use a clear, front-facing photo.")
#         return None
#     if len(face_locations) > 1:
#         print("Error: Reference image has more than one face. Use a photo with only the candidate.")
#         return None

#     return face_recognition.face_encodings(reference_image, known_face_locations=face_locations)[0]


# # ─────────────────────────────────────────────────────────────────────────────
# # Main interview session
# # ─────────────────────────────────────────────────────────────────────────────

# def start_interview_recording(reference_image_path: Path, match_tolerance=0.6, verify_every_seconds=2.0):
#     candidate_name = reference_image_path.stem.replace("_", " ").replace("-", " ").title()

#     print(f"\n🔍  Encoding reference photo for: {candidate_name} ...")
#     reference_encoding = load_reference_encoding(reference_image_path)
#     if reference_encoding is None:
#         print("Aborting: fix the reference image and try again.")
#         return
#     print(f"✅  Reference photo loaded successfully.")

#     # 1. Setup Directories
#     video_dir = Path("video_interview")
#     snapshot_dir = video_dir / "snapshots"
#     video_dir.mkdir(parents=True, exist_ok=True)
#     snapshot_dir.mkdir(parents=True, exist_ok=True)

#     # 2. Initialize Camera & Video Writer
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return

#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     safe_name  = reference_image_path.stem
#     video_path = video_dir / f"interview_{safe_name}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
#     writer     = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

#     # 3. Setup MediaPipe & State Variables
#     mp_face          = mp.solutions.face_detection
#     last_snap_time   = 0
#     last_verify_time = 0
#     identity_ok      = None      # None = pending first check, True/False after a check
#     font             = cv2.FONT_HERSHEY_SIMPLEX

#     print(f"\n🎥  Recording to: {video_path}")
#     print(f"    Candidate : {candidate_name}")
#     print(f"    Tolerance : {match_tolerance}  |  Re-verify every: {verify_every_seconds}s")
#     print("    Press 'q' in the video window to stop.\n")

#     try:
#         with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_det:
#             while cap.isOpened():
#                 success, frame = cap.read()
#                 if not success:
#                     break

#                 # Detect faces
#                 rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results    = face_det.process(rgb_frame)
#                 face_count = len(results.detections) if results.detections else 0

#                 # Draw bounding boxes
#                 if face_count > 0:
#                     for det in results.detections:
#                         box = det.location_data.relative_bounding_box
#                         x, y = int(box.xmin * width), int(box.ymin * height)
#                         w, h = int(box.width  * width), int(box.height * height)
#                         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

#                 # ── Business logic ───────────────────────────────────────────

#                 if face_count > 1:
#                     label = f"WARNING: {face_count} PEOPLE!"
#                     cv2.putText(frame, label, (20, 50), font, 1, (0, 0, 255), 3)
#                     cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)

#                     if time.time() - last_snap_time > 3.0:
#                         snap_path = snapshot_dir / f"violation_{int(time.time())}.jpg"
#                         cv2.imwrite(str(snap_path), frame)
#                         print(f"🚨 Multiple people detected! Snapshot: {snap_path}")
#                         last_snap_time = time.time()

#                 elif face_count == 0:
#                     cv2.putText(frame, "WARNING: NO CANDIDATE", (20, 50), font, 1, (0, 165, 255), 3)
#                     identity_ok = None

#                 else:
#                     # ── Identity verification (throttled) ─────────────────
#                     if time.time() - last_verify_time > verify_every_seconds:
#                         last_verify_time = time.time()

#                         box = results.detections[0].location_data.relative_bounding_box
#                         x, y = int(box.xmin * width), int(box.ymin * height)
#                         w, h = int(box.width  * width), int(box.height * height)
#                         top, left   = max(0, y),          max(0, x)
#                         bottom, right = min(height, y+h), min(width, x+w)

#                         try:
#                             encodings = face_recognition.face_encodings(
#                                 rgb_frame,
#                                 known_face_locations=[(top, right, bottom, left)]
#                             )
#                             if encodings:
#                                 distance   = face_recognition.face_distance([reference_encoding], encodings[0])[0]
#                                 identity_ok = bool(distance <= match_tolerance)
#                                 if not identity_ok:
#                                     print(
#                                         f"🚨 IDENTITY MISMATCH — person in frame does not match "
#                                         f"'{candidate_name}' "
#                                         f"(distance={distance:.3f}, tolerance={match_tolerance})"
#                                     )
#                                     snap_path = snapshot_dir / f"identity_mismatch_{int(time.time())}.jpg"
#                                     cv2.imwrite(str(snap_path), frame)
#                                     print(f"   Snapshot saved: {snap_path}")
#                         except Exception as e:
#                             print(f"⚠️  Verification skipped: {e}")

#                     # ── Overlay based on last check result ────────────────
#                     if identity_ok is True:
#                         cv2.putText(frame, f"Verified: {candidate_name}", (20, 50), font, 0.9, (0, 255, 0), 2)
#                     elif identity_ok is False:
#                         cv2.putText(frame, "WARNING: UNRECOGNIZED PERSON", (20, 50), font, 1, (0, 0, 255), 3)
#                         cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)
#                     else:
#                         cv2.putText(frame, "Verifying...", (20, 50), font, 1, (0, 165, 255), 2)

#                 # Record & display
#                 writer.write(frame)
#                 cv2.imshow('Interview Proctoring System', frame)

#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

#     except KeyboardInterrupt:
#         print("\nStopped via terminal (Ctrl+C).")
#     finally:
#         cap.release()
#         writer.release()
#         cv2.destroyAllWindows()
#         print(f"\n✅  Interview finished. Video saved to: {video_path}")


# # ─────────────────────────────────────────────────────────────────────────────
# # Entry point
# # ─────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     chosen = select_candidate()
#     if chosen is not None:
#         start_interview_recording(reference_image_path=chosen)








