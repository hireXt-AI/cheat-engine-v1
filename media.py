import cv2
import mediapipe as mp
import face_recognition
import time
import os
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Candidate selection
# ─────────────────────────────────────────────────────────────────────────────

IMAGES_DIR = Path("images")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_candidates(images_dir: Path) -> list[Path]:
    """Return all image files found in the images/ folder, sorted by name."""
    if not images_dir.exists():
        return []
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

def select_candidate() -> Path | None:
    """
    Scans the images/ folder, lists every candidate photo by number, and lets
    the user pick one by typing a number (or 'q' to quit).
    Returns the chosen Path, or None if the user quits / folder is empty.
    """
    candidates = list_candidates(IMAGES_DIR)

    if not candidates:
        print(f"\n❌  No images found in '{IMAGES_DIR}/'.")
        print(f"    Add candidate photos there (e.g. images/john_doe.jpg) and try again.")
        return None

    print("\n" + "═" * 50)
    print("   📋  CANDIDATE SELECTION")
    print("═" * 50)
    for idx, path in enumerate(candidates, start=1):
        name = path.stem.replace("_", " ").replace("-", " ").title()
        print(f"  [{idx}]  {name}  ({path.name})")
    print("  [q]  Quit")
    print("═" * 50)

    while True:
        raw = input("Select candidate number: ").strip().lower()
        if raw == "q":
            return None
        if raw.isdigit():
            choice = int(raw)
            if 1 <= choice <= len(candidates):
                selected = candidates[choice - 1]
                name = selected.stem.replace("_", " ").replace("-", " ").title()
                print(f"\n✅  Selected: {name}  →  {selected}")
                return selected
        print(f"   ⚠️  Enter a number between 1 and {len(candidates)}, or 'q'.")


# ─────────────────────────────────────────────────────────────────────────────
# Reference encoding
# ─────────────────────────────────────────────────────────────────────────────

def load_reference_encoding(reference_image_path: Path):
    """
    Loads the candidate's sample photo and returns its 128-d face encoding.
    Returns None on any error (missing file, no face, multiple faces).
    """
    if not reference_image_path.exists():
        print(f"Error: Reference image not found at '{reference_image_path}'.")
        return None

    reference_image = face_recognition.load_image_file(str(reference_image_path))
    face_locations = face_recognition.face_locations(reference_image)

    if len(face_locations) == 0:
        print("Error: No face detected in the reference image. Use a clear, front-facing photo.")
        return None
    if len(face_locations) > 1:
        print("Error: Reference image has more than one face. Use a photo with only the candidate.")
        return None

    return face_recognition.face_encodings(reference_image, known_face_locations=face_locations)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Iris Tracking Helper
# ─────────────────────────────────────────────────────────────────────────────

def detect_iris_direction(landmarks, frame_width, frame_height):
    """
    Calculates normalized position of iris relative to eye corners and eyelids.
    Returns (direction_label, right_iris_pixel, left_iris_pixel)
    """
    LEFT_IRIS_CENTER = 468
    LEFT_EYE_INNER   = 133
    LEFT_EYE_OUTER   = 33
    LEFT_EYE_TOP     = 159
    LEFT_EYE_BOTTOM  = 145

    RIGHT_IRIS_CENTER = 473
    RIGHT_EYE_INNER   = 362
    RIGHT_EYE_OUTER   = 263
    RIGHT_EYE_TOP     = 386
    RIGHT_EYE_BOTTOM  = 374

    def get_pt(idx):
        return np.array([landmarks[idx].x * frame_width, landmarks[idx].y * frame_height])

    # Left Eye
    l_iris = get_pt(LEFT_IRIS_CENTER)
    l_inner, l_outer = get_pt(LEFT_EYE_INNER), get_pt(LEFT_EYE_OUTER)
    l_top, l_bottom   = get_pt(LEFT_EYE_TOP), get_pt(LEFT_EYE_BOTTOM)

    l_horiz_dist = np.linalg.norm(l_outer - l_inner)
    l_vert_dist  = np.linalg.norm(l_bottom - l_top)

    l_h_ratio = (np.linalg.norm(l_iris - l_inner) / l_horiz_dist) if l_horiz_dist > 0 else 0.5
    l_v_ratio = (np.linalg.norm(l_iris - l_top) / l_vert_dist) if l_vert_dist > 0 else 0.5

    # Right Eye
    r_iris = get_pt(RIGHT_IRIS_CENTER)
    r_inner, r_outer = get_pt(RIGHT_EYE_INNER), get_pt(RIGHT_EYE_OUTER)
    r_top, r_bottom   = get_pt(RIGHT_EYE_TOP), get_pt(RIGHT_EYE_BOTTOM)

    r_horiz_dist = np.linalg.norm(r_outer - r_inner)
    r_vert_dist  = np.linalg.norm(r_bottom - r_top)

    r_h_ratio = (np.linalg.norm(r_iris - r_inner) / r_horiz_dist) if r_horiz_dist > 0 else 0.5
    r_v_ratio = (np.linalg.norm(r_iris - r_top) / r_vert_dist) if r_vert_dist > 0 else 0.5

    # Averages
    avg_h_ratio = (l_h_ratio + r_h_ratio) / 2.0
    avg_v_ratio = (l_v_ratio + r_v_ratio) / 2.0

    # Direction thresholds
    direction = None
    if avg_h_ratio < 0.35:
        direction = "IRIS LEFT"
    elif avg_h_ratio > 0.65:
        direction = "IRIS RIGHT"
    elif avg_v_ratio < 0.30:
        direction = "IRIS UP"
    elif avg_v_ratio > 0.75:
        direction = "IRIS DOWN"

    right_iris_pixel = (int(r_iris[0]), int(r_iris[1]))
    left_iris_pixel  = (int(l_iris[0]), int(l_iris[1]))

    return direction, right_iris_pixel, left_iris_pixel


# ─────────────────────────────────────────────────────────────────────────────
# Main interview session
# ─────────────────────────────────────────────────────────────────────────────

def start_interview_recording(reference_image_path: Path, match_tolerance=0.6, verify_every_seconds=2.0):
    candidate_name = reference_image_path.stem.replace("_", " ").replace("-", " ").title()

    print(f"\n🔍  Encoding reference photo for: {candidate_name} ...")
    reference_encoding = load_reference_encoding(reference_image_path)
    if reference_encoding is None:
        print("Aborting: fix the reference image and try again.")
        return
    print(f"✅  Reference photo loaded successfully.")

    # 1. Setup Directories
    video_dir = Path("video_interview")
    snapshot_dir = video_dir / "snapshots"
    video_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # 2. Initialize Camera & Video Writer
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    safe_name  = reference_image_path.stem
    video_path = video_dir / f"interview_{safe_name}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    writer     = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

    # 3. Setup MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    
    # Cooldown trackers
    last_snap_time = 0
    last_verify_time = 0
    last_iris_snap_time = 0
    
    identity_ok = None
    font = cv2.FONT_HERSHEY_SIMPLEX

    print(f"\n🎥  Recording to: {video_path}")
    print(f"    Candidate : {candidate_name}")
    print(f"    Tolerance : {match_tolerance}  |  Re-verify every: {verify_every_seconds}s")
    print("    Press 'q' in the video window to stop.\n")

    try:
        with mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,  # Enables iris tracking (landmarks 468-477)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results   = face_mesh.process(rgb_frame)
                face_count = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0

                # ── Multiple People Check ────────────────────────────────────
                if face_count > 1:
                    label = f"WARNING: {face_count} PEOPLE!"
                    cv2.putText(frame, label, (20, 50), font, 1, (0, 0, 255), 3)
                    cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)

                    if time.time() - last_snap_time > 3.0:
                        snap_path = snapshot_dir / f"violation_{int(time.time())}.jpg"
                        cv2.imwrite(str(snap_path), frame)
                        print(f"🚨 Multiple people detected! Snapshot: {snap_path}")
                        last_snap_time = time.time()

                # ── No Person Check ──────────────────────────────────────────
                elif face_count == 0:
                    cv2.putText(frame, "WARNING: NO CANDIDATE", (20, 50), font, 1, (0, 165, 255), 3)
                    identity_ok = None

                # ── Single Candidate Track ────────────────────────────────────
                else:
                    face_landmarks = results.multi_face_landmarks[0]

                    # 1. Iris Detection
                    iris_direction, r_iris_pt, l_iris_pt = detect_iris_direction(
                        face_landmarks.landmark, width, height
                    )
                    
                    # Draw green visual tracking points on pupils
                    cv2.circle(frame, r_iris_pt, 3, (0, 255, 0), -1)
                    cv2.circle(frame, l_iris_pt, 3, (0, 255, 0), -1)

                    if iris_direction:
                        cv2.putText(frame, f"WARNING: {iris_direction}", (20, 90), font, 1, (0, 165, 255), 3)
                        
                        # 3-second cooldown between snapshots
                        if time.time() - last_iris_snap_time > 3.0:
                            snap_path = snapshot_dir / f"iris_violation_{int(time.time())}.jpg"
                            cv2.imwrite(str(snap_path), frame)
                            print(f"🚨 Iris Movement Detected ({iris_direction})! Snapshot: {snap_path}")
                            last_iris_snap_time = time.time()

                    # 2. Get Face Bounding Box
                    h_pts = [int(pt.y * height) for pt in face_landmarks.landmark]
                    w_pts = [int(pt.x * width) for pt in face_landmarks.landmark]
                    top, bottom = max(0, min(h_pts)), min(height, max(h_pts))
                    left, right = max(0, min(w_pts)), min(width, max(w_pts))
                    
                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)

                    # 3. Throttled Identity Verification
                    if time.time() - last_verify_time > verify_every_seconds:
                        last_verify_time = time.time()

                        try:
                            encodings = face_recognition.face_encodings(
                                rgb_frame,
                                known_face_locations=[(top, right, bottom, left)]
                            )
                            if encodings:
                                distance   = face_recognition.face_distance([reference_encoding], encodings[0])[0]
                                identity_ok = bool(distance <= match_tolerance)
                                if not identity_ok:
                                    print(
                                        f"🚨 IDENTITY MISMATCH — person in frame does not match "
                                        f"'{candidate_name}' "
                                        f"(distance={distance:.3f}, tolerance={match_tolerance})"
                                    )
                                    snap_path = snapshot_dir / f"identity_mismatch_{int(time.time())}.jpg"
                                    cv2.imwrite(str(snap_path), frame)
                                    print(f"   Snapshot saved: {snap_path}")
                        except Exception as e:
                            print(f"⚠️  Verification skipped: {e}")

                    # 4. Status Overlay
                    if identity_ok is True:
                        cv2.putText(frame, f"Verified: {candidate_name}", (20, 50), font, 0.9, (0, 255, 0), 2)
                    elif identity_ok is False:
                        cv2.putText(frame, "WARNING: UNRECOGNIZED PERSON", (20, 50), font, 1, (0, 0, 255), 3)
                        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 10)
                    else:
                        cv2.putText(frame, "Verifying...", (20, 50), font, 1, (0, 165, 255), 2)

                # Record frame & display feed
                writer.write(frame)
                cv2.imshow('Interview Proctoring System', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nStopped via terminal (Ctrl+C).")
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f"\n✅  Interview finished. Video saved to: {video_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    chosen = select_candidate()
    if chosen is not None:
        start_interview_recording(reference_image_path=chosen)



# # iris_detector.py
# import numpy as np

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