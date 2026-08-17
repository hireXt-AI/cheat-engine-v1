import argparse
import cv2
import numpy as np
import os


def load_cascade(filename):
    """Load a Haar cascade classifier by filename."""
    cascade_path = cv2.data.haarcascades + filename
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(
            f"Could not load cascade classifier from {cascade_path}. "
            "Verify OpenCV installation and the cascade path."
        )
    return cascade


def load_face_cascade():
    return load_cascade('haarcascade_frontalface_default.xml')


def load_eye_cascades():
    return (
        load_cascade('haarcascade_eye.xml'),
        load_cascade('haarcascade_eye_tree_eyeglasses.xml'),
    )


def detect_faces_in_image(image_path):
    """Detect faces in a single image and return the image, faces, and grayscale image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}. Check the file path.")

    face_cascade = load_face_cascade()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return img, gray, faces


def draw_faces(image, faces, label=None):
    """Draw rectangles and optional label for each detected face."""
    for (x, y, w, h) in faces:
        color = (255, 0, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        if label:
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def save_image(image, output_path):
    cv2.imwrite(output_path, image)
    return output_path


def crop_face_region(image, face):
    x, y, w, h = face
    return image[y:y + h, x:x + w]


def compute_orb_features(gray):
    orb = cv2.ORB_create(500)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def compare_faces(reference_gray, candidate_gray):
    ref_kp, ref_des = compute_orb_features(reference_gray)
    cand_kp, cand_des = compute_orb_features(candidate_gray)
    if ref_des is None or cand_des is None or len(ref_des) == 0 or len(cand_des) == 0:
        return 0.0, 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(ref_des, cand_des)
    good_matches = [m for m in matches if m.distance < 60]
    similarity = len(good_matches) / max(1, min(len(ref_kp), len(cand_kp)))
    return similarity, len(good_matches)


def detect_eyes(face_gray):
    eye_cascade, eye_glasses_cascade = load_eye_cascades()
    eyes = eye_cascade.detectMultiScale(
        face_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(15, 15),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    glass_eyes = eye_glasses_cascade.detectMultiScale(
        face_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(15, 15),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return eyes, glass_eyes


def classify_eye_state(eyes, glass_eyes):
    if len(eyes) >= 2:
        return "Eyes open", False
    if len(eyes) == 0 and len(glass_eyes) > 0:
        return "Sunglasses", True
    if len(eyes) == 0:
        return "Eyes closed/occluded", False
    return "One eye visible", False


def find_pupil_center(eye_gray):
    blurred = cv2.GaussianBlur(eye_gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 20:
        return None

    moments = cv2.moments(contour)
    if moments['m00'] == 0:
        return None

    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    return cx / max(1, eye_gray.shape[1]), cy / max(1, eye_gray.shape[0]), (cx, cy)


def estimate_gaze(face_gray, eyes):
    centers = []
    for (ex, ey, ew, eh) in eyes[:2]:
        eye_phase = face_gray[ey:ey + eh, ex:ex + ew]
        pupil = find_pupil_center(eye_phase)
        if pupil is not None:
            centers.append(pupil[:2])

    if not centers:
        return "Gaze unknown", False

    avg_x = np.mean([c[0] for c in centers])
    avg_y = np.mean([c[1] for c in centers])
    looking = 0.25 < avg_x < 0.75 and 0.25 < avg_y < 0.75
    return ("Looking at screen" if looking else "Looking away"), looking


def get_light_status(frame_gray):
    brightness = float(np.mean(frame_gray))
    low_light = brightness < 60
    light_on = brightness > 100
    return brightness, low_light, light_on


def annotate_status(frame, status_lines):
    y = 30
    for line in status_lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 22


def detect_faces(image_path, output_path="detected_faces.jpg"):
    img, _, faces = detect_faces_in_image(image_path)
    draw_faces(img, faces)
    output_path = save_image(img, output_path)
    return len(faces), output_path


def match_reference_image_in_camera(reference_path, camera_index=0, match_threshold=0.25, min_good_matches=10):
    """Match reference face image against live camera frames and perform live status detection."""
    if not os.path.isfile(reference_path):
        raise FileNotFoundError(f"Reference image file not found: {reference_path}")

    ref_img, ref_gray, ref_faces = detect_faces_in_image(reference_path)
    if len(ref_faces) == 0:
        raise ValueError("No face detected in the reference image. Use a clear frontal face.")

    reference_face = crop_face_region(ref_gray, ref_faces[0])
    ref_face = cv2.resize(reference_face, (200, 200))
    face_cascade = load_face_cascade()
    eye_cascade, eye_glasses_cascade = load_eye_cascades()

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    print("Starting live camera. Press 'q' to quit.")
    while True:
        grabbed, frame = camera.read()
        if not grabbed:
            print("Could not read frame from camera.")
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness, low_light, light_on = get_light_status(frame_gray)

        faces = face_cascade.detectMultiScale(
            frame_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        status_lines = [
            f"Person in frame: {'Yes' if len(faces) > 0 else 'No'}",
            f"Light on: {'Yes' if light_on else 'No'}",
            f"Low light: {'Yes' if low_light else 'No'}",
            f"Brightness: {brightness:.0f}",
        ]

        match_found = False
        for face in faces:
            x, y, w, h = face
            face_gray = frame_gray[y:y + h, x:x + w]
            eyes, glass_eyes = detect_eyes(face_gray)
            eye_state, sunglasses = classify_eye_state(eyes, glass_eyes)
            gaze_text, looking = estimate_gaze(face_gray, eyes)

            face_region = cv2.resize(face_gray, (200, 200))
            similarity, good_matches = compare_faces(ref_face, face_region)
            match_label = "Match" if similarity >= match_threshold and good_matches >= min_good_matches else "No match"
            color = (0, 255, 0) if match_label == "Match" else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{match_label} ({good_matches})", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, eye_state, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, gaze_text, (x, y + h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            if sunglasses:
                cv2.putText(frame, "Sunglasses detected", (x, y + h + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if match_label == "Match":
                match_found = True

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 255), 1)

        if match_found:
            status_lines.append("Reference image found in camera")

        annotate_status(frame, status_lines)
        cv2.imshow("Live Face Match", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Detect faces in an image or match a reference face to a live camera feed."
    )
    parser.add_argument("--image", help="Path to the input image for face detection.")
    parser.add_argument("--reference", help="Reference image path for matching against live camera.")
    parser.add_argument("--camera", action="store_true", help="Use live camera for matching.")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index to use for live matching.")
    parser.add_argument("--match-threshold", type=float, default=0.25, help="Similarity threshold for face matching.")
    parser.add_argument("--min-good-matches", type=int, default=10, help="Minimum number of good ORB matches required to consider a match.")
    parser.add_argument("--output", default="detected_faces.jpg", help="Output image path for saved detection results.")
    args = parser.parse_args()

    if args.reference and args.camera:
        match_reference_image_in_camera(
            args.reference,
            camera_index=args.camera_index,
            match_threshold=args.match_threshold,
            min_good_matches=args.min_good_matches,
        )
        return

    if args.image:
        count, saved_path = detect_faces(args.image, args.output)
        print(f"Detected {count} face(s). Saved output to {saved_path}.")
        return

    if args.reference:
        count, saved_path = detect_faces(args.reference, args.output)
        print(f"Detected {count} face(s) in reference image. Saved output to {saved_path}.")
        return

    parser.error("Provide --image to detect faces from an image, or --reference --camera to match a reference image against live camera.")


if __name__ == "__main__":
    main()

