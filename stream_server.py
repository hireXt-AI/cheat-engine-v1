import argparse
import json
import os
import time
import numpy as np
import cv2
import socketio
from aiohttp import web
import asyncio

from face_detection import (
    load_face_cascade,
    detect_eyes,
    classify_eye_state,
    estimate_gaze,
    get_light_status,
)

# ── Server setup ─────────────────────────────────────────────────────────────
sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')
app = web.Application()
sio.attach(app)

async def index(request):
    return web.FileResponse(os.path.join(os.path.dirname(__file__), 'test.html'))

app.router.add_get('/', index)

EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), '.evidence')
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# ── Engine globals ───────────────────────────────────────────────────────────
face_cascade = None
MATCH_TOLERANCE = 0.55  # combined score threshold (higher = stricter)

# Per-session state: sessionId -> { ref_hist, ref_des, gaze_away_start, ... }
sessions = {}  # keyed by sessionId (from client), not socket sid
sid_to_session_id = {}  # socket sid -> sessionId mapping
latest_frames = {}  # sessionId -> jpeg_bytes (most recent frame for live monitoring)


def compute_face_signature(face_gray):
    """Compute histogram + ORB descriptors for a face region."""
    face_resized = cv2.resize(face_gray, (150, 150))
    hist = cv2.calcHist([face_resized], [0], None, [64], [0, 256])
    cv2.normalize(hist, hist)
    orb = cv2.ORB_create(300)
    _, des = orb.detectAndCompute(face_resized, None)
    return hist, des


def compare_face_signatures(ref_hist, ref_des, cand_hist, cand_des):
    """Returns (combined_score, distance). Higher score = more similar."""
    hist_score = cv2.compareHist(ref_hist, cand_hist, cv2.HISTCMP_CORREL)
    orb_score = 0.0
    if ref_des is not None and cand_des is not None and len(ref_des) > 0 and len(cand_des) > 0:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(ref_des, cand_des)
        good = [m for m in matches if m.distance < 60]
        orb_score = len(good) / max(1, min(len(ref_des), len(cand_des)))
    combined = 0.6 * hist_score + 0.4 * min(orb_score * 4, 1.0)
    return combined, round(1.0 - combined, 3)


def new_session(session_id, ref_hist=None, ref_des=None):
    return {
        "session_id": session_id,
        "ref_hist": ref_hist,
        "ref_des": ref_des,
        "gaze_away_start": None,
        "face_gone_start": None,
        "suspicion_score": 0.0,
        "events": [],
        "last_screenshot": 0,
    }


def add_event(session, event_type, detail=""):
    session["events"].append({
        "time": round(time.time(), 2),
        "type": event_type,
        "detail": detail,
    })
    if len(session["events"]) > 200:
        session["events"] = session["events"][-200:]


# ── REST: Register face ──────────────────────────────────────────────────────
async def register_face(request):
    """POST /register-face  body: multipart with 'image' file and 'sessionId' field."""
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

    # Decode and compute face signature
    arr = np.frombuffer(image_data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return web.json_response({"error": "Invalid image"}, status=400)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return web.json_response({"error": "No face detected in selfie. Ensure face is clearly visible."}, status=400)

    x, y, w, h = faces[0]
    face_gray = gray[y:y+h, x:x+w]
    ref_hist, ref_des = compute_face_signature(face_gray)

    # Save selfie as evidence
    session_dir = os.path.join(EVIDENCE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    cv2.imwrite(os.path.join(session_dir, "reference.jpg"), img)

    # Store session
    sessions[session_id] = new_session(session_id, ref_hist, ref_des)
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
        "suspicion_score": round(session["suspicion_score"], 1),
        "events": session["events"],
        "total_events": len(session["events"]),
    })


app.router.add_post('/register-face', register_face)
app.router.add_get('/report/{sessionId}', get_report)


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


app.router.add_get('/evidence', list_evidence_sessions)
app.router.add_get('/evidence/{sessionId}/{filename}', get_evidence_image)
app.router.add_get('/evidence/{sessionId}/mapping/{filename}', get_evidence_mapping)


async def get_live_frame(request):
    """GET /live-frame/{sessionId} — serve the latest video frame as JPEG."""
    session_id = request.match_info['sessionId']
    frame_data = latest_frames.get(session_id)
    if not frame_data:
        return web.json_response({"error": "No frame available"}, status=404)
    return web.Response(body=frame_data, content_type='image/jpeg')


app.router.add_get('/live-frame/{sessionId}', get_live_frame)


# ── Frame processing ─────────────────────────────────────────────────────────
def process_frame(jpeg_bytes, session):
    if session.get("ref_hist") is None:
        return None, None

    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, None

    now = time.time()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_h, img_w = frame.shape[:2]

    brightness, low_light, light_on = get_light_status(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    face_count = len(faces)

    result = {
        "person_in_frame": face_count > 0,
        "face_count": face_count,
        "multi_face": face_count > 1,
        "light_on": light_on,
        "low_light": low_light,
        "brightness": round(brightness),
        "face_match": False,
        "match_distance": None,
        "eye_state": "N/A",
        "gaze": "N/A",
        "sunglasses": False,
        "face_box": None,
        "eye_boxes": [],
        "gaze_away_duration": 0,
        "face_gone_duration": 0,
        "suspicion_score": session["suspicion_score"],
        "flags": [],
        "events": session["events"][-10:],
    }

    if face_count > 1:
        result["flags"].append("multiple_faces")
        session["suspicion_score"] += 5
        add_event(session, "multi_face", f"{face_count} faces detected")

    if face_count == 0:
        if session["face_gone_start"] is None:
            session["face_gone_start"] = now
        gone_dur = now - session["face_gone_start"]
        result["face_gone_duration"] = round(gone_dur, 1)
        if gone_dur > 5:
            result["flags"].append("face_gone_long")
            session["suspicion_score"] += 2
            add_event(session, "face_gone", f"{gone_dur:.1f}s")
    else:
        session["face_gone_start"] = None

    if face_count > 0:
        x, y, w, h = faces[0]
        face_gray_roi = frame_gray[y:y+h, x:x+w]

        # Face matching via histogram + ORB
        cand_hist, cand_des = compute_face_signature(face_gray_roi)
        score, distance = compare_face_signatures(session["ref_hist"], session["ref_des"], cand_hist, cand_des)
        matched = score >= MATCH_TOLERANCE
        result["face_match"] = matched
        result["match_distance"] = distance
        if not matched:
            result["flags"].append("identity_mismatch")
            session["suspicion_score"] += 3
            add_event(session, "identity_mismatch", f"dist={distance}")

        eyes, glass_eyes = detect_eyes(face_gray_roi)
        eye_state, sunglasses = classify_eye_state(eyes, glass_eyes)
        gaze_text, looking = estimate_gaze(face_gray_roi, eyes)

        result["eye_state"] = eye_state
        result["gaze"] = gaze_text
        result["sunglasses"] = sunglasses
        result["face_box"] = [x / img_w, y / img_h, w / img_w, h / img_h]
        result["eye_boxes"] = [
            [(x + ex) / img_w, (y + ey) / img_h, ew / img_w, eh / img_h]
            for (ex, ey, ew, eh) in eyes
        ]

        if not looking and gaze_text != "Gaze unknown":
            if session["gaze_away_start"] is None:
                session["gaze_away_start"] = now
            away_dur = now - session["gaze_away_start"]
            result["gaze_away_duration"] = round(away_dur, 1)
            if away_dur > 3:
                result["flags"].append("gaze_away_long")
                session["suspicion_score"] += 1
                add_event(session, "gaze_away", f"{away_dur:.1f}s")
        else:
            session["gaze_away_start"] = None

    session["suspicion_score"] = min(session["suspicion_score"], 100)
    result["suspicion_score"] = round(session["suspicion_score"], 1)

    # Screenshot on flags
    screenshot_path = None
    if result["flags"] and (now - session["last_screenshot"] > 5):
        session_dir = os.path.join(EVIDENCE_DIR, session["session_id"])
        os.makedirs(session_dir, exist_ok=True)
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"SID: {session['session_id']}", (10, img_h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, timestamp_str, (10, img_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        filename = f"{int(now)}_{'-'.join(result['flags'])}.jpg"
        screenshot_path = os.path.join(session_dir, filename)
        cv2.imwrite(screenshot_path, frame)
        session["last_screenshot"] = now
        add_event(session, "screenshot", filename)

        # Save mapping data alongside the image
        mapping = {
            "face_box": result.get("face_box"),
            "eye_boxes": result.get("eye_boxes", []),
            "gaze": result.get("gaze"),
            "eye_state": result.get("eye_state"),
            "face_match": result.get("face_match"),
            "match_distance": result.get("match_distance"),
            "face_count": result.get("face_count"),
            "flags": result.get("flags", []),
            "suspicion_score": result.get("suspicion_score"),
            "brightness": result.get("brightness"),
            "timestamp": timestamp_str,
        }
        mapping_path = os.path.join(session_dir, filename.replace('.jpg', '.json'))
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f)

    return result, screenshot_path


# ── Socket handlers ──────────────────────────────────────────────────────────
LOG_INTERVAL = 0.5
last_log_time = 0


@sio.on('video_frame', namespace='/interview')
async def on_video_frame(sid, data):
    global last_log_time
    now = time.time()
    if now - last_log_time < LOG_INTERVAL:
        return

    image_data = data.get('image') if isinstance(data, dict) else None
    session_id = data.get('sessionId') if isinstance(data, dict) else None
    if image_data is None or session_id is None:
        return

    jpeg_bytes = bytes(image_data)

    # Store latest frame for live monitoring
    latest_frames[session_id] = jpeg_bytes

    # Map socket sid to session and ensure session exists
    sid_to_session_id[sid] = session_id
    session = sessions.get(session_id)
    if not session:
        return  # Face not registered yet

    result, screenshot = await asyncio.to_thread(process_frame, jpeg_bytes, session)
    if result is None:
        return

    # Convert numpy types for JSON
    result = json.loads(json.dumps(result, default=lambda o: bool(o) if isinstance(o, (np.bool_,)) else int(o) if isinstance(o, (np.integer,)) else float(o) if isinstance(o, (np.floating,)) else o))

    last_log_time = now

    flags_str = ','.join(result['flags']) if result['flags'] else '-'
    print(
        f"[ENGINE] "
        f"Faces:{result['face_count']} | "
        f"Match:{'✓' if result['face_match'] else '✗'}"
        f"({result['match_distance'] or '-'}) | "
        f"Gaze:{result['gaze']} | "
        f"Eyes:{result['eye_state']} | "
        f"Score:{result['suspicion_score']} | "
        f"Flags:[{flags_str}]"
    )
    if screenshot:
        print(f"[SCREENSHOT] Saved: {screenshot}")

    await sio.emit('cheating_status', result, room=sid, namespace='/interview')


@sio.on('screen_recording', namespace='/interview')
async def on_screen_recording(sid, data):
    session_id = sid_to_session_id.get(sid) or data.get('sessionId')
    if not session_id:
        return
    session = sessions.get(session_id)
    if not session:
        return
    session["suspicion_score"] = min(session["suspicion_score"] + 10, 100)
    add_event(session, "screen_recording", data.get("detail", "detected"))
    print(f"[ALERT] Screen recording detected for {session_id}: {data}")


@sio.on('connect', namespace='/interview')
async def on_connect(sid, environ):
    print(f"[CONNECT] {sid}")


@sio.on('disconnect', namespace='/interview')
async def on_disconnect(sid):
    session_id = sid_to_session_id.pop(sid, None)
    if session_id:
        latest_frames.pop(session_id, None)
    session = sessions.get(session_id) if session_id else None
    if session:
        print(f"[DISCONNECT] {session_id} | Final score: {session['suspicion_score']} | Events: {len(session['events'])}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Cheating detection server")
    parser.add_argument("--port", type=int, default=6544)
    parser.add_argument("--tolerance", type=float, default=0.6, help="Face match tolerance (lower=stricter)")
    args = parser.parse_args()

    global face_cascade, MATCH_TOLERANCE
    MATCH_TOLERANCE = args.tolerance
    face_cascade = load_face_cascade()

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
    print(f"         GET  /report/{{sessionId}}")
    print(f"         WS   /interview  (video_frame event)")
    web.run_app(app, host='0.0.0.0', port=args.port, print=None)


if __name__ == "__main__":
    main()
