# Cheat Engine v1 — Development Harness

A standalone dev server to run and tune the v1 detection pipeline locally —
**no LiveKit, no Node, no whole codebase required**. It reuses the exact
production code paths (`process_frame` / `ProctorSession` from
`stream_server.py`, `ViolationTracker` + detection helpers from `main.py`), so
what you see on the dashboard is what the engine does in production.

It also mirrors the **AI-side messages exactly**, so you can see and tune the
alerts the interviewer will actually receive:

- **Trigger points** (12-occurrence threshold) → published as
  `{"type":"trigger", ...}` and injected as
  `[SYSTEM ALERT] Proctoring trigger point reached: ...`
- **Score-threshold injections** (10/30/50/70/90) → the agent's
  `[SYSTEM NOTICE]` / `[SYSTEM ALERT - HIGH]` messages.

---

## Prerequisites

- Python **3.11** (the pinned `mediapipe==0.10.21` / `dlib==20.0.1` only have
  wheels up to 3.11; they will NOT install on 3.12/3.13/3.14).
- The engine dependencies. Use the repo venv, or create one:

```bash
cd deps/interview-cheating-v1
python3.11 -m venv .venv
./.venv/bin/pip install -r requirements.txt
# if you hit "ModuleNotFoundError: No module named 'pkg_resources'" while
# importing face_recognition, also install setuptools:
./.venv/bin/pip install "setuptools==69.5.1"
```

> The repo venv is already set up on the dev/prod server. On a fresh machine,
> create one as above.

---

## Run

```bash
cd deps/interview-cheating-v1
./.venv/bin/python dev_server.py --port 6545 --source 0 --fps 5
```

Then open **http://localhost:6545** in your browser.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `6545` | Port for the dashboard |
| `--source` | `0` | Webcam index (`0`), a video file (loops), or an image |
| `--ref` | — | Reference face image for identity matching |
| `--fps` | `5` | Detection rate (frames/second) — lower = less CPU |
| `--threshold` | `3` | Violation occurrences to fire a trigger point (dev default 3 so alerts are quick to see; production uses 12) |
| `--no-auto` | off | Don't auto-start the feed (use the dashboard button) |

### Examples

```bash
# Webcam (recommended) — register your own face on the dashboard
./.venv/bin/python dev_server.py --source 0 --fps 5

# Video file, loops
./.venv/bin/python dev_server.py --source test_clip.mp4 --fps 5

# Static image with a reference for identity matching
./.venv/bin/python dev_server.py --source images/tarun_sample.jpeg --ref images/deepesh_sample.jpg

# Single static frame, no auto-start
./.venv/bin/python dev_server.py --source images/rahul_sample.png --no-auto
```

---

## Dashboard

| Panel | What it shows |
|-------|---------------|
| **Live Detection Feed** | The annotated frame — face box, iris/gaze dots, flags, score overlay (same as production Live Proctoring). |
| **Status cards** | Suspicion score, faces, face match, eye state, gaze, brightness. |
| **Flags** | Active detection flags (e.g. `no_blink`, `gaze_away`, `identity_mismatch`). |
| **Buffered Violations** | Per-violation count out of the trigger threshold (dev default 3; production 12). |
| **System Alerts** | The exact messages the AI receives — trigger points and score-threshold injections. |
| **Active Reference** | The image currently used as the reference face for identity matching. |
| **Snapshots** | Evidence thumbnails written to `.evidence/dev_session/` (auto-refresh; click to enlarge). |

### Controls (dashboard)

- **Set Reference** — opens the **Reference Capture page** (`/ref`) where you
  can:
  - **Capture** a photo from your camera (with **Retake** / **Use as reference**)
  - **Upload** an image file to test
  - Pick or **delete** images from the **Reference Gallery** (`references/`
    folder)
  - Clear the active reference
- **Reset session** — clears the violation tracker, score, and threshold state.
- **Start / Stop** — toggle the detection loop.

> Opening the reference page stops the detection feed first (releasing the
> webcam) so the capture camera can start without "device already in use".

### Reference images

Captured and uploaded reference images are stored in the repo's **`references/`**
folder (git-ignored). You can pick any stored image as the active reference for
identity matching, or delete it.

---

## What it reuses from production

- `process_frame()` — the full per-frame detection pipeline
- `ProctorSession` — per-session state + `ViolationTracker` (CSV + trigger points)
- `load_reference_encoding()` — `face_recognition` identity encoding
- `_json_safe()` — numpy-scalar-safe serialization

So any tuning you do (thresholds, EAR values, confidence, tolerances) applies
to the real engine too — they live in `stream_server.py` / `main.py`.

---

## Example output (live)

Detection status streaming over the WebSocket:

```json
{
  "face_count": 1,
  "face_match": true,
  "match_distance": 0.58,
  "flags": ["no_blink"],
  "suspicion_score": 2.0,
  "buffered": [{"name": "No Blink (>8s)", "count": 1}]
}
```

AI-side alert shown on the dashboard (and printed to the server console):

```
[SYSTEM INFO] Proctoring active. Minor flag detected: no_blink. No action needed yet.
[SYSTEM ALERT - HIGH] Suspicion score: 70/100. Repeated violations: no_blink. ...
[SYSTEM ALERT] Proctoring trigger point reached: No Blink (>8s). AI message to relay to candidate: ...
```

---

## Notes / troubleshooting

- **No camera found** (`OpenCV: camera failed to initialize`): run with a video
  or image source instead, or grant camera permission / use a real browser
  session on a machine with a camera.
- **`No module named 'pkg_resources'`**: install `setuptools==69.5.1` (see
  Prerequisites).
- **face_recognition warning on import**: only affects `--ref` identity
  matching; the core detection (face/iris/gaze/blink/light) still works.
- Evidence/trigger snapshots are written under `.evidence/dev_session/`
  (git-ignored).
- Reference images are stored under `references/` (git-ignored).
- **"device already in use" when capturing a reference**: the dashboard stops
  the webcam feed before opening the reference page, so this should not happen.
  If it does, stop the feed (`Stop`) and reload the reference page.
