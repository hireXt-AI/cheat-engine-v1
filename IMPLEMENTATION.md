# Cheat Engine v1 — LiveKit Integration Implementation Plan

This document captures the integration plan for embedding **cheat-engine-v1** into the
HireXt interview flow. It covers architecture, the chosen transport, the work items,
challenges, and likely edge cases. It is a living plan — update it as decisions change.

---

## 1. Goal

Give the AI interviewer live proctoring during the interview:

- v1 detects violations (face missing, identity mismatch, eye closure, no-blink, iris/gaze,
  glasses, multiple people, lighting) using MediaPipe + face_recognition.
- When a violation reaches a **trigger threshold**, v1 emits a **trigger point** (CSV row)
  and sends an **AI alert** to the interview LLM so the AI can respond naturally.
- Must run **without disturbing the AI interview flow** and **without wasting bandwidth**.

## 2. Key architecture decisions (agreed)

| Decision | Choice | Why |
|---|---|---|
| Where v1 runs | **Node server** (PM2), same host as the existing v0 engine | Matches v0 ops; keeps Jarvis VM focused on the interview; no GPU pressure on the engine host needed for inference is low. |
| How v1 gets the stream | **LiveKit proctor** — v1 joins the interview room, subscribes to the candidate's video+audio directly from the SFU | No browser re-upload → **zero extra upload bandwidth**; the stream already flows once to the SFU. |
| Reference image | **Browser captures a selfie** at interview start, uploaded via `POST /register-face` | Reuses v0's proven flow; lets each candidate register fresh. |
| Trigger → AI alert | v1 publishes `cheating_status`/`trigger` over the **room data channel**; the Jarvis agent's `_inject_system_context` pushes the **AI Message** into the LLM | No extra network hop; agent already receives room data. |
| Version switch | `CHEATING_ENGINE_VERSION` / enabled flag selects v0 vs v1 | Lets ops roll back to v0 instantly. |

## 3. Files & responsibilities

- `deps/interview-cheating-v1/main.py` — **v1 core detection, kept as-is**: `ViolationTracker`
  (CSV writer), `calculate_ear`, `analyze_lighting_conditions`, `detect_black_glasses`,
  `detect_iris_direction`, `load_reference_encoding`, mediapipe/face_recognition helpers,
  `TRIGGER_CSV_HEADER`.
- `deps/interview-cheating-v1/stream_server.py` — **NEW** server (replaces the current broken
  one). Uses v0's LiveKitProctor transport + drives v1's detection per frame. Publishes
  `cheating_status`/`trigger` over the room data channel.
- `deps/interview-cheating-v1/runner.sh` — PM2 launcher (already `cheating-engine`, port 6544).
- `deps/interview-ai/main.py` — Jarvis agent. Receives `cheating_status`/`trigger` over the
  room data channel and injects the AI Message into the LLM via `_inject_system_context`
  (already exists for the LiveKit-proctor path).

## 4. Transport design (mirrors v0's LiveKitProctor)

- `POST /api/proctor/start` body `{ sessionId, livekitUrl, token }` → join the room as a
  proctor, `add_subscription(candidate_identity, KIND_VIDEO/AUDIO)`.
- On `track_subscribed` for the candidate's video → decode frame → run v1 detection → drive
  `ViolationTracker.log_violation(...)`.
- On `log_violation` returning `True` (threshold reached) → write CSV (v1 core) + publish
  `{"type":"cheating_status"|"trigger", ...}` with the **AI Message** over the data channel.
- `POST /register-face` (multipart `image`+`sessionId`) → browser selfie → v1 face encoding
  via `load_reference_encoding` + save evidence.
- Keep the same REST/evidence endpoints v0 exposed so the admin UI keeps working
  (`/report/:id`, `/evidence`, `/live-frame/:id`).

## 5. Detection pipeline per frame (from v1 core)

1. MediaPipe `FaceDetection` → face count + box.
2. `face_count == 0` → "Missing Candidate" (buffer; trigger on threshold).
3. `face_count > 1` → "Multiple People Detected".
4. face present → `face_recognition.face_distance` vs reference → "Identity Mismatch".
5. MediaPipe `FaceMesh` → `calculate_ear` → eye closure / no-blink.
6. `detect_black_glasses` → glasses flag.
7. `detect_iris_direction` + EAR threshold → gaze/iris streaks.
8. `analyze_lighting_conditions` → low/high light.
9. Each violation → `ViolationTracker.log_violation(type, ts, hr_text, ai_warning, frame)`.

## 6. Work items

- [ ] Rewrite `stream_server.py` (transport + per-frame v1 detection + CSV + data-channel publish).
- [ ] Confirm `main.py` import is safe (no module-level camera/mic side effects) — verified safe.
- [ ] Wire `runner.sh` to start the new server (already points at `stream_server.py`).
- [ ] Jarvis agent: ensure `cheating_status`/`trigger` over data channel maps to
      `_inject_system_context` (already wired for `cheating_status`).
- [ ] Add `CHEATING_ENGINE_VERSION` switch / enable flag in `.env` + docs.
- [ ] Deploy v1 to Node server + install v1 deps (mediapipe, face_recognition, dlib, opencv).
- [ ] Test end-to-end with a real LiveKit room.

---

## 7. Challenges

### 7.1 Dependencies are heavy (mediapipe, dlib, face_recognition)
- `dlib` needs a C++ build / prebuilt wheels; `face_recognition` pulls `dlib` + model files.
- `mediapipe==0.10.21` is pinned; older/newer OpenCV can conflict.
- **Mitigation**: install in the v1 `.venv`; keep the pinned `requirements.txt`; verify
  `face_recognition` model files (`dlib` 68-landmark) download correctly on the Node server.

### 7.2 Frame rate / CPU on the Node server
- LiveKit frames arrive at ~20–30fps; running MediaPipe face + face_mesh + face_recognition on
  every frame is CPU-heavy.
- **Mitigation**: process at a throttled rate (e.g. every N ms / skip frames), drop frames when
  the previous one is still processing (v0 already uses a `_processing` flag), run heavy work in
  `asyncio.to_thread`.

### 7.3 The v1 codebase is messy / not server-ready
- v1's `main.py` mixes recording + detection; `stream_server.py` is currently broken (imports
  functions that don't exist in `main.py`).
- **Mitigation**: reuse only the clean pieces (`ViolationTracker`, detection helpers); drive them
  from our own server. Keep `main.py` untouched (submodule → changes push to the original repo).

### 7.4 Identity matching needs a good reference
- v1's `load_reference_encoding` requires a single clear face. Browser selfies vary in quality.
- **Mitigation**: reject selfies with no/ambiguous face in `register-face` (mirror v0),
  return a clear error so the UI can re-prompt.

### 7.5 False positives / liveliness
- No-blink and gaze heuristics are sensitive; lighting changes can cause false triggers.
- **Mitigation**: keep the 12-occurrence threshold buffering (v1's `ViolationTracker` already
  buffers), add cooldowns, and make the AI alert advisory (the agent decides how to react).

## 8. Likely edge cases

1. **Candidate joins before register-face** — v1 must tolerate no reference yet (skip identity
   check, still run presence/light checks).
2. **Agent voice track** — v1 must not treat the AI's own audio as the candidate; track identity
   filter (v0 uses `participant.identity != candidate_identity`).
3. **Candidate rejoin / identity change** — participant identity is `candidate_{sessionId}`;
   a rejoin keeps the same session. Watch duplicate proctors (guard with `active_proctors`).
4. **Silence during AI talking** — don't flag candidate silence while the agent is speaking
   (v0 tracks `agent_speaking` from the agent audio track).
5. **Camera off / track muted** — `track_subscribed` may fire with no frames, or the track
   disappears; handle missing frames gracefully, don't crash the proctor.
6. **Multi-room concurrency** — many interviews → many proctors; ensure per-session state is
   isolated and cleanup on disconnect.
7. **LiveKit token expiry** — long interviews; the proctor token may expire. Decide on refresh
   or accept proctor drop + reconnect.
8. **CSV/evidence disk growth** — trigger snapshots + recordings accumulate on the Node server;
   add retention/cleanup.
9. **`_inject_system_context` cooldown** — the agent already cooldowns LLM injections (30s) and
   respects `cheatingLlmEnabled`; don't spam the LLM with every trigger.
10. **Deprecated import of `main` in the old `stream_server.py`** — the current file fails at
    import; the rewrite removes it. Don't rely on the old file.
11. **`face_recognition` model download at first use** — may fail behind firewalls; pre-download
    or verify connectivity on the Node server.

## 9. Rollback / safety

- Keep v0's engine running; the `CHEATING_ENGINE_VERSION`/enable switch determines which is used.
- v1 must **fail soft**: if v1 deps are missing or a frame can't be processed, the interview
  must continue normally (no exception bubbling into the agent).

## 10. Open questions (to confirm before/while building)

- Should v1 also record the raw video/audio (v1's `start_interview_recording`) for later review,
  or only run live detection? (Recording adds disk + CPU; decide per requirement.)
- Exact threshold/cooldown values for each violation.
- Whether the AI should react to *every* trigger or only high-severity ones.
- Who owns the LiveKit token minting for the proctor (Node server vs Jarvis agent).
