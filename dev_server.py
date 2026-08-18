#!/usr/bin/env python3
"""
Cheat Engine v1 — standalone development harness.

Run the v1 detection pipeline against a LOCAL camera / video file / image —
no LiveKit, no Node, no whole codebase — and watch everything live on a
dashboard so detection can be tuned before deploying.

It reuses the exact production code paths (`process_frame` / `ProctorSession`
from stream_server.py, `ViolationTracker` + detection helpers from main.py), so
what you see here is what the engine does in production.

It also mirrors the AI-side messages exactly:
  - `trigger` (12-occurrence trigger point)  -> published to the agent's
    data channel as {"type":"trigger", ...} and injected as
    "[SYSTEM ALERT] Proctoring trigger point reached: ..."
  - `cheating_status` score thresholds (10/30/50/70/90) -> the agent's
    _handle_cheating_status injections ("[SYSTEM NOTICE] ...", etc.)

Usage:
    python dev_server.py --port 6545 [--source 0] [--ref reference.jpg] [--fps 5]
        --source  0              webcam index (default)
                 path/video.mp4  video file (loops)
                 path/image.jpg  static image (single frame)
        --ref     path/to.jpg    reference face for identity matching
        --fps     5              detection rate (frames/second)
        --no-auto                don't auto-start the feed (use dashboard button)

Then open http://localhost:6545
"""

import argparse
import asyncio
import base64
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from aiohttp import web, WSMsgType

import mediapipe as mp

from stream_server import (
    ProctorSession,
    process_frame,
    _json_safe,
    EVIDENCE_DIR,
    load_reference_encoding,
)

DEV_SESSION = "dev_session"

# ── Dashboard (embedded single-file HTML) ────────────────────────────────────
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Cheat Engine v1 — Dev Dashboard</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
         background:#0f1115; color:#e6e8ec; }
  header { display:flex; align-items:center; gap:12px; padding:14px 20px; border-bottom:1px solid #2a2e37;
           background:#14171d; position:sticky; top:0; z-index:10; }
  header h1 { font-size:16px; margin:0; font-weight:600; }
  header .tag { font-size:11px; color:#8ab4f8; background:#1b2733; border:1px solid #274a68;
                padding:2px 8px; border-radius:999px; }
  .wrap { display:grid; grid-template-columns: 1.2fr 1fr; gap:16px; padding:16px 20px; max-width:1500px; margin:0 auto; }
  .card { background:#161a21; border:1px solid #2a2e37; border-radius:12px; padding:16px; }
  .card h2 { font-size:13px; margin:0 0 10px; color:#9aa0a6; font-weight:600; text-transform:uppercase; letter-spacing:.05em; }
  #feed { width:100%; aspect-ratio:16/9; background:#000; border-radius:8px; object-fit:contain; border:1px solid #2a2e37; }
  .stats { display:grid; grid-template-columns: repeat(3,1fr); gap:8px; margin-top:12px; }
  .stat { background:#1d2129; border:1px solid #2a2e37; border-radius:8px; padding:8px 10px; }
  .stat .k { font-size:10px; color:#9aa0a6; text-transform:uppercase; letter-spacing:.04em; }
  .stat .v { font-size:16px; font-weight:600; margin-top:2px; }
  .flags { display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; min-height:22px; }
  .flag { font-size:10px; background:#4a1d1d; color:#ff8a8a; border:1px solid #7a2b2b; padding:2px 8px; border-radius:999px; }
  .buffered { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
  .buf { font-size:10px; background:#1e2a20; color:#9be69b; border:1px solid #2c4a33; padding:2px 8px; border-radius:999px; }
  .alerts { max-height:280px; overflow-y:auto; display:flex; flex-direction:column; gap:6px; }
  .alert { font-size:11px; line-height:1.45; padding:8px 10px; border-radius:8px; border:1px solid #5a4517;
           background:#2a210f; color:#ffd27a; white-space:pre-wrap; }
  .alert .t { font-size:9px; color:#9aa0a6; display:block; margin-bottom:3px; font-family:ui-monospace,monospace; }
  .snaps { display:grid; grid-template-columns: repeat(auto-fill,minmax(120px,1fr)); gap:8px; max-height:280px; overflow-y:auto; }
  .snap { position:relative; border-radius:8px; overflow:hidden; border:1px solid #2a2e37; cursor:pointer; }
  .snap img { width:100%; display:block; aspect-ratio:4/3; object-fit:cover; }
  .snap span { position:absolute; bottom:0; left:0; right:0; font-size:9px; background:rgba(0,0,0,.7);
               padding:2px 5px; color:#ffd27a; }
  .bar { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:14px; }
  button { background:#1b2733; color:#8ab4f8; border:1px solid #274a68; padding:7px 14px; border-radius:8px;
           font-size:12px; cursor:pointer; }
  button:hover { background:#22354a; }
  button:disabled { opacity:.5; cursor:default; }
  #lightbox { display:none; position:fixed; inset:0; background:rgba(0,0,0,.85); align-items:center; justify-content:center; z-index:50; }
  #lightbox img { max-width:92vw; max-height:92vh; border-radius:8px; }
  .pulse { animation:pulse 1.2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }
  .score-ok { color:#9be69b; } .score-warn { color:#ffd27a; } .score-bad { color:#ff8a8a; }
  .src { font-size:11px; color:#9aa0a6; font-family:ui-monospace,monospace; }
</style>
</head>
<body>
<header>
  <h1>🛡️ Cheat Engine v1 — Dev Dashboard</h1>
  <span class="tag">development harness</span>
  <span class="src" id="srcLabel"></span>
  <div style="flex:1"></div>
  <span id="connBadge" style="font-size:11px;color:#9aa0a6">connecting…</span>
</header>

<div class="wrap">
  <!-- Left: live feed -->
  <div class="card">
    <h2>Live Detection Feed</h2>
    <img id="feed" alt="live annotated feed"/>
    <div class="bar" style="margin-top:12px">
      <button id="registerBtn">📸 Register current frame as reference</button>
      <button id="resetBtn">↺ Reset session</button>
      <span style="flex:1"></span>
      <button id="startBtn" disabled>Start</button>
      <button id="stopBtn" disabled>Stop</button>
    </div>
    <div class="stats">
      <div class="stat"><div class="k">Suspicion</div><div class="v" id="score">0/100</div></div>
      <div class="stat"><div class="k">Faces</div><div class="v" id="faces">0</div></div>
      <div class="stat"><div class="k">Face Match</div><div class="v" id="match">—</div></div>
      <div class="stat"><div class="k">Eye State</div><div class="v" id="eye">N/A</div></div>
      <div class="stat"><div class="k">Gaze</div><div class="v" id="gaze">N/A</div></div>
      <div class="stat"><div class="k">Brightness</div><div class="v" id="bright">—</div></div>
    </div>
    <div class="flags" id="flags"></div>
    <h2 style="margin-top:14px">Buffered Violations (per 12)</h2>
    <div class="buffered" id="buffered"></div>
  </div>

  <!-- Right: alerts + snapshots -->
  <div class="card">
    <h2>System Alerts — exactly what the AI receives</h2>
    <div class="alerts" id="alerts">
      <div class="alert"><span class="t">system</span>Waiting for violations… (trigger points &amp; score-threshold injections appear here)</div>
    </div>

    <h2 style="margin-top:16px">Snapshots / Evidence</h2>
    <div class="snaps" id="snaps"></div>
  </div>
</div>

<div id="lightbox" onclick="this.style.display='none'"><img id="lightboxImg" alt=""/></div>

<script>
const ws = new WebSocket(`ws://${location.host}/ws`);
const $ = id => document.getElementById(id);
let status = {};

ws.onopen = () => { $('connBadge').textContent = 'connected'; $('connBadge').style.color = '#9be69b'; };
ws.onclose = () => { $('connBadge').textContent = 'disconnected'; $('connBadge').style.color = '#ff8a8a'; };
ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.type === 'frame') { $('feed').src = 'data:image/jpeg;base64,' + msg.data; }
  else if (msg.type === 'status') { renderStatus(msg.data); }
  else if (msg.type === 'alert') { addAlert(msg.data); }
  else if (msg.type === 'violation') { addAlert({line: msg.line, kind:'violation'}); }
  else if (msg.type === 'snapshots') { renderSnaps(msg.data); }
  else if (msg.type === 'ref') { addAlert({line: msg.message, kind: msg.ok ? 'ok' : 'err'}); }
  else if (msg.type === 'run') {
    $('startBtn').disabled = !msg.running; $('stopBtn').disabled = msg.running;
    if (msg.source) $('srcLabel').textContent = 'source: ' + msg.source;
  }
};

function renderStatus(s) {
  status = s;
  const sc = s.suspicion_score || 0;
  const el = $('score'); el.textContent = sc + '/100';
  el.className = 'v ' + (sc >= 70 ? 'score-bad' : sc >= 30 ? 'score-warn' : 'score-ok');
  $('faces').textContent = s.face_count ?? 0;
  $('match').textContent = s.face_match === null ? '—' : s.face_match ? '✓ Match' : '✗ Mismatch';
  $('eye').textContent = s.eye_state || 'N/A';
  $('gaze').textContent = s.gaze || 'N/A';
  $('bright').textContent = s.brightness != null ? s.brightness + '%' : '—';
  const flags = s.flags || [];
  $('flags').innerHTML = flags.length ? flags.map(f => `<span class="flag">${f}</span>`).join('') : '';
  const buf = s.buffered || [];
  $('buffered').innerHTML = buf.length ? buf.map(b => `<span class="buf">${b.name}: ${b.count}/12</span>`).join('') : '<span style="font-size:10px;color:#9aa0a6">none</span>';
}

function addAlert(a) {
  const t = new Date().toLocaleTimeString();
  const kind = a.kind === 'violation' ? 'violation' : a.kind === 'err' ? 'err' : 'alert';
  const color = kind === 'violation' ? 'border-color:#2c4a33;background:#142018;color:#9be69b'
              : kind === 'err' ? 'border-color:#7a2b2b;background:#241414;color:#ff8a8a'
              : 'border-color:#5a4517;background:#2a210f;color:#ffd27a';
  const div = document.createElement('div');
  div.className = 'alert'; div.style.cssText = color;
  div.innerHTML = `<span class="t">${t} — ${kind.toUpperCase()}</span>${(a.line || a.text || '').replace(/</g,'&lt;')}`;
  const box = $('alerts');
  box.insertBefore(div, box.firstChild);
  while (box.children.length > 200) box.removeChild(box.lastChild);
}

function renderSnaps(list) {
  $('snaps').innerHTML = list.map(f =>
    `<div class="snap" onclick="view('${f.url}')"><img src="${f.url}" alt=""/><span>${f.name}</span></div>`
  ).join('');
}
function view(url) { $('lightboxImg').src = url; $('lightbox').style.display = 'flex'; }

function send(m) { if (ws.readyState === 1) ws.send(JSON.stringify(m)); }
$('registerBtn').onclick = () => send({type:'register'});
$('resetBtn').onclick = () => send({type:'reset'});
$('startBtn').onclick = () => send({type:'start'});
$('stopBtn').onclick = () => send({type:'stop'});
setInterval(() => send({type:'snapshots'}), 5000);
</script>
</body>
</html>
"""


# ── Detection harness ────────────────────────────────────────────────────────
class DevHarness:
    def __init__(self, source, ref_path=None, fps=5, auto=True):
        self.source = source
        self.fps = max(1, fps)
        self.auto = auto
        self.running = False
        self.clients = set()
        self.session = ProctorSession(DEV_SESSION)
        self.latest_frame = None
        self._cap = None
        self._static_image = None
        self._face_det = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True, min_detection_confidence=0.5
        )
        self._thresholds = {10: False, 30: False, 50: False, 70: False, 90: False}
        self.alert_log = []

        if ref_path:
            enc = load_reference_encoding(Path(ref_path))
            if enc is not None:
                self.session.reference_encoding = enc
                print(f"[DEV] Reference face loaded from {ref_path}")
            else:
                print(f"[DEV] WARNING: could not load reference from {ref_path}")

    # ── source helpers ─────────────────────────────────────────────────────
    def _is_digit(self):
        return isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit())

    def _is_image(self):
        if self._is_digit():
            return False
        return str(self.source).lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))

    def open_source(self):
        if self._is_image():
            img = cv2.imread(str(self.source))
            if img is None:
                return False
            self._static_image = img
            return True
        idx = int(self.source) if self._is_digit() else self.source
        self._cap = cv2.VideoCapture(idx)
        return bool(self._cap.isOpened())

    def read_frame(self):
        if self._static_image is not None:
            return self._static_image.copy()
        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        return frame if ok else None

    # ── streaming ──────────────────────────────────────────────────────────
    async def broadcast(self, msg):
        if not self.clients:
            return
        data = json.dumps(msg, default=str)
        for ws in list(self.clients):
            try:
                await ws.send_str(data)
            except Exception:
                self.clients.discard(ws)

    async def detection_loop(self):
        while True:
            if not self.running:
                await asyncio.sleep(0.2)
                continue
            frame = await asyncio.to_thread(self.read_frame)
            if frame is None:
                await asyncio.sleep(0.2)
                continue
            try:
                result, trigger = await asyncio.to_thread(
                    process_frame, frame, self.session, self._face_det, self._face_mesh
                )
            except Exception as e:
                print(f"[DEV] process_frame error: {e}")
                await asyncio.sleep(0.2)
                continue

            ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ok:
                self.latest_frame = enc.tobytes()
                await self.broadcast({"type": "frame", "data": base64.b64encode(enc.tobytes()).decode()})

            status = _json_safe(result)
            status["buffered"] = [
                {"name": k, "count": min(len(v["timestamps"]), 12)}
                for k, v in self.session.tracker.violations.items()
            ]
            await self.broadcast({"type": "status", "data": status})

            if trigger:
                await self._emit_trigger(trigger)
            await self._emit_threshold_alerts(status)

            await asyncio.sleep(1.0 / self.fps)

    def _agent_injection(self, status):
        """Mirror the agent's _handle_cheating_status injections."""
        score = status.get("suspicion_score", 0) or 0
        flags = ", ".join(status.get("flags", [])) or "none"
        if score >= 90 and not self._thresholds[90]:
            self._thresholds[90] = True
            return ("[SYSTEM ALERT - CRITICAL] Suspicion score: 90/100. Multiple severe violations "
                    f"detected ({flags}). The candidate appears to be cheating extensively. End the "
                    "interview immediately. Thank them for their time and inform them the session is "
                    "being terminated due to integrity concerns. Do not ask more questions.", True)
        if score >= 70 and not self._thresholds[70]:
            self._thresholds[70] = True
            return ("[SYSTEM ALERT - HIGH] Suspicion score: 70/100. Repeated violations: "
                    f"{flags}. Firmly warn the candidate that continued suspicious behavior will "
                    "result in interview termination.", True)
        if score >= 50 and not self._thresholds[50]:
            self._thresholds[50] = True
            return ("[SYSTEM NOTICE] Suspicion score: 50/100. Issues detected: "
                    f"{flags}. Politely remind the candidate to maintain eye contact with the "
                    "camera and ensure no one else is assisting them.", True)
        if score >= 30 and not self._thresholds[30]:
            self._thresholds[30] = True
            return ("[SYSTEM NOTICE] Suspicion score: 30/100. Minor issues: "
                    f"{flags}. Gently ask the candidate if everything is okay and if they can "
                    "see/hear you clearly.", True)
        if score >= 10 and not self._thresholds[10]:
            self._thresholds[10] = True
            return ("[SYSTEM INFO] Proctoring active. Minor flag detected: "
                    f"{flags}. No action needed yet.", False)
        return None

    async def _emit_trigger(self, trigger):
        line = (
            f"[SYSTEM ALERT] Proctoring trigger point reached: {trigger.get('violation')}. "
            f"AI message to relay to candidate: {trigger.get('aiWarning')}. "
            f"Current suspicion score: {trigger.get('suspicion_score')}/100. "
            "Address this naturally within the conversation."
        )
        print(f"[DEV][ALERT] {line}")
        await self.broadcast({"type": "alert", "text": line})
        await self.broadcast({"type": "alert", "text": "raw trigger: " + json.dumps(trigger, default=str), "kind": "violation"})

    async def _emit_threshold_alerts(self, status):
        inj = self._agent_injection(status)
        if inj:
            text, _trigger_reply = inj
            print(f"[DEV][AGENT-INJECTION] {text}")
            await self.broadcast({"type": "alert", "text": text, "kind": "violation"})

    # ── actions ────────────────────────────────────────────────────────────
    def register_current_frame(self):
        frame = self.read_frame()
        if frame is None:
            return False, "No frame available"
        ref_dir = os.path.join(EVIDENCE_DIR, DEV_SESSION)
        os.makedirs(ref_dir, exist_ok=True)
        ref_path = os.path.join(ref_dir, "reference.jpg")
        cv2.imwrite(ref_path, frame)
        enc = load_reference_encoding(Path(ref_path))
        if enc is None:
            return False, "No/ambiguous face in reference frame — retry with a clear face"
        self.session.reference_encoding = enc
        return True, f"Reference face registered ({len(enc)} dims)"

    def reset(self):
        self.session = ProctorSession(DEV_SESSION)
        self._thresholds = {10: False, 30: False, 50: False, 70: False, 90: False}
        if self.alert_log:
            self.alert_log.clear()
        return True, "Session reset"

    def list_snapshots(self):
        d = os.path.join(EVIDENCE_DIR, DEV_SESSION)
        if not os.path.isdir(d):
            return []
        files = sorted(f for f in os.listdir(d) if f.endswith(".jpg") and f != "reference.jpg")
        return [{"name": f, "url": f"/snapshot/{f}"} for f in files]


# ── HTTP / WebSocket ─────────────────────────────────────────────────────────
def make_app(harness):
    app = web.Application()

    async def _start_detection(app_):
        app_["detection_task"] = asyncio.create_task(harness.detection_loop())

    async def _stop_detection(app_):
        harness.running = False
        task = app_.get("detection_task")
        if task:
            task.cancel()

    app.on_startup.append(_start_detection)
    app.on_cleanup.append(_stop_detection)

    async def index(request):
        return web.Response(text=DASHBOARD_HTML, content_type="text/html")

    async def ws_handler(request):
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        harness.clients.add(ws)
        await harness.broadcast({"type": "run", "running": harness.running, "source": str(harness.source)})
        await ws.send_str(json.dumps({"type": "snapshots", "data": harness.list_snapshots()}))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except Exception:
                        continue
                    mtype = data.get("type")
                    if mtype == "start":
                        harness.running = True
                        await harness.broadcast({"type": "run", "running": True})
                    elif mtype == "stop":
                        harness.running = False
                        await harness.broadcast({"type": "run", "running": False})
                    elif mtype == "register":
                        ok, message = await asyncio.to_thread(harness.register_current_frame)
                        await ws.send_str(json.dumps({"type": "ref", "ok": ok, "message": message}))
                    elif mtype == "reset":
                        ok, message = harness.reset()
                        await ws.send_str(json.dumps({"type": "ref", "ok": ok, "message": message}))
                    elif mtype == "snapshots":
                        await ws.send_str(json.dumps({"type": "snapshots", "data": harness.list_snapshots()}))
        finally:
            harness.clients.discard(ws)
        return ws

    async def snapshot(request):
        filename = request.match_info["filename"]
        if "/" in filename or ".." in filename:
            return web.Response(status=400)
        path = os.path.join(EVIDENCE_DIR, DEV_SESSION, filename)
        if not os.path.exists(path):
            return web.Response(status=404)
        return web.FileResponse(path)

    async def status(request):
        return web.json_response({"running": harness.running, "source": str(harness.source)})

    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/snapshot/{filename}", snapshot)
    app.router.add_get("/api/status", status)
    return app


def main():
    parser = argparse.ArgumentParser(description="Cheat Engine v1 dev harness")
    parser.add_argument("--port", type=int, default=6545)
    parser.add_argument("--source", default="0", help="webcam index, video file, or image (default 0)")
    parser.add_argument("--ref", default=None, help="reference face image for identity matching")
    parser.add_argument("--fps", type=int, default=5, help="detection frames/second (default 5)")
    parser.add_argument("--no-auto", action="store_true", help="don't auto-start the feed")
    args = parser.parse_args()

    harness = DevHarness(args.source, ref_path=args.ref, fps=args.fps, auto=not args.no_auto)
    harness.running = harness.auto

    if not harness.open_source():
        print(f"[DEV] ERROR: could not open source '{args.source}'")
        return

    app = make_app(harness)

    print("=" * 60)
    print("  Cheat Engine v1 — Development Dashboard")
    print(f"  Source : {args.source}")
    print(f"  FPS    : {args.fps}")
    print(f"  URL    : http://localhost:{args.port}")
    print("  Reference: " + (args.ref or "register from the dashboard button"))
    print("  Alerts shown = the exact messages the AI receives in production")
    print("=" * 60)

    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
