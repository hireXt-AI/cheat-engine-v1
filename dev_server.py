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
from xmlrpc import server

import cv2
import numpy as np
from aiohttp import web, WSMsgType
from s3_storage import upload_evidence

import mediapipe as mp

from stream_server import (
    ProctorSession,
    process_frame,
    _json_safe,
    EVIDENCE_DIR,
    load_reference_encoding,
)

DEV_SESSION = "dev_session"
REF_DIR = os.path.join(os.path.dirname(__file__), "references")
os.makedirs(REF_DIR, exist_ok=True)

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
      <button id="openRefBtn">📸 Set Reference</button>
      <button id="resetBtn">↺ Reset session</button>
      <span style="flex:1"></span>
      <button id="startBtn" disabled>Start</button>
      <button id="stopBtn" disabled>Stop</button>
    </div>
    <div id="refStatus" style="font-size:11px;color:#9aa0a6;margin-top:4px"></div>
    <!-- Active reference display -->
    <div id="refBox" style="display:none;margin-top:8px;display:none;align-items:center;gap:10px;background:#1d2129;border:1px solid #2a2e37;border-radius:8px;padding:8px 10px">
      <img id="refImg" style="width:52px;height:52px;object-fit:cover;border-radius:6px;border:1px solid #3a4a5a" alt="active reference"/>
      <div style="min-width:0">
        <div style="font-size:10px;color:#9aa0a6;text-transform:uppercase;letter-spacing:.04em">Active Reference</div>
        <div id="refName" style="font-size:11px;color:#9be69b;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">—</div>
      </div>
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
$('resetBtn').onclick = () => send({type:'reset'});
$('startBtn').onclick = () => send({type:'start'});
$('stopBtn').onclick = () => send({type:'stop'});
setInterval(() => send({type:'snapshots'}), 5000);

// "Set Reference" opens a dedicated capture PAGE. The detection feed holds the
// webcam, so stop it (releasing the camera) before navigating — otherwise the
// capture page's getUserMedia hits "device already in use". Clear the feed/logs
// too so the page starts fresh.
$('openRefBtn').onclick = () => {
  send({ type: 'stop' });               // release the webcam from the feed loop
  send({ type: 'clear' });              // clear logs + score + alerts + snaps
  location.href = '/ref';
};

ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.type === 'frame') { $('feed').src = 'data:image/jpeg;base64,' + msg.data; }
  else if (msg.type === 'status') { renderStatus(msg.data); }
  else if (msg.type === 'alert') { addAlert(msg); }
  else if (msg.type === 'violation') { addAlert({line: msg.line, kind:'violation'}); }
  else if (msg.type === 'snapshots') { renderSnaps(msg.data); }
  else if (msg.type === 'clear') {
    $('alerts').innerHTML = '<div class="alert"><span class="t">system</span>Waiting for violations…</div>';
    $('score').textContent = '0/100'; $('faces').textContent = '0';
    $('match').textContent = '—'; $('eye').textContent = 'N/A'; $('gaze').textContent = 'N/A';
    $('bright').textContent = '—'; $('flags').innerHTML = ''; $('buffered').innerHTML = '';
    $('snaps').innerHTML = ''; $('refStatus').textContent = '';
  }
  else if (msg.type === 'ref') {
    $('refStatus').textContent = '✓ ' + msg.message;
    $('refStatus').style.color = '#9be69b';
  }
  else if (msg.type === 'ref_active') {
    if (msg.url) {
      $('refImg').src = msg.url; $('refName').textContent = msg.name;
      $('refBox').style.display = 'flex';
    } else {
      $('refBox').style.display = 'none';
    }
  }
  else if (msg.type === 'run') {
    $('startBtn').disabled = !msg.running; $('stopBtn').disabled = msg.running;
    if (msg.source) $('srcLabel').textContent = 'source: ' + msg.source;
    if (msg.ref) { $('refStatus').textContent = 'Reference: ' + msg.ref; $('refStatus').style.color = '#9be69b'; }
  }
};
</script>
</body>
</html>
"""


# ── Reference capture page (standalone) ──────────────────────────────────────
REF_PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Cheat Engine v1 — Reference Capture</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
         background:#0f1115; color:#e6e8ec; }
  header { display:flex; align-items:center; gap:12px; padding:14px 20px; border-bottom:1px solid #2a2e37;
           background:#14171d; position:sticky; top:0; z-index:10; }
  header h1 { font-size:16px; margin:0; font-weight:600; }
  .wrap { display:grid; grid-template-columns: 1.2fr 1fr; gap:16px; padding:16px 20px; max-width:1400px; margin:0 auto; }
  .card { background:#161a21; border:1px solid #2a2e37; border-radius:12px; padding:16px; }
  .card h2 { font-size:13px; margin:0 0 10px; color:#9aa0a6; font-weight:600; text-transform:uppercase; letter-spacing:.05em; }
  .vidwrap { position:relative; background:#000; border-radius:8px; overflow:hidden; aspect-ratio:4/3; }
  .vidwrap video, .vidwrap img { width:100%; height:100%; object-fit:cover; transform:scaleX(-1); }
  .bar { display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }
  button { background:#1b2733; color:#8ab4f8; border:1px solid #274a68; padding:7px 14px; border-radius:8px;
           font-size:12px; cursor:pointer; }
  button:hover { background:#22354a; }
  button:disabled { opacity:.5; cursor:default; }
  .err { display:none; font-size:11px; color:#ff8a8a; margin-top:8px; }
  .msg { font-size:12px; color:#9be69b; margin-top:8px; min-height:16px; }
  .upl { border:1px dashed #274a68; border-radius:8px; padding:16px; text-align:center; color:#9aa0a6; font-size:12px; cursor:pointer; }
  .upl:hover { background:#1b2733; }
  .gallery { display:grid; grid-template-columns: repeat(auto-fill,minmax(110px,1fr)); gap:8px; max-height:420px; overflow-y:auto; }
  .gitem { position:relative; border-radius:8px; overflow:hidden; border:1px solid #2a2e37; background:#000; }
  .gitem img { width:100%; aspect-ratio:1/1; object-fit:cover; display:block; cursor:pointer; }
  .gitem .nm { font-size:9px; color:#9aa0a6; padding:2px 4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .gitem .del { position:absolute; top:4px; right:4px; background:rgba(0,0,0,.75); border:none; color:#ff8a8a;
                font-size:11px; padding:2px 6px; border-radius:6px; cursor:pointer; }
  .gitem.active { border-color:#4caf50; }
  .empty { font-size:12px; color:#9aa0a6; padding:20px; text-align:center; }
  a.back { color:#8ab4f8; text-decoration:none; font-size:12px; margin-left:auto; }
</style>
</head>
<body>
<header>
  <h1>🛡️ Reference Capture</h1>
  <span style="font-size:11px;color:#9aa0a6">capture / upload / pick a reference face for identity matching</span>
  <a class="back" href="/">← Back to dashboard</a>
</header>

<div class="wrap">
  <!-- Left: capture + upload -->
  <div class="card">
    <h2>Capture from camera</h2>
    <div class="vidwrap">
      <video id="cam" autoplay muted playsinline></video>
      <img id="preview" style="display:none"/>
    </div>
    <div class="err" id="err"></div>
    <div class="msg" id="msg"></div>
    <div class="bar">
      <button id="capBtn">Capture</button>
      <button id="retakeBtn" style="display:none">Retake</button>
      <button id="useBtn" style="display:none">Use as reference</button>
      <span style="flex:1"></span>
      <button id="clearBtn">Clear active ref</button>
    </div>

    <h2 style="margin-top:20px">Or upload an image</h2>
    <label class="upl" for="fileInput">📁 Click to choose an image file to upload as a test reference</label>
    <input id="fileInput" type="file" accept="image/*" style="display:none"/>
  </div>

  <!-- Right: gallery of stored references -->
  <div class="card">
    <h2>Reference Gallery (references/ folder)</h2>
    <div class="gallery" id="gallery"><div class="empty">No references yet — capture or upload one.</div></div>
  </div>
</div>

<script>
const ws = new WebSocket(`ws://${location.host}/ws`);
const $ = id => document.getElementById(id);
let stream = null, captured = null, activeName = null;

const cam = $('cam'), preview = $('preview');
const err = $('err'), msg = $('msg');

async function startCam() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
    cam.srcObject = stream; cam.style.display = 'block'; preview.style.display = 'none';
    $('capBtn').style.display = 'inline-block'; $('retakeBtn').style.display = 'none'; $('useBtn').style.display = 'none';
    err.style.display = 'none';
  } catch (e) {
    err.textContent = 'Camera error: ' + e.message + ' — you can still upload an image below.';
    err.style.display = 'block';
  }
}
function stopCam() { if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; } cam.srcObject = null; }

$('capBtn').onclick = () => {
  const c = document.createElement('canvas');
  c.width = cam.videoWidth || 640; c.height = cam.videoHeight || 480;
  c.getContext('2d').drawImage(cam, 0, 0, c.width, c.height);
  captured = c.toDataURL('image/jpeg', 0.9);
  preview.src = captured; preview.style.display = 'block'; cam.style.display = 'none';
  $('capBtn').style.display = 'none'; $('retakeBtn').style.display = 'inline-block'; $('useBtn').style.display = 'inline-block';
};
$('retakeBtn').onclick = () => { captured = null; cam.style.display = 'block'; preview.style.display = 'none';
  $('capBtn').style.display = 'inline-block'; $('retakeBtn').style.display = 'none'; $('useBtn').style.display = 'none'; };

function sendBytes(bytes) { ws.send(JSON.stringify({ type: 'ref_capture', data: Array.from(bytes) })); }
$('useBtn').onclick = () => {
  if (!captured) return;
  const bin = atob(captured.split(',')[1]); const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  sendBytes(bytes); $('useBtn').textContent = 'Registering…'; $('useBtn').disabled = true;
};

$('clearBtn').onclick = () => { ws.send(JSON.stringify({ type: 'ref_clear' })); msg.textContent = 'Active reference cleared.'; };

// Upload
$('fileInput').addEventListener('change', async () => {
  const f = $('fileInput').files[0];
  if (!f) return;
  const fd = new FormData(); fd.append('image', f);
  try {
    const r = await fetch('/ref/upload', { method: 'POST', body: fd });
    const j = await r.json();
    msg.textContent = j.message; msg.style.color = j.ok ? '#9be69b' : '#ff8a8a';
    ws.send(JSON.stringify({ type: 'refs_list' }));
  } catch (e) { msg.textContent = 'Upload failed: ' + e.message; msg.style.color = '#ff8a8a'; }
});

// Gallery
function renderGallery(refs) {
  const g = $('gallery');
  if (!refs.length) { g.innerHTML = '<div class="empty">No references yet.</div>'; return; }
  g.innerHTML = refs.map(r =>
    `<div class="gitem ${r.name === activeName ? 'active' : ''}">
       <img src="${r.url}" onclick="ws.send(JSON.stringify({type:'ref_select',name:'${r.name}'}))" title="Click to use as reference"/>
       <span class="nm">${r.name}</span>
       <button class="del" onclick="event.stopPropagation();ws.send(JSON.stringify({type:'ref_delete',name:'${r.name}'}))">✕</button>
     </div>`
  ).join('');
}

ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.type === 'refs') renderGallery(m.data);
  else if (m.type === 'ref') {
    msg.textContent = m.message; msg.style.color = m.ok ? '#9be69b' : '#ff8a8a';
    $('useBtn').disabled = false; $('useBtn').textContent = 'Use as reference';
    if (m.ok) { stopCam(); captured = null; }
    ws.send(JSON.stringify({ type: 'refs_list' }));
  }
  else if (m.type === 'source') { activeName = null; }
};

startCam();
ws.onopen = () => ws.send(JSON.stringify({ type: 'refs_list' }));

// Release the camera if the page is navigated away, reloaded, or hidden —
// otherwise the stream stays held and reopening /ref reports "device in use".
function releaseCamera() { stopCam(); }
window.addEventListener('pagehide', releaseCamera);
window.addEventListener('beforeunload', releaseCamera);
document.addEventListener('visibilitychange', () => { if (document.hidden) releaseCamera(); });
</script>
</body>
</html>
"""
from stream_server import (
    ProctorSession,
)

import signal
import sys

def signal_handler(sig, frame):
    print("\n[DEV SERVER] Stopping server gracefully & finalizing video...")
    if hasattr(server, 'session') and server.session:
        server.session.finalize_recording()
    sys.exit(0)

# Main block me server start hone se pehle register karein
signal.signal(signal.SIGINT, signal_handler)

import threading
import wave


class AudioRecorder:

  def __init__(self, output_path):
    self.output_path = output_path
    self.is_recording = False
    self._thread = None

  def start(self):
    if self.is_recording:
      return
    self.is_recording = True
    self._thread = threading.Thread(target=self._record, daemon=True)
    self._thread.start()

  def _record(self):
    try:
      import pyaudio
    except ImportError:
      print("[AUDIO] PyAudio installed nahi hai! Run: pip install pyaudio")
      return

    p = pyaudio.PyAudio()
    try:
      stream = p.open(
          format=pyaudio.paInt16,
          channels=1,
          rate=44100,
          input=True,
          frames_per_buffer=1024,
      )
    except Exception as e:
      print(f"[AUDIO ERROR] Mic open nahi ho paya: {e}")
      p.terminate()
      return

    frames = []
    print("[AUDIO] Mic recording started...")
    while self.is_recording:
      try:
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)
      except Exception:
        break

    stream.stop_stream()
    stream.close()

    os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
    wf = wave.open(self.output_path, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(44100)
    wf.writeframes(b"".join(frames))
    wf.close()
    p.terminate()
    print(f"[AUDIO] Audio saved: {self.output_path}")

  def stop(self):
    if not self.is_recording:
      return
    self.is_recording = False
    if self._thread:
      self._thread.join(timeout=2.0)



# ── Detection harness ────────────────────────────────────────────────────────
class DevHarness:
    def __init__(self, source, ref_path=None, fps=5, auto=True, threshold=None):
        if threshold is not None:
            import stream_server as ss
            ss.TRIGGER_THRESHOLD = threshold
        self.source = source
        self.fps = max(1, fps)
        self.auto = auto
        self.running = False
        self.clients = set()
        self.session = ProctorSession(DEV_SESSION)
        self.latest_frame = None
        audio_dir = os.path.join(EVIDENCE_DIR, "recordings", DEV_SESSION)
        audio_path = os.path.join(audio_dir, f"{DEV_SESSION}.wav")
        self.audio_recorder = AudioRecorder(audio_path)
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
        self.active_ref = None  # filename of the active reference in REF_DIR

        if ref_path:
            enc = load_reference_encoding(Path(ref_path))
            if enc is not None:
                self.session.reference_encoding = enc
                # copy the --ref image into the references folder so it shows in the gallery
                try:
                    img = cv2.imread(str(ref_path))
                    if img is not None:
                        fname = f"cli_{int(time.time()*1000)}.jpg"
                        os.makedirs(REF_DIR, exist_ok=True)
                        cv2.imwrite(os.path.join(REF_DIR, fname), img)
                        self.active_ref = fname
                except Exception:
                    pass
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
            if not self.audio_recorder.is_recording:
                    self.audio_recorder.start()

            frame = await asyncio.to_thread(self.read_frame)
            if frame is None:
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
                repeat_count = max(1, int(20 / self.fps))
                for _ in range(repeat_count):
                 await asyncio.to_thread(self.session.write_video_frame, frame)

            status = _json_safe(result)
            print(
                "[DEV][STATUS]",
                "score=", status.get("suspicion_score"),
                "flags=", status.get("flags"),
                "trigger=", trigger
                )
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
        line = trigger.get("aiWarning", "")
        print(f"[DEV][ALERT] {line}")
        await self.broadcast({"type": "alert", "text": line, "kind": "violation"})

    async def _emit_threshold_alerts(self, status):
        inj = self._agent_injection(status)
        if inj:
            text, _trigger_reply = inj
            print(f"[DEV][AGENT-INJECTION] {text}")
            await self.broadcast({"type": "alert", "text": text, "kind": "violation"})

    # ── actions ────────────────────────────────────────────────────────────
    def _save_and_register_reference(self, img, name_hint="ref"):
        """Save an image into the references folder and, if it has a single
        clear face, set it as the active reference."""
        if img is None:
            return False, "Could not decode image"
        os.makedirs(REF_DIR, exist_ok=True)
        fname = f"{name_hint}_{int(time.time()*1000)}.jpg"
        ref_path = os.path.join(REF_DIR, fname)
        cv2.imwrite(ref_path, img)
        enc = load_reference_encoding(Path(ref_path))
        if enc is None:
            # keep the file in the gallery but don't activate it
            return False, f"No/ambiguous face — saved as {fname} but NOT set as reference"
        self.session.reference_encoding = enc
        self.active_ref = fname
        return True, f"Reference registered: {fname} ({len(enc)} dims)"

    def register_reference_image(self, jpeg_bytes):
        """Register a reference face from a JPEG the browser captured/uploaded."""
        if not jpeg_bytes:
            return False, "No image data received"
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return self._save_and_register_reference(img, "captured")

    def register_reference_file(self, name):
        """Set an existing file in the references folder as the active reference."""
        if "/" in name or ".." in name or not name.endswith((".jpg", ".jpeg", ".png")):
            return False, "Invalid filename"
        ref_path = os.path.join(REF_DIR, name)
        if not os.path.exists(ref_path):
            return False, "Reference file not found"
        enc = load_reference_encoding(Path(ref_path))
        if enc is None:
            return False, f"No/ambiguous face in {name}"
        self.session.reference_encoding = enc
        self.active_ref = name
        return True, f"Reference set from gallery: {name} ({len(enc)} dims)"

    def clear_active_reference(self):
        self.session.reference_encoding = None
        self.active_ref = None
        return True, "Active reference cleared"

    def list_references(self):
        if not os.path.isdir(REF_DIR):
            return []
        files = sorted(
            (f for f in os.listdir(REF_DIR)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))),
            reverse=True,
        )
        return [{"name": f, "url": f"/refimg/{f}"} for f in files]

    def delete_reference(self, name):
        if "/" in name or ".." in name:
            return False, "Invalid filename"
        ref_path = os.path.join(REF_DIR, name)
        if not os.path.exists(ref_path):
            return False, "Reference not found"
        os.remove(ref_path)
        if name == self.active_ref:
            self.active_ref = None
            self.session.reference_encoding = None
            msg = f"Deleted {name} (was the active reference)"
        else:
            msg = f"Deleted {name}"
        return True, msg

    def reset(self):
        self.session = ProctorSession(DEV_SESSION)
        self._thresholds = {10: False, 30: False, 50: False, 70: False, 90: False}
        self.active_ref = None
        if self.alert_log:
            self.alert_log.clear()
        return True, "Session reset"

    def list_snapshots(self):
        d = os.path.join(EVIDENCE_DIR, DEV_SESSION)
        if not os.path.isdir(d):
            return []
        files = sorted(f for f in os.listdir(d) if f.endswith(".jpg") and f != "reference.jpg")
        return [{"name": f, "url": f"/snapshot/{f}"} for f in files]
        
    async def finalize_and_upload(self):
        """
        Finalize the recorded video using the same production
        ProctorSession.finalize_recording() path.
        """
        print("[DEV] Stopping audio recording...")
        self.audio_recorder.stop()
        await asyncio.sleep(1.0)

        print("\n[DEV] ========================================")
        print("[DEV] FINALIZING RECORDING")
        print("[DEV] ========================================")

        try:
            finalize_method = getattr(
                self.session,
                "finalize_recording",
                None
            )

            if not callable(finalize_method):
                print(
                    "[DEV][ERROR] ProctorSession.finalize_recording() "
                    "does not exist"
                )
                return None

            print("[DEV] Calling ProctorSession.finalize_recording()...")

            result = await asyncio.to_thread(
                finalize_method
            )

            if result:
                print("[DEV] VIDEO FINALIZE/UPLOAD SUCCESS")
                print(f"[DEV] S3 KEY: {result}")
                return result

            print(
                "[DEV][ERROR] finalize_recording() returned None"
            )
            return None

        except Exception as e:
            print("[DEV][ERROR] VIDEO FINALIZE/UPLOAD FAILED")
            print(f"[DEV][ERROR] {repr(e)}")

            import traceback
            traceback.print_exc()

            return None


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

        async def safe_send(obj):
            """Send a JSON message, tolerating a client that disconnected
            mid-write (ConnectionResetError / closing transport)."""
            try:
                await ws.send_str(json.dumps(obj, default=str))
            except Exception:
                # client is gone — mark closed so the outer loop stops cleanly
                return False
            return True

        harness.clients.add(ws)
        await harness.broadcast({
            "type": "run",
            "running": harness.running,
            "source": str(harness.source),
            "ref": "set" if harness.session.reference_encoding is not None else None,
            "active_ref": harness.active_ref,
        })
        await safe_send({"type": "snapshots", "data": harness.list_snapshots()})
        await safe_send({"type": "refs", "data": harness.list_references()})
        await safe_send({"type": "ref_active", "name": harness.active_ref, "url": f"/refimg/{harness.active_ref}" if harness.active_ref else None})
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
                        result = await harness.finalize_and_upload()
                        if result:
                            await safe_send({"type": "ref", "ok": True, "message": f"Video uploaded to S3: {result}"})
                        else:
                            await safe_send({"type": "ref", "ok": False, "message": "Video finalize/upload failed — check console"})
                    elif mtype == "ref_capture":
                        # browser-captured reference image (list of byte ints from data URL)
                        payload = data.get("data")
                        jpeg = bytes(payload) if isinstance(payload, list) else None
                        ok, message = await asyncio.to_thread(harness.register_reference_image, jpeg)
                        await safe_send({"type": "ref", "ok": ok, "message": message})
                        await safe_send({"type": "refs", "data": harness.list_references()})
                        await safe_send({"type": "ref_active", "name": harness.active_ref, "url": f"/refimg/{harness.active_ref}" if harness.active_ref else None})
                    elif mtype == "ref_select":
                        ok, message = await asyncio.to_thread(harness.register_reference_file, data.get("name", ""))
                        await safe_send({"type": "ref", "ok": ok, "message": message})
                        await safe_send({"type": "ref_active", "name": harness.active_ref, "url": f"/refimg/{harness.active_ref}" if harness.active_ref else None})
                    elif mtype == "ref_delete":
                        ok, message = await asyncio.to_thread(harness.delete_reference, data.get("name", ""))
                        await safe_send({"type": "ref", "ok": ok, "message": message})
                        await safe_send({"type": "refs", "data": harness.list_references()})
                        await safe_send({"type": "ref_active", "name": harness.active_ref, "url": f"/refimg/{harness.active_ref}" if harness.active_ref else None})
                    elif mtype == "ref_clear":
                        ok, message = harness.clear_active_reference()
                        await safe_send({"type": "ref", "ok": ok, "message": message})
                        await safe_send({"type": "ref_active", "name": None, "url": None})
                    elif mtype == "refs_list":
                        await safe_send({"type": "refs", "data": harness.list_references()})
                    elif mtype == "reset":
                        ok, message = harness.reset()
                        await safe_send({"type": "ref", "ok": ok, "message": message})
                    elif mtype == "snapshots":
                        await safe_send({"type": "snapshots", "data": harness.list_snapshots()})
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

    async def ref_page(request):
        return web.Response(text=REF_PAGE_HTML, content_type="text/html")

    async def ref_image(request):
        name = request.match_info["name"]
        if "/" in name or ".." in name:
            return web.Response(status=400)
        path = os.path.join(REF_DIR, name)
        if not os.path.exists(path):
            return web.Response(status=404)
        return web.FileResponse(path)

    async def ref_upload(request):
        """POST /ref/upload — multipart with an 'image' file. Saves it into the
        references folder and, if it has a clear face, sets it as the reference."""
        reader = await request.multipart()
        image_data = None
        async for part in reader:
            if part.name == "image":
                image_data = await part.read()
        if not image_data:
            return web.json_response({"ok": False, "message": "No image uploaded"})
        arr = np.frombuffer(image_data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        ok, message = await asyncio.to_thread(harness._save_and_register_reference, img, "upload")
        return web.json_response({"ok": ok, "message": message})

    async def ref_source(request):
        return web.json_response({
            "source": str(harness.source),
            "hasRef": harness.session.reference_encoding is not None,
        })

    app.router.add_get("/", index)
    app.router.add_get("/ref", ref_page)
    app.router.add_get("/refimg/{name}", ref_image)
    app.router.add_post("/ref/upload", ref_upload)
    app.router.add_get("/ref/source", ref_source)
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
    parser.add_argument("--threshold", type=int, default=3, help="violation occurrences to fire a trigger point (default 3 for dev; production uses 12)")
    parser.add_argument("--no-auto", action="store_true", help="don't auto-start the feed")
    args = parser.parse_args()

    harness = DevHarness(args.source, ref_path=args.ref, fps=args.fps, auto=not args.no_auto, threshold=args.threshold)
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

import asyncio
import signal
import sys


def signal_handler(sig, frame):
  print("\n[DEV SERVER] Stopping server gracefully & finalizing video...")
  try:
    from stream_server import ProctorSession

    # Safe retrieval using getattr
    session_obj = getattr(ProctorSession, "LAST_SESSION", None)

    if session_obj and hasattr(session_obj, "finalize_recording"):
      res = session_obj.finalize_recording()
      if asyncio.iscoroutine(res):
        try:
          loop = asyncio.get_event_loop()
          if loop.is_running():
            loop.create_task(res)
          else:
            loop.run_until_complete(res)
        except Exception:
          asyncio.run(res)
      print("[DEV SERVER] Video finalized and saved!")
    else:
      print("[DEV SERVER ERROR] No active session found to finalize.")
  except Exception as e:
    print(f"[DEV SERVER ERROR] Cleanup failed: {repr(e)}")

  print("[DEV SERVER] Exit complete.")
  sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


if __name__ == "__main__":
    main()
