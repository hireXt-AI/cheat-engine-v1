#!/bin/bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "=== Cheating Detection Engine ==="

# Prefer Python 3.11: the pinned mediapipe==0.10.21 / dlib==20.0.1 only have
# wheels up to Python 3.11 (no wheels for 3.12/3.13+). Fall back to python3 for
# environments where only the system default is available.
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
if ! command -v "$PYTHON_BIN" > /dev/null 2>&1; then
  echo "WARNING: $PYTHON_BIN not found; falling back to python3 (mediapipe/dlib install may fail on 3.12+)"
  PYTHON_BIN="python3"
fi

# Install dependencies if needed
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment with $PYTHON_BIN..."
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate

echo "Installing dependencies..."
pip install -q -r requirements.txt

# Run with PM2
PORT="${CHEATING_ENGINE_PORT:-6544}"

# Always start fresh. `pm2 restart` keeps the ORIGINAL script path, so if an
# older engine (e.g. v0) is already registered under the name "cheating-engine",
# a restart would keep running that old code forever. Deleting first guarantees
# THIS engine's venv + stream_server.py is what actually runs.
pm2 delete cheating-engine > /dev/null 2>&1 || true
pm2 start .venv/bin/python --name "cheating-engine" -- stream_server.py --port "$PORT"
pm2 save

echo "=== Cheating engine running on port $PORT ==="
