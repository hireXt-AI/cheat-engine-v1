#!/bin/bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "=== Cheating Detection Engine ==="

# Install dependencies if needed
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate

echo "Installing dependencies..."
pip install -q -r requirements.txt

# Run with PM2
PORT="${CHEATING_ENGINE_PORT:-6544}"

if pm2 describe cheating-engine > /dev/null 2>&1; then
  pm2 restart cheating-engine --update-env
else
  pm2 start .venv/bin/python --name "cheating-engine" -- stream_server.py --port "$PORT"
  pm2 save
fi

echo "=== Cheating engine running on port $PORT ==="
