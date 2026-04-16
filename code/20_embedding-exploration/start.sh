#!/usr/bin/env bash
# Start the Embedding Explorer (backend + frontend)
# Usage: ./start.sh

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"

# ── Load API keys ──────────────────────────────────────────────────────────────
if [ -f "$REPO_ROOT/keys.env" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$REPO_ROOT/keys.env"
  set +a
  echo "Loaded keys from $REPO_ROOT/keys.env"
else
  echo "WARNING: $REPO_ROOT/keys.env not found — OPENAI_API_KEY must be set in environment"
fi

# ── Backend ────────────────────────────────────────────────────────────────────
echo "Starting backend on :8000"
cd "$ROOT/backend"

if ! python3 -c "import fastapi, uvicorn, numpy, openai" 2>/dev/null; then
  echo "Installing backend deps..."
  pip install fastapi uvicorn numpy openai
fi

if ! python3 -c "import umap" 2>/dev/null; then
  echo "Installing umap-learn (first time takes a minute)..."
  pip install umap-learn
fi

python3 server.py &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# ── Frontend ───────────────────────────────────────────────────────────────────
echo "Starting frontend on :5173"
cd "$ROOT/frontend"

if [ ! -d node_modules ]; then
  echo "Installing frontend deps..."
  npm install
fi

npm run dev &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

# ── Cleanup ────────────────────────────────────────────────────────────────────
cleanup() {
  echo "Stopping..."
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo ""
echo "  Embedding Explorer: http://localhost:5173"
echo "  Backend API:        http://localhost:8000/api"
echo "  (Ctrl-C to stop)"
echo ""

wait
