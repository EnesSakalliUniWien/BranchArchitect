#!/bin/zsh

# Backend Server Startup Script
# Starts the Flask backend server

# Kill any existing processes on backend ports and verify
for PORT in 5002; do
  echo "[backend] Checking for processes on port $PORT..."
  PIDS=$(lsof -ti :$PORT)
  if [ -n "$PIDS" ]; then
    echo "[backend] Killing processes on port $PORT: $PIDS"
    kill -9 $PIDS 2>/dev/null
    sleep 1
  fi
  # Double check port is free
  if lsof -i :$PORT | grep LISTEN; then
    echo "[backend] ERROR: Port $PORT is still in use after kill. Please free it and try again."
    exit 1
  fi
done

# Navigate to project root
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# Navigate to the webapp directory for Flask backend
cd "$PROJECT_ROOT/webapp"

# Install dependencies with poetry
echo "[backend] Installing webapp dependencies with poetry..."
if ! poetry install; then
  echo "[backend] ERROR: Poetry install failed. Check your dependencies and lock file."
  exit 1
fi

# Check brancharchitect version
echo "[backend] Checking brancharchitect version..."
BRANCHARCHITECT_VERSION=$(poetry run pip list | grep brancharchitect | awk '{print $2}' 2>/dev/null || echo "not installed via pip")
echo "[backend] Using brancharchitect version: $BRANCHARCHITECT_VERSION"

# Start the Flask backend using poetry
echo "[backend] Starting Flask backend with poetry..."
echo "[backend] Using Poetry environment"

# Set PYTHONPATH to include the project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Set library path for cairo (optional, only needed if using advanced SVG rendering)
# Uncomment the line below if you installed the 'plotting' extras
# export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"

# Check if run.py exists before starting backend
if [ ! -f "run.py" ]; then
  echo "[backend] ERROR: run.py not found in webapp directory."
  exit 1
fi

# Create a detailed startup log
STARTUP_LOG="$PROJECT_ROOT/backend_startup.log"
echo "[backend] Starting backend at $(date)" > "$STARTUP_LOG"
echo "[backend] Python path: $PYTHONPATH" >> "$STARTUP_LOG"
echo "[backend] Library path: $DYLD_LIBRARY_PATH" >> "$STARTUP_LOG"

# Run the server using poetry with proper environment, log output for diagnostics
cd "$PROJECT_ROOT"
echo "[backend] Running command: PYTHONPATH=\"$PROJECT_ROOT:$PYTHONPATH\" poetry run python webapp/run.py --host=127.0.0.1 --port=5002" >> "$STARTUP_LOG"
PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" poetry run python webapp/run.py --host=127.0.0.1 --port=5002 >"$PROJECT_ROOT/backend.log" 2>&1 &
BACKEND_PID=$!
echo "[backend] Backend PID: $BACKEND_PID" >> "$STARTUP_LOG"
echo "[backend] Backend log: $PROJECT_ROOT/backend.log"
echo "[backend] Startup log: $PROJECT_ROOT/backend_startup.log"

# Wait for backend to be ready
echo "[backend] Waiting for backend to start..."
for i in {1..30}; do
  if curl -s http://127.0.0.1:5002/about >/dev/null 2>&1; then
    echo "[backend] Backend is ready!"
    echo "[backend] Backend ready at $(date)" >> "$STARTUP_LOG"
    break
  fi
  if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "[backend] ERROR: Backend process died during startup"
    echo "[backend] Process died at $(date)" >> "$STARTUP_LOG"
    echo "[backend] Exit status: $?" >> "$STARTUP_LOG"
    echo "[backend] Last 20 lines of backend.log:"
    tail -n 20 "$PROJECT_ROOT/backend.log"
    echo "[backend] Full backend.log contents:" >> "$STARTUP_LOG"
    cat "$PROJECT_ROOT/backend.log" >> "$STARTUP_LOG"
    exit 1
  fi
  echo "[backend] Waiting for backend... ($i/30)"
  sleep 1
done

if ! curl -s http://127.0.0.1:5002/about >/dev/null 2>&1; then
  echo "[backend] ERROR: Backend failed to start within 30 seconds"
  kill $BACKEND_PID 2>/dev/null
  exit 1
fi

# Cleanup function
cleanup() {
  echo "[backend] Cleaning up..."
  if [ -n "$BACKEND_PID" ] && kill -0 $BACKEND_PID 2>/dev/null; then
    kill $BACKEND_PID 2>/dev/null
  fi
  wait $BACKEND_PID 2>/dev/null
}

# Trap cleanup on SIGINT (Ctrl+C), SIGTERM, and EXIT
trap cleanup SIGINT SIGTERM EXIT

echo "[backend] Backend PID: $BACKEND_PID"
echo "[backend] Backend server is running. Press Ctrl+C to stop."
echo "[backend] Flask backend: http://127.0.0.1:5002/"
echo "[backend] Health check endpoint: http://127.0.0.1:5002/about"

# Keep the script running
wait $BACKEND_PID || true
