#!/bin/bash
# Startup script to run Prompt Analytics Assistant backend & overlay in the background
# (Ideal for GUI, IDE, and Web Browser users)

WORKSPACE_DIR="/home/aiml-pragna/Token_evaluator"

echo "Stopping any running instances..."
# Kill existing backend on port 8000 and Tkinter overlay process
fuser -k 8000/tcp &> /dev/null || true
pkill -f "prompt_analytics_assistant.overlay.app" &> /dev/null || true
pkill -f "prompt_analytics_assistant/overlay/app.py" &> /dev/null || true

echo "Starting FastAPI backend server..."
nohup $WORKSPACE_DIR/.venv/bin/python3 -m uvicorn prompt_analytics_assistant.backend.main:app --host 127.0.0.1 --port 8000 > "$WORKSPACE_DIR/prompt_analytics_assistant/backend.log" 2>&1 &
BACKEND_PID=$!
disown

# Wait for backend to boot
sleep 2

echo "Launching desktop overlay UI..."
nohup $WORKSPACE_DIR/.venv/bin/python3 $WORKSPACE_DIR/prompt_analytics_assistant/overlay/app.py > "$WORKSPACE_DIR/prompt_analytics_assistant/overlay.log" 2>&1 &
OVERLAY_PID=$!
disown

echo "--------------------------------------------------------"
echo "Prompt Analytics Assistant is now running in the background!"
echo "- Backend: http://127.0.0.1:8000"
echo "- Logs: backend.log and overlay.log"
echo "To stop them at any time, run: kill $BACKEND_PID $OVERLAY_PID"
echo "--------------------------------------------------------"
