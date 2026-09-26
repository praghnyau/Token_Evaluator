# Prompt Analytics Assistant — User Documentation

## What is this?

A real-time prompt analysis tool that runs alongside your Antigravity CLI (or browser).
As you type a prompt, it shows you token counts, estimated cost, prompt quality, and suggestions to improve it.

---

## Requirements

- Linux (Ubuntu recommended)
- Python 3.12
- A display (for the desktop overlay)
- `GEMINI_API_KEY` set in `.env` (optional but recommended for accurate quality evaluation)

---

## Setup (One Time)

### 1. Set your Gemini API Key

Edit the `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

### 2. Install dependencies

```bash
cd /home/aiml-pragna/Token_evaluator
.venv/bin/pip install -r requirements.txt
```

> If no `requirements.txt` exists, the `.venv` is already pre-configured with all packages.

---

## How to Start

Run the backend and overlay in the background:

```bash
cd /home/aiml-pragna/Token_evaluator
bash prompt_analytics_assistant/run_background.sh
```

- Backend runs silently at `http://127.0.0.1:8000`
- Overlay window appears on screen
- You can now type prompts in the overlay editor or use the browser extension

---

## Using the Desktop Overlay

Once running, a small floating window appears on your screen.

| Element | What it does |
|---|---|
| Prompt Editor | Type or paste your prompt here to analyze it live |
| Copy button | Copies the prompt to clipboard |
| Input / Output / Total | Live token counts as you type |
| Token bar | Visual bar showing input vs output token ratio |
| Quality badge | Good / Average / Bad rating of your prompt |
| Confidence % | How confident the model is in its rating |
| Cost (top right) | Estimated API cost for this prompt |

**Drag** the window by clicking the dark header bar.
**Close** it with the ✕ button.

The overlay auto-snaps next to your terminal window if a terminal is detected on screen.

---

## Using the Prompt File (Alternative input)

You can also edit the file `prompt.md` in the project root directly:

```bash
nano /home/aiml-pragna/Token_evaluator/prompt.md
```

The backend watches this file every 0.5 seconds. Any change triggers a live analysis and updates the overlay automatically.

> Lines starting with `#` are ignored.

---

## Using the Browser Extension

Analyzes prompts on Gemini, ChatGPT, and Claude websites in real time.

### Install

1. Open Chrome/Edge and go to `chrome://extensions`
2. Enable **Developer Mode** (top right toggle)
3. Click **Load unpacked**
4. Select the folder: `prompt_analytics_assistant/extension/`

### Usage

- Visit `https://gemini.google.com`, `https://chatgpt.com`, or `https://claude.ai`
- A small overlay appears at the bottom-right of the page
- As you type your prompt, it shows quality badge + suggestions
- After the AI responds, the full prompt+response is logged to your history automatically
- The extension auto-detects which model you're using

> The backend must be running (`run_background.sh`) for the extension to work.

---

## History Dashboard

View all past prompts, token usage, and costs in a web UI:

```
http://127.0.0.1:8000
```

Features:
- Total prompts, total cost, total tokens summary
- Search prompts by text or feedback
- Filter by model or quality (Good / Average / Bad)
- Click any prompt row to see full text and feedback

---

## API Endpoints (for developers)

| Endpoint | Method | Description |
|---|---|---|
| `/api/history` | GET | Get all prompt history |
| `/api/feedback` | POST | Submit quality feedback for a prompt |
| `/api/browser_submit` | POST | Called by browser extension after each run |
| `/api/retrain` | POST | Manually trigger ML model retraining |
| `/api/ws/overlay` | WebSocket | Overlay UI connects here for live updates |
| `/api/ws/typing` | WebSocket | Sends live keystrokes for real-time analysis |

---

## Stopping the App

If you used `run_background.sh`, stop everything with:

```bash
fuser -k 8000/tcp
pkill -f "overlay/app.py"
```

If you used `run.sh`, just exit the CLI session (Ctrl+D or `exit`) — it cleans up automatically.

---

## Troubleshooting

**Overlay doesn't appear**
- Make sure you have a display (not headless SSH). Run `echo $DISPLAY` — it should return `:0` or similar.

**Quality always shows "Average"**
- Set `GEMINI_API_KEY` in `.env`. Without it, the offline ML model is used which needs more training data.

**Browser extension not connecting**
- Make sure the backend is running: `curl http://127.0.0.1:8000/api/history`
- Check `backend.log` for errors: `cat prompt_analytics_assistant/backend.log`

**Port 8000 already in use**
```bash
fuser -k 8000/tcp
```
Then restart.
