from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import asyncio
import os
import urllib.request
import json
from typing import Set

import prompt_analytics_assistant.backend.db as db
import prompt_analytics_assistant.backend.tokenizer as tokenizer
import prompt_analytics_assistant.backend.ml as ml

# Model cost constants (Price per 1M tokens)
# Mid-2024 Gemini pricing:
# gemini-1.5-flash: Input $0.075 / 1M, Output $0.30 / 1M
# gemini-1.5-pro: Input $1.25 / 1M, Output $5.00 / 1M
# gemini-1.0-pro: Input $0.50 / 1M, Output $1.50 / 1M
PRICING = {
    "gemini-1.5-flash": {"input": 0.075 / 1_000_000, "output": 0.30 / 1_000_000},
    "gemini-1.5-pro": {"input": 1.25 / 1_000_000, "output": 5.00 / 1_000_000},
    "gemini-1.0-pro": {"input": 0.50 / 1_000_000, "output": 1.50 / 1_000_000},
    "gemini-3.6-flash": {"input": 0.075 / 1_000_000, "output": 0.30 / 1_000_000},
    "gemini-3.6-thinking": {"input": 1.25 / 1_000_000, "output": 5.00 / 1_000_000},
    "gemini-3.1-pro": {"input": 1.25 / 1_000_000, "output": 5.00 / 1_000_000}
}

app = FastAPI(title="Prompt Analytics Assistant API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared memory state
class AppState:
    active_prompt: str = ""
    active_model: str = "gemini-3.6-flash"
    last_saved_id: int = -1
    last_analysis: dict = {}
    overlay_sockets: Set[WebSocket] = set()

state = AppState()

WATCH_FILE_PATH = "/home/aiml-pragna/Token_evaluator/prompt.md"

async def watch_prompt_file():
    """Watches prompt.md in the workspace root for edits when using GUI/IDE."""
    last_mtime = 0
    last_content = ""
    
    if not os.path.exists(WATCH_FILE_PATH):
        try:
            with open(WATCH_FILE_PATH, "w", encoding="utf-8") as f:
                f.write("# Write/Paste your prompt here to analyze it in real time.\n")
        except:
            pass
            
    while True:
        try:
            if os.path.exists(WATCH_FILE_PATH):
                mtime = os.path.getmtime(WATCH_FILE_PATH)
                if mtime != last_mtime:
                    last_mtime = mtime
                    with open(WATCH_FILE_PATH, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        
                    # Skip placeholder lines
                    if content and not content.startswith("#"):
                        if content != last_content:
                            last_content = content
                            state.active_prompt = content
                            analysis = perform_analysis(content, state.active_model)
                            state.last_analysis = analysis
                            await broadcast_overlay_update(analysis)
        except Exception as e:
            print(f"File watcher error: {e}")
            
        await asyncio.sleep(0.5)

@app.on_event("startup")
def startup_event():
    db.init_db()
    ml.train_models()
    # Start the prompt file watcher in the background event loop
    asyncio.create_task(watch_prompt_file())

class HookSubmitRequest(BaseModel):
    model: str
    prompt: str
    input_tokens: int
    output_tokens: int

class BrowserSubmitRequest(BaseModel):
    model: str
    prompt: str
    response: str

class FeedbackRequest(BaseModel):
    prompt_id: int
    quality: str
    feedback: str = ""

def run_retrain():
    """Background task to retrain ML models."""
    print("Continuous learning trigger: Retraining models...")
    ml.train_models()

def evaluate_prompt_with_gemini(prompt: str) -> dict:
    """Calls Gemini API directly using urllib.request to evaluate prompt quality."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    system_instruction = (
        "You are an expert prompt engineer. Evaluate the user's prompt. "
        "Return a JSON object with keys: "
        "'quality' (must be exactly 'Good', 'Average', or 'Bad'), "
        "'feedback' (a brief explanation of the rating), and "
        "'suggestions' (a list of 1-3 specific suggestions to improve prompt efficiency/effectiveness)."
    )
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"Evaluate this prompt: {prompt}"}
                ]
            }
        ],
        "systemInstruction": {
            "parts": [
                {"text": system_instruction}
            ]
        },
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2
        }
    }
    
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=4) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            content_text = res_data["candidates"][0]["content"]["parts"][0]["text"]
            eval_result = json.loads(content_text)
            return {
                "quality": eval_result.get("quality", "Average"),
                "feedback": eval_result.get("feedback", ""),
                "suggestions": eval_result.get("suggestions", [])
            }
    except Exception as e:
        print(f"Gemini API quality check error: {e}")
        return None

def perform_analysis(prompt: str, model: str) -> dict:
    """Helper to tokenize and run ML analysis on a prompt."""
    input_tokens = tokenizer.count_tokens(prompt, model)
    ml_results = ml.analyze_prompt(prompt, model, input_tokens)
    
    predicted_output = ml_results["predicted_output_tokens"]
    total_tokens = input_tokens + predicted_output
    
    # Calculate estimated cost
    pricing = PRICING.get(model, PRICING["gemini-1.5-flash"])
    estimated_cost = (input_tokens * pricing["input"]) + (predicted_output * pricing["output"])
    
    quality = ml_results["quality"]
    suggestions = ml_results["suggestions"]
    confidence = ml_results["confidence"]
    
    # Query Gemini directly for quality if key is set
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key and len(prompt.strip()) > 3:
        gemini_eval = evaluate_prompt_with_gemini(prompt)
        if gemini_eval:
            quality = gemini_eval["quality"]
            suggestions = gemini_eval["suggestions"]
            confidence = 1.0 # Real LLM evaluation has high confidence
    
    return {
        "prompt": prompt,
        "model": model,
        "input_tokens": input_tokens,
        "predicted_output_tokens": predicted_output,
        "total_tokens": total_tokens,
        "estimated_cost": round(estimated_cost, 6),
        "quality": quality,
        "confidence": confidence,
        "similar_prompts": ml_results["similar_prompts"],
        "suggestions": suggestions,
        "last_saved_id": state.last_saved_id
    }

async def broadcast_overlay_update(analysis_data: dict):
    """Broadcasts active prompt analysis to all active overlay window connections."""
    if not state.overlay_sockets:
        return
    disconnected = set()
    for ws in state.overlay_sockets:
        try:
            await ws.send_json(analysis_data)
        except Exception:
            disconnected.add(ws)
    state.overlay_sockets.difference_update(disconnected)

@app.websocket("/api/ws/typing")
async def ws_typing(websocket: WebSocket):
    """WebSocket for terminal wrapper to stream user keystrokes/active prompt buffers."""
    await websocket.accept()
    print("Terminal monitor typing socket connected.")
    try:
        while True:
            data = await websocket.receive_json()
            prompt = data.get("prompt", "")
            model = data.get("model", "gemini-3.6-flash")
            
            # Update shared state
            state.active_prompt = prompt
            state.active_model = model
            
            # Perform prompt analysis
            analysis = perform_analysis(prompt, model)
            state.last_analysis = analysis
            
            # Stream the updated analysis to the overlay UI
            await broadcast_overlay_update(analysis)
            
            # Confirm analysis back to typing script (optional)
            await websocket.send_json({"status": "analyzed"})
    except WebSocketDisconnect:
        print("Terminal monitor typing socket disconnected.")
    except Exception as e:
        print(f"Typing WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass

@app.websocket("/api/ws/overlay")
async def ws_overlay(websocket: WebSocket):
    """WebSocket for the desktop overlay UI to receive real-time prompt analysis updates."""
    await websocket.accept()
    state.overlay_sockets.add(websocket)
    print(f"Desktop overlay socket connected. Total overlays: {len(state.overlay_sockets)}")
    try:
        # Immediately push the last known analysis to the new client
        if state.last_analysis:
            await websocket.send_json(state.last_analysis)
        else:
            # Send initial empty state
            await websocket.send_json({
                "prompt": "",
                "model": state.active_model,
                "input_tokens": 0,
                "predicted_output_tokens": 0,
                "total_tokens": 0,
                "estimated_cost": 0.0,
                "quality": "N/A",
                "confidence": 0.0,
                "similar_prompts": [],
                "suggestions": ["Please type a prompt in the Antigravity CLI to begin analysis."],
                "last_saved_id": state.last_saved_id
            })
            
        while True:
            # Keep-alive loop
            await websocket.receive_text()
    except WebSocketDisconnect:
        state.overlay_sockets.discard(websocket)
        print("Desktop overlay socket disconnected.")
    except Exception as e:
        state.overlay_sockets.discard(websocket)
        print(f"Overlay WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass

@app.post("/api/hook_submit")
async def hook_submit(req: HookSubmitRequest, background_tasks: BackgroundTasks):
    """Endpoint triggered by the CLI PostInvocation hook when an LLM run finishes."""
    try:
        # Estimate quality using current ML classifier
        analysis = perform_analysis(req.prompt, req.model)
        estimated_quality = analysis["quality"]
        
        # Calculate costs
        pricing = PRICING.get(req.model, PRICING["gemini-1.5-flash"])
        input_cost = req.input_tokens * pricing["input"]
        output_cost = req.output_tokens * pricing["output"]
        total_cost = input_cost + output_cost
        
        # Save prompt run to local SQLite history
        prompt_id = db.insert_prompt(
            model=req.model,
            prompt=req.prompt,
            input_tokens=req.input_tokens,
            output_tokens=req.output_tokens,
            quality=estimated_quality,
            feedback="",
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost
        )
        
        state.last_saved_id = prompt_id
        
        # Notify overlay that a run completed and feedback is eligible
        # Force refresh overlay state with last run prompt ID
        if state.last_analysis:
            state.last_analysis["last_saved_id"] = prompt_id
        
        # Broadcast immediately to let the overlay show the feedback card
        run_data = {
            "type": "run_completed",
            "prompt_id": prompt_id,
            "model": req.model,
            "prompt": req.prompt,
            "input_tokens": req.input_tokens,
            "output_tokens": req.output_tokens,
            "estimated_cost": total_cost,
            "quality": estimated_quality,
            "suggestions": analysis["suggestions"]
        }
        
        await broadcast_overlay_update(run_data)
        
        # Queue background model retrain
        background_tasks.add_task(run_retrain)
        
        return {"status": "success", "prompt_id": prompt_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/browser_submit")
async def browser_submit(req: BrowserSubmitRequest, background_tasks: BackgroundTasks):
    """Logs a completed prompt run from the web browser extension."""
    try:
        # 1. Count actual tokens using the tokenizer
        input_tokens = tokenizer.count_tokens(req.prompt, req.model)
        output_tokens = tokenizer.count_tokens(req.response, req.model)
        
        # 2. Get prompt quality from Gemini (if key is set) or offline ML fallback
        analysis = perform_analysis(req.prompt, req.model)
        quality = analysis["quality"]
        
        # 3. Calculate costs
        pricing = PRICING.get(req.model, PRICING["gemini-1.5-flash"])
        input_cost = input_tokens * pricing["input"]
        output_cost = output_tokens * pricing["output"]
        total_cost = input_cost + output_cost
        
        # 4. Insert into DB
        prompt_id = db.insert_prompt(
            model=req.model,
            prompt=req.prompt,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            quality=quality,
            feedback="",
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost
        )
        
        state.last_saved_id = prompt_id
        
        # 5. Broadcast to overlay so the UI updates with the actual values and shows the feedback card
        run_data = {
            "type": "run_completed",
            "prompt_id": prompt_id,
            "model": req.model,
            "prompt": req.prompt,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost": total_cost,
            "quality": quality,
            "suggestions": analysis["suggestions"]
        }
        await broadcast_overlay_update(run_data)
        
        # Retrain in background
        background_tasks.add_task(run_retrain)
        
        return {"status": "success", "prompt_id": prompt_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest, background_tasks: BackgroundTasks):
    """Submits manual user feedback (quality rating & text comment) for a prompt run."""
    try:
        db.update_prompt_feedback(
            prompt_id=req.prompt_id,
            quality=req.quality,
            feedback=req.feedback
        )
        # Retrain ML models in the background to incorporate this feedback immediately
        background_tasks.add_task(run_retrain)
        return {"status": "success", "message": "Feedback recorded, models retraining."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/retrain")
def trigger_retrain(background_tasks: BackgroundTasks):
    """Triggers manual retraining of the ML models in a background task."""
    try:
        background_tasks.add_task(run_retrain)
        return {"status": "success", "message": "ML models retraining triggered."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
def get_history():
    """Returns the full historical dataset from the local database."""
    try:
        return db.get_all_history()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    """Serves a beautiful, interactive single-page app history dashboard."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prompt History Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0b0f19;
            --card: #151b2c;
            --border: #232b40;
            --accent: #4e82ee;
            --green: #10b981;
            --yellow: #f59e0b;
            --red: #ef4444;
            --text-primary: #f3f4f6;
            --text-muted: #9ca3af;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            background-color: var(--bg);
            color: var(--text-primary);
            font-family: 'Outfit', sans-serif;
            padding: 40px 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        header {
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #ffffff 0%, #4e82ee 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .subtitle {
            color: var(--text-muted);
            margin-top: 5px;
            font-size: 1rem;
        }
        
        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .stat-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            display: flex;
            flex-direction: column;
            position: relative;
            overflow: hidden;
        }
        
        .stat-card::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--accent);
        }
        
        .stat-card.cost::after { background: var(--green); }
        .stat-card.tokens::after { background: var(--yellow); }
        
        .stat-label {
            color: var(--text-muted);
            font-size: 0.9rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .stat-val {
            font-size: 2rem;
            font-weight: 700;
            margin-top: 10px;
        }
        
        /* Filters */
        .filter-bar {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            align-items: center;
        }
        
        .search-box {
            flex: 1;
            min-width: 250px;
            position: relative;
        }
        
        .search-box input {
            width: 100%;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 16px;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s;
        }
        
        .search-box input:focus {
            border-color: var(--accent);
        }
        
        .select-filter {
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 16px;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.95rem;
            outline: none;
            cursor: pointer;
            min-width: 150px;
        }
        
        .select-filter:focus {
            border-color: var(--accent);
        }
        
        /* Table */
        .table-container {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }
        
        th {
            background: rgba(255, 255, 255, 0.02);
            color: var(--text-muted);
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 16px 24px;
            border-bottom: 1px solid var(--border);
        }
        
        td {
            padding: 18px 24px;
            border-bottom: 1px solid var(--border);
            font-size: 0.95rem;
            color: var(--text-primary);
        }
        
        tr:last-child td {
            border-bottom: none;
        }
        
        tr:hover td {
            background: rgba(255, 255, 255, 0.01);
        }
        
        /* Badges */
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            text-align: center;
        }
        
        .badge.good { background: rgba(16, 185, 129, 0.15); color: var(--green); }
        .badge.average { background: rgba(245, 158, 11, 0.15); color: var(--yellow); }
        .badge.bad { background: rgba(239, 68, 68, 0.15); color: var(--red); }
        .badge.na { background: rgba(255, 255, 255, 0.05); color: var(--text-muted); }
        
        .model-badge {
            font-family: monospace;
            background: rgba(78, 130, 238, 0.1);
            color: var(--accent);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.85rem;
        }
        
        .prompt-preview {
            max-width: 450px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            cursor: pointer;
            transition: color 0.2s;
        }
        
        .prompt-preview:hover {
            color: var(--accent);
        }
        
        /* Modal */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(7, 10, 19, 0.8);
            backdrop-filter: blur(8px);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            opacity: 0;
            transition: opacity 0.2s ease;
        }
        
        .modal.active {
            display: flex;
            opacity: 1;
        }
        
        .modal-content {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            width: 90%;
            max-width: 650px;
            padding: 30px;
            position: relative;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            transform: scale(0.95);
            transition: transform 0.2s ease;
        }
        
        .modal.active .modal-content {
            transform: scale(1);
        }
        
        .modal-close {
            position: absolute;
            top: 20px;
            right: 20px;
            background: none;
            border: none;
            color: var(--text-muted);
            font-size: 1.5rem;
            cursor: pointer;
            outline: none;
        }
        
        .modal-close:hover {
            color: var(--text-primary);
        }
        
        .modal-title {
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 20px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 10px;
        }
        
        .modal-section {
            margin-bottom: 20px;
        }
        
        .modal-section h4 {
            color: var(--text-muted);
            font-size: 0.85rem;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        
        .modal-section p {
            background: rgba(0, 0, 0, 0.2);
            padding: 15px;
            border-radius: 8px;
            font-size: 0.95rem;
            line-height: 1.5;
            white-space: pre-wrap;
            border: 1px solid rgba(255, 255, 255, 0.02);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>Prompt History Dashboard</h1>
                <div class="subtitle">Real-time log of prompts, actual token usage, estimated costs, and LLM evaluations</div>
            </div>
        </header>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Prompts</div>
                <div class="stat-val" id="stat-total">0</div>
            </div>
            <div class="stat-card cost">
                <div class="stat-label">Total Cost</div>
                <div class="stat-val" id="stat-cost">$0.000000</div>
            </div>
            <div class="stat-card tokens">
                <div class="stat-label">Total Tokens</div>
                <div class="stat-val" id="stat-tokens">0</div>
            </div>
        </div>
        
        <div class="filter-bar">
            <div class="search-box">
                <input type="text" id="search-input" placeholder="Search prompts or feedback...">
            </div>
            <select class="select-filter" id="model-filter">
                <option value="all">All Models</option>
                <option value="gemini-1.5-flash">gemini-1.5-flash</option>
                <option value="gemini-1.5-pro">gemini-1.5-pro</option>
                <option value="gemini-1.0-pro">gemini-1.0-pro</option>
            </select>
            <select class="select-filter" id="quality-filter">
                <option value="all">All Qualities</option>
                <option value="Good">Good</option>
                <option value="Average">Average</option>
                <option value="Bad">Bad</option>
                <option value="N/A">N/A</option>
            </select>
        </div>
        
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th style="width: 80px;">ID</th>
                        <th>Model</th>
                        <th>Prompt (Click to expand)</th>
                        <th style="width: 140px;">Input / Output</th>
                        <th style="width: 130px;">Cost</th>
                        <th style="width: 120px;">Quality</th>
                        <th style="width: 180px;">Timestamp</th>
                    </tr>
                </thead>
                <tbody id="history-table-body">
                    <tr>
                        <td colspan="7" style="text-align: center; padding: 40px; color: var(--text-muted);">Loading logs...</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
    
    <!-- Detail Modal -->
    <div class="modal" id="detail-modal">
        <div class="modal-content">
            <button class="modal-close" onclick="closeModal()">&times;</button>
            <div class="modal-title">Prompt Details</div>
            <div class="modal-section">
                <h4>Prompt Text</h4>
                <p id="modal-prompt-text"></p>
            </div>
            <div class="modal-section" id="modal-feedback-section">
                <h4>Feedback & Suggestions</h4>
                <p id="modal-feedback-text"></p>
            </div>
        </div>
    </div>
    
    <script>
        let allLogs = [];
        
        async function fetchHistory() {
            try {
                const res = await fetch('/api/history');
                allLogs = await res.json();
                renderDashboard(allLogs);
            } catch (err) {
                console.error("Failed to load history:", err);
            }
        }
        
        function renderDashboard(logs) {
            // Update stats
            document.getElementById('stat-total').innerText = logs.length;
            
            const totalCost = logs.reduce((sum, row) => sum + (row.total_cost || 0), 0);
            document.getElementById('stat-cost').innerText = '$' + totalCost.toFixed(6);
            
            const totalTokens = logs.reduce((sum, row) => sum + (row.input_tokens || 0) + (row.output_tokens || 0), 0);
            document.getElementById('stat-tokens').innerText = totalTokens.toLocaleString();
            
            // Filter logs
            const searchVal = document.getElementById('search-input').value.toLowerCase();
            const modelVal = document.getElementById('model-filter').value;
            const qualityVal = document.getElementById('quality-filter').value;
            
            const filtered = logs.filter(row => {
                const matchesSearch = row.prompt.toLowerCase().includes(searchVal) || (row.feedback || '').toLowerCase().includes(searchVal);
                const matchesModel = modelVal === 'all' || row.model === modelVal;
                const matchesQuality = qualityVal === 'all' || row.quality === qualityVal;
                return matchesSearch && matchesModel && matchesQuality;
            });
            
            const tbody = document.getElementById('history-table-body');
            tbody.innerHTML = '';
            
            if (filtered.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; padding: 40px; color: var(--text-muted);">No records found.</td></tr>';
                return;
            }
            
            filtered.forEach(row => {
                const tr = document.createElement('tr');
                
                // Format values
                const cost = row.total_cost !== undefined ? '$' + row.total_cost.toFixed(6) : '$0.000000';
                const qualityClass = row.quality ? row.quality.toLowerCase().replace('/', '') : 'na';
                const timeStr = new Date(row.timestamp + 'Z').toLocaleString();
                
                tr.innerHTML = `
                    <td style="font-weight: 600; color: var(--text-muted);">${row.id}</td>
                    <td><span class="model-badge">${row.model}</span></td>
                    <td><div class="prompt-preview" onclick="showDetail(${row.id})">${escapeHtml(row.prompt)}</div></td>
                    <td><span style="font-weight:600;">${row.input_tokens}</span> <span style="color:var(--text-muted);font-size:0.8rem;">in</span> / <span style="font-weight:600;">${row.output_tokens}</span> <span style="color:var(--text-muted);font-size:0.8rem;">out</span></td>
                    <td style="font-weight: 700; color: var(--green);">${cost}</td>
                    <td><span class="badge ${qualityClass}">${row.quality || 'N/A'}</span></td>
                    <td style="color: var(--text-muted); font-size: 0.85rem;">${timeStr}</td>
                `;
                tbody.appendChild(tr);
            });
        }
        
        function showDetail(id) {
            const item = allLogs.find(row => row.id === id);
            if (!item) return;
            
            document.getElementById('modal-prompt-text').innerText = item.prompt;
            
            const feedbackSection = document.getElementById('modal-feedback-section');
            if (item.feedback) {
                document.getElementById('modal-feedback-text').innerText = item.feedback;
                feedbackSection.style.display = 'block';
            } else {
                feedbackSection.style.display = 'none';
            }
            
            document.getElementById('detail-modal').classList.add('active');
        }
        
        function closeModal() {
            document.getElementById('detail-modal').classList.remove('active');
        }
        
        function escapeHtml(text) {
            return text
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }
        
        // Listeners
        document.getElementById('search-input').addEventListener('input', () => renderDashboard(allLogs));
        document.getElementById('model-filter').addEventListener('change', () => renderDashboard(allLogs));
        document.getElementById('quality-filter').addEventListener('change', () => renderDashboard(allLogs));
        
        // Close modal on click outside content
        document.getElementById('detail-modal').addEventListener('click', (e) => {
            if (e.target.id === 'detail-modal') closeModal();
        });
        
        // Initial Fetch
        fetchHistory();
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run("prompt_analytics_assistant.backend.main:app", host="127.0.0.1", port=8000, log_level="info")
