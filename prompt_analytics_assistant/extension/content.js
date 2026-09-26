// ── State ────────────────────────────────────────────────────────────────────
let socket = null;
let debounceTimer = null;
let isConnected = false;
let lastLoggedPrompt = "";
let lastLoggedResponse = "";
let lastSeenResponseText = "";
let stableCycleCount = 0;

// ── Overlay UI ───────────────────────────────────────────────────────────────
function createOverlay() {
  if (document.getElementById('paa-overlay')) return;

  const el = document.createElement('div');
  el.id = 'paa-overlay';
  el.innerHTML = `
    <div id="paa-header">
      <span id="paa-dot"></span>
      <span id="paa-title">Prompt Analytics</span>
      <span id="paa-toggle" title="Minimize">−</span>
    </div>
    <div id="paa-body">
      <div id="paa-quality-row">
        <span id="paa-badge">—</span>
        <span id="paa-conf"></span>
      </div>
      <div id="paa-bar-wrap"><div id="paa-bar"></div></div>
      <div id="paa-suggestions"></div>
    </div>
  `;

  const style = document.createElement('style');
  style.textContent = `
    #paa-overlay {
      position: fixed; bottom: 80px; right: 20px; z-index: 999999;
      width: 260px; background: #0f0f0f; border: 1px solid #2a2a4a;
      border-radius: 10px; font-family: 'Segoe UI', sans-serif;
      box-shadow: 0 8px 32px rgba(0,0,0,0.6); transition: all 0.2s;
    }
    #paa-header {
      background: #1a1a2e; padding: 8px 12px; border-radius: 10px 10px 0 0;
      display: flex; align-items: center; gap: 6px; cursor: pointer;
      border-bottom: 1px solid #2a2a4a;
    }
    #paa-dot { width:8px; height:8px; border-radius:50%; background:#ef4444; flex-shrink:0; }
    #paa-dot.on { background:#22c55e; }
    #paa-title { font-size:12px; font-weight:600; color:#a78bfa; flex:1; }
    #paa-toggle { font-size:14px; color:#6b7280; cursor:pointer; padding: 0 2px; }
    #paa-body { padding: 10px 12px; }
    #paa-body.hidden { display:none; }
    #paa-quality-row { display:flex; align-items:center; gap:8px; margin-bottom:6px; }
    #paa-badge {
      padding: 3px 10px; border-radius:20px; font-size:12px; font-weight:600;
      background:#1e1e1e; color:#9ca3af;
    }
    #paa-badge.Good { background:#14532d; color:#4ade80; }
    #paa-badge.Average { background:#713f12; color:#fbbf24; }
    #paa-badge.Bad { background:#450a0a; color:#f87171; }
    #paa-conf { font-size:10px; color:#6b7280; }
    #paa-bar-wrap { height:3px; background:#1e1e1e; border-radius:2px; margin-bottom:8px; }
    #paa-bar { height:100%; width:0%; background:#a78bfa; border-radius:2px; transition:width 0.3s; }
    .paa-tip {
      font-size:10px; color:#d1d5db; background:#1a1a1a;
      border-left:2px solid #a78bfa; padding:5px 7px;
      border-radius:0 4px 4px 0; margin-bottom:4px; line-height:1.4;
    }
    #paa-offline { font-size:10px; color:#ef4444; text-align:center; padding:6px 0; }
  `;

  document.head.appendChild(style);
  document.body.appendChild(el);

  // Toggle minimize
  let minimized = false;
  document.getElementById('paa-toggle').addEventListener('click', (e) => {
    e.stopPropagation();
    minimized = !minimized;
    document.getElementById('paa-body').classList.toggle('hidden', minimized);
    document.getElementById('paa-toggle').textContent = minimized ? '+' : '−';
  });
}

function updateOverlay(quality, confidence, suggestions) {
  const badge = document.getElementById('paa-badge');
  const conf = document.getElementById('paa-conf');
  const bar = document.getElementById('paa-bar');
  const tips = document.getElementById('paa-suggestions');
  if (!badge) return;

  const pct = Math.round((confidence || 0) * 100);
  badge.textContent = quality || '—';
  badge.className = quality || '';
  conf.textContent = quality ? `${pct}% confidence` : '';
  bar.style.width = `${pct}%`;
  tips.innerHTML = (suggestions || []).slice(0, 3)
    .map(s => `<div class="paa-tip">${s}</div>`).join('');
}

function setConnected(val) {
  isConnected = val;
  const dot = document.getElementById('paa-dot');
  if (dot) dot.className = val ? 'on' : '';
  chrome.storage.local.set({ connected: val });
}

// ── WebSocket ────────────────────────────────────────────────────────────────
function connect() {
  socket = new WebSocket("ws://127.0.0.1:8000/api/ws/typing");

  socket.onopen = () => { setConnected(true); };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      const quality = data.quality || null;
      const confidence = data.confidence || 0;
      const suggestions = data.suggestions || [];
      updateOverlay(quality, confidence, suggestions);
      chrome.storage.local.set({ quality, confidence, suggestions });
    } catch (e) {}
  };

  socket.onclose = () => {
    setConnected(false);
    setTimeout(connect, 3000);
  };

  socket.onerror = () => { setConnected(false); };
}

// ── Model Detection ──────────────────────────────────────────────────────────
function getActiveModel() {
  const host = window.location.hostname;

  // Debug: log all buttons text to console so we can find the right selector
  const allBtns = document.querySelectorAll("button");
  const btnTexts = [...allBtns].map(b => (b.innerText || b.textContent || "").trim()).filter(t => t.length > 0 && t.length < 60);
  console.log("[PAA] All button texts:", btnTexts);

  // Site-agnostic: scan all buttons for known model name keywords
  for (const btn of allBtns) {
    const txt = (btn.innerText || btn.textContent || "").trim().toLowerCase();
    // ChatGPT models
    if (txt.includes("gpt-4o"))     return "gpt-4o";
    if (txt.includes("gpt-4"))      return "gpt-4";
    if (txt.includes("gpt-3.5"))    return "gpt-3.5-turbo";
    // Claude models
    if (txt.includes("claude 3 opus") || txt.includes("claude-3-opus"))   return "claude-opus";
    if (txt.includes("claude 3 sonnet") || txt.includes("claude-sonnet")) return "claude-sonnet";
    if (txt.includes("claude 3 haiku") || txt.includes("claude-haiku"))   return "claude-haiku";
    // Gemini models
    if (txt.includes("2.0 flash") || txt.includes("gemini 2.0")) return "gemini-2.0-flash";
    if (txt.includes("thinking"))   return "gemini-3.6-thinking";
    if (txt.includes("1.5 flash"))  return "gemini-1.5-flash";
    if (txt.includes("1.5 pro"))    return "gemini-1.5-pro";
    if (txt.includes("3.6 flash"))  return "gemini-3.6-flash";
    if (txt.includes("3.1 pro"))    return "gemini-3.1-pro";
  }

  // Fallback based on hostname
  if (host.includes("chatgpt.com")) return "gpt-4o";
  if (host.includes("claude.ai"))   return "claude-sonnet";
  if (host.includes("gemini.google.com")) return "gemini-3.6-flash";
  return "gemini-3.6-flash";
}

// ── Input Handler ────────────────────────────────────────────────────────────
function handleInput(e) {
  const target = (e.composedPath?.().length > 0) ? e.composedPath()[0] : e.target;
  if (!target) return;

  const isEditable = target.tagName === "TEXTAREA" ||
    (target.tagName === "INPUT" && target.type === "text") ||
    target.getAttribute("contenteditable") === "true" ||
    target.closest("[contenteditable='true']");

  if (!isEditable) return;

  const el = target.closest("[contenteditable='true']") || target;
  const text = (el.innerText || el.textContent || el.value || "").trim();
  if (text.startsWith("#")) return;

  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ prompt: text, model: getActiveModel() }));
    }
  }, 150);
}

document.addEventListener("input", handleInput);
document.addEventListener("keyup", handleInput);

// ── Completed Run Logger ─────────────────────────────────────────────────────
function cleanUserText(raw) {
  return raw.replace(/^(you said|you)\s*\n*/i, "").trim();
}

function submitCompletedRun(prompt, response) {
  fetch("http://127.0.0.1:8000/api/browser_submit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: getActiveModel(), prompt, response })
  }).catch(() => {});
}

function checkConversation() {
  const userEls = document.querySelectorAll("query-content, .query-content, [data-message-author-role='user']");
  const assistantEls = document.querySelectorAll("message-content, .message-content, [data-message-author-role='assistant'], .markdown, .prose, .font-claude");
  if (!userEls.length || !assistantEls.length) return;

  const latestUser = userEls[userEls.length - 1];
  const latestAssistant = assistantEls[assistantEls.length - 1];
  const isAfter = (latestUser.compareDocumentPosition(latestAssistant) & Node.DOCUMENT_POSITION_FOLLOWING) > 0;
  if (!isAfter) return;

  const promptText = cleanUserText(latestUser.innerText || latestUser.textContent || "");
  const responseText = (latestAssistant.innerText || latestAssistant.textContent || "").trim();

  if (promptText && (promptText !== lastLoggedPrompt || responseText !== lastLoggedResponse)) {
    if (responseText && responseText === lastSeenResponseText) {
      stableCycleCount++;
      if (stableCycleCount >= 2) {
        lastLoggedPrompt = promptText;
        lastLoggedResponse = responseText;
        stableCycleCount = 0;
        submitCompletedRun(promptText, responseText);
      }
    } else {
      stableCycleCount = 0;
      lastSeenResponseText = responseText;
    }
  }
}

// ── Init ─────────────────────────────────────────────────────────────────────
createOverlay();
connect();
setInterval(checkConversation, 1500);
