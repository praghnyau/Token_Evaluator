import tkinter as tk
from tkinter import ttk, messagebox
import json
import threading
import queue
import asyncio
import websockets
import subprocess
import re
import urllib.request

# Palette (Catppuccin Mocha inspired dark theme)
COLOR_BG = "#1e1e2e"          # Main window background
COLOR_CARD = "#181825"        # Card background
COLOR_HEADER = "#313244"      # Drag handle / Header background
COLOR_BORDER = "#45475a"      # Subtle borders
COLOR_TXT_PRIMARY = "#cdd6f4" # Main white text
COLOR_TXT_MUTED = "#a6adc8"   # Secondary gray text
COLOR_ACCENT = "#89b4fa"      # Accent Blue
COLOR_GREEN = "#a6e3a1"       # Good prompt color
COLOR_YELLOW = "#f9e2af"      # Average prompt color
COLOR_RED = "#f38ba8"         # Bad prompt color

WS_URL = "ws://127.0.0.1:8000/api/ws/overlay"
FEEDBACK_API_URL = "http://127.0.0.1:8000/api/feedback"

class OverlayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Prompt Analytics Assistant")
        self.root.configure(bg=COLOR_BG)
        
        # Borderless and stay-on-top window setup
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        
        # Setup dimensions
        self.width = 380
        self.height = 380
        self.minimized = False
        
        # Dragging variables
        self.drag_x = 0
        self.drag_y = 0
        
        # State
        self.current_prompt_id = -1
        self.feedback_rating = "Average"
        self.active_model = "gemini-3.6-flash"
        self.last_clipboard_val = ""
        
        # Typing state for bidirectional sync
        self.debounce_timer = None
        self.typing_queue = queue.Queue()
        
        self.create_widgets()
        
        # Start background snap tracking (polls active terminal size/location)
        self.track_terminal_geometry()
        
        # Start WebSocket listeners
        self.start_ws_listener()
        self.start_typing_ws()

    def create_widgets(self):
        # 1. Header (acts as drag handle)
        self.header = tk.Frame(self.root, bg=COLOR_HEADER, height=35)
        self.header.pack(fill=tk.X, side=tk.TOP)
        self.header.pack_propagate(False)
        
        self.header.bind("<Button-1>", self.on_drag_start)
        self.header.bind("<B1-Motion>", self.on_drag_motion)
        
        header_title = tk.Label(self.header, text="▲ PROMPT ANALYTICS ASSISTANT", bg=COLOR_HEADER, fg=COLOR_TXT_PRIMARY, font=("Outfit", 10, "bold"))
        header_title.pack(side=tk.LEFT, padx=10)
        header_title.bind("<Button-1>", self.on_drag_start)
        header_title.bind("<B1-Motion>", self.on_drag_motion)
        
        close_btn = tk.Label(self.header, text="✕", bg=COLOR_HEADER, fg=COLOR_TXT_MUTED, font=("Outfit", 11, "bold"), cursor="hand2")
        close_btn.pack(side=tk.RIGHT, padx=10)
        close_btn.bind("<Button-1>", lambda e: self.root.quit())

        min_btn = tk.Label(self.header, text="─", bg=COLOR_HEADER, fg=COLOR_TXT_MUTED, font=("Outfit", 11, "bold"), cursor="hand2")
        min_btn.pack(side=tk.RIGHT, padx=(0, 4))
        min_btn.bind("<Button-1>", self.toggle_minimize)
        
        # Container frame with padding
        self.container = tk.Frame(self.root, bg=COLOR_BG, padx=12, pady=12)
        self.container.pack(fill=tk.BOTH, expand=True)
        
        # 2. Model Label & Cost
        self.meta_frame = tk.Frame(self.container, bg=COLOR_BG)
        self.meta_frame.pack(fill=tk.X, pady=(0, 8))
        
        self.model_lbl = tk.Label(self.meta_frame, text="Active Model: gemini-1.5-flash", bg=COLOR_BG, fg=COLOR_ACCENT, font=("Outfit", 10, "bold"))
        self.model_lbl.pack(side=tk.LEFT)
        
        self.cost_lbl = tk.Label(self.meta_frame, text="$0.000000", bg=COLOR_BG, fg=COLOR_GREEN, font=("Outfit", 10, "bold"))
        self.cost_lbl.pack(side=tk.RIGHT)
        
        # 2.5. Active Prompt Editor (for GUI / IDE users)
        self.input_card = tk.Frame(self.container, bg=COLOR_CARD, bd=1, highlightbackground=COLOR_BORDER, highlightthickness=1)
        self.input_card.pack(fill=tk.X, pady=(0, 10))
        
        input_header = tk.Frame(self.input_card, bg=COLOR_CARD)
        input_header.pack(fill=tk.X, padx=10, pady=(8, 2))
        
        tk.Label(input_header, text="ACTIVE PROMPT EDITOR", bg=COLOR_CARD, fg=COLOR_TXT_MUTED, font=("Outfit", 8, "bold")).pack(side=tk.LEFT)
        
        self.copy_btn = tk.Label(input_header, text="🗐 Copy", bg=COLOR_HEADER, fg=COLOR_TXT_PRIMARY, font=("Outfit", 7, "bold"), padx=5, pady=1, cursor="hand2")
        self.copy_btn.pack(side=tk.RIGHT)
        self.copy_btn.bind("<Button-1>", self.copy_prompt_to_clipboard)
        
        self.prompt_input = tk.Text(self.input_card, bg=COLOR_BG, fg=COLOR_TXT_PRIMARY, font=("Outfit", 9), height=3, wrap=tk.WORD, bd=1, highlightthickness=0, insertbackground=COLOR_TXT_PRIMARY)
        self.prompt_input.pack(fill=tk.X, padx=10, pady=(2, 8))
        self.prompt_input.bind("<KeyRelease>", self.on_typing_keyup)

        # 3. Tokens Card
        self.tokens_card = tk.Frame(self.container, bg=COLOR_CARD, bd=1, highlightbackground=COLOR_BORDER, highlightthickness=1)
        self.tokens_card.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(self.tokens_card, text="TOKEN ESTIMATION", bg=COLOR_CARD, fg=COLOR_TXT_MUTED, font=("Outfit", 8, "bold")).pack(anchor=tk.W, padx=10, pady=(8, 2))
        
        # Token layout
        self.token_grid = tk.Frame(self.tokens_card, bg=COLOR_CARD)
        self.token_grid.pack(fill=tk.X, padx=10, pady=(2, 6))
        
        # Labels
        tk.Label(self.token_grid, text="Input:", bg=COLOR_CARD, fg=COLOR_TXT_PRIMARY, font=("Outfit", 9)).grid(row=0, column=0, sticky=tk.W)
        self.input_tokens_lbl = tk.Label(self.token_grid, text="0", bg=COLOR_CARD, fg=COLOR_TXT_PRIMARY, font=("Outfit", 11, "bold"))
        self.input_tokens_lbl.grid(row=0, column=1, sticky=tk.W, padx=(5, 15))
        
        tk.Label(self.token_grid, text="Output (max):", bg=COLOR_CARD, fg=COLOR_TXT_PRIMARY, font=("Outfit", 9)).grid(row=0, column=2, sticky=tk.W)
        self.output_tokens_lbl = tk.Label(self.token_grid, text="0", bg=COLOR_CARD, fg=COLOR_TXT_PRIMARY, font=("Outfit", 11, "bold"))
        self.output_tokens_lbl.grid(row=0, column=3, sticky=tk.W, padx=(5, 15))
        
        tk.Label(self.token_grid, text="Total:", bg=COLOR_CARD, fg=COLOR_TXT_PRIMARY, font=("Outfit", 9)).grid(row=0, column=4, sticky=tk.W)
        self.total_tokens_lbl = tk.Label(self.token_grid, text="0", bg=COLOR_CARD, fg=COLOR_ACCENT, font=("Outfit", 11, "bold"))
        self.total_tokens_lbl.grid(row=0, column=5, sticky=tk.W, padx=(5, 0))
        
        # Token usage bar (canvas drawing)
        self.bar_canvas = tk.Canvas(self.tokens_card, bg=COLOR_CARD, height=6, highlightthickness=0)
        self.bar_canvas.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.draw_token_bar(0, 0)
        
        # 4. Prompt Quality Card
        self.quality_card = tk.Frame(self.container, bg=COLOR_CARD, bd=1, highlightbackground=COLOR_BORDER, highlightthickness=1)
        self.quality_card.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(self.quality_card, text="PROMPT QUALITY", bg=COLOR_CARD, fg=COLOR_TXT_MUTED, font=("Outfit", 8, "bold")).pack(anchor=tk.W, padx=10, pady=(8, 2))
        
        self.quality_details = tk.Frame(self.quality_card, bg=COLOR_CARD)
        self.quality_details.pack(fill=tk.X, padx=10, pady=(0, 8))
        
        # Quality badge
        self.quality_badge = tk.Label(self.quality_details, text="Average", bg=COLOR_YELLOW, fg=COLOR_BG, font=("Outfit", 9, "bold"), width=8, pady=2)
        self.quality_badge.pack(side=tk.LEFT)
        
        self.confidence_lbl = tk.Label(self.quality_details, text="Confidence: 100%", bg=COLOR_CARD, fg=COLOR_TXT_MUTED, font=("Outfit", 9))
        self.confidence_lbl.pack(side=tk.LEFT, padx=10)
        

        


    def toggle_minimize(self, event=None):
        if self.minimized:
            self.container.pack(fill=tk.BOTH, expand=True)
            self.root.geometry(f"{self.width}x{self.height}")
            self.minimized = False
        else:
            self.container.pack_forget()
            self.root.geometry(f"{self.width}x35")
            self.minimized = True

    def draw_token_bar(self, input_tok, output_tok):
        self.bar_canvas.delete("all")
        width = self.width - 45
        # Normalize: maximum expected tokens for standard tasks is 1000. Clamp ratio
        total = input_tok + output_tok
        if total == 0:
            return
        
        max_normal = max(1000.0, total * 1.2)
        input_w = (input_tok / max_normal) * width
        output_w = (output_tok / max_normal) * width
        
        # Draw input tokens (blue)
        self.bar_canvas.create_rectangle(0, 0, input_w, 6, fill=COLOR_ACCENT, outline="")
        # Draw output tokens (green/yellow)
        self.bar_canvas.create_rectangle(input_w, 0, input_w + output_w, 6, fill=COLOR_GREEN, outline="")




    # Drag handles
    def on_drag_start(self, event):
        self.drag_x = event.x
        self.drag_y = event.y

    def on_drag_motion(self, event):
        x = self.root.winfo_x() - self.drag_x + event.x
        y = self.root.winfo_y() - self.drag_y + event.y
        self.root.geometry(f"+{x}+{y}")

    # WebSocket connection
    def start_ws_listener(self):
        def ws_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.ws_client())
            
        t = threading.Thread(target=ws_loop, daemon=True)
        t.start()

    async def ws_client(self):
        while True:
            try:
                async with websockets.connect(WS_URL) as ws:
                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        self.root.after(0, self.update_ui, data)
            except Exception:
                await asyncio.sleep(2.0) # wait and retry connection

    def update_ui(self, data):
        # Check if message is a completed run event
        if data.get("type") == "run_completed":
            self.current_prompt_id = data.get("prompt_id", -1)
            
            # Populate prompt text back to the editor
            prompt = data.get("prompt", "")
            self.prompt_input.delete("1.0", tk.END)
            self.prompt_input.insert("1.0", prompt)
            
            # Update labels with actual, accurate counts
            input_tok = data.get("input_tokens", 0)
            output_tok = data.get("output_tokens", 0)
            total_tok = input_tok + output_tok
            cost = data.get("estimated_cost", 0.0)
            
            self.input_tokens_lbl.config(text=str(input_tok))
            self.output_tokens_lbl.config(text=str(output_tok))
            self.total_tokens_lbl.config(text=str(total_tok))
            self.draw_token_bar(input_tok, output_tok)
            self.cost_lbl.config(text=f"${cost:.6f}")
            
            # Update quality badge and confidence
            quality = data.get("quality", "Average")
            self.quality_badge.config(text=quality)
            if quality == "Good":
                self.quality_badge.config(bg=COLOR_GREEN, fg=COLOR_BG)
            elif quality == "Average":
                self.quality_badge.config(bg=COLOR_YELLOW, fg=COLOR_BG)
            elif quality == "Bad":
                self.quality_badge.config(bg=COLOR_RED, fg=COLOR_BG)
            else:
                self.quality_badge.config(bg=COLOR_HEADER, fg=COLOR_TXT_MUTED)
                
            self.confidence_lbl.config(text="Confidence: 100%")
            
            return

        # Regular typing update
        model = data.get("model", "gemini-1.5-flash")
        self.active_model = model
        
        prompt = data.get("prompt", "")
        # Update overlay text input if it changed and the user is NOT currently focusing/editing it
        try:
            focused_widget = self.root.focus_get()
            if focused_widget != self.prompt_input:
                current_val = self.prompt_input.get("1.0", tk.END).strip()
                if current_val != prompt.strip():
                    self.prompt_input.delete("1.0", tk.END)
                    self.prompt_input.insert("1.0", prompt)
        except Exception:
            pass
            
        input_tok = data.get("input_tokens", 0)
        output_tok = data.get("predicted_output_tokens", 0)
        total_tok = data.get("total_tokens", 0)
        cost = data.get("estimated_cost", 0.0)
        quality = data.get("quality", "Average")
        confidence = data.get("confidence", 1.0)
        suggestions = data.get("suggestions", [])
        similar = data.get("similar_prompts", [])
        
        # Update widgets
        self.model_lbl.config(text=f"Active Model: {model}")
        self.cost_lbl.config(text=f"${cost:.6f}")
        
        self.input_tokens_lbl.config(text=str(input_tok))
        self.output_tokens_lbl.config(text=str(output_tok))
        self.total_tokens_lbl.config(text=str(total_tok))
        
        self.draw_token_bar(input_tok, output_tok)
        
        # Update quality badge
        self.quality_badge.config(text=quality)
        if quality == "Good":
            self.quality_badge.config(bg=COLOR_GREEN, fg=COLOR_BG)
            self.confidence_lbl.config(text=f"Confidence: {int(confidence * 100)}%")
        elif quality == "Average":
            self.quality_badge.config(bg=COLOR_YELLOW, fg=COLOR_BG)
            self.confidence_lbl.config(text=f"Confidence: {int(confidence * 100)}%")
        elif quality == "Bad":
            self.quality_badge.config(bg=COLOR_RED, fg=COLOR_BG)
            self.confidence_lbl.config(text=f"Confidence: {int(confidence * 100)}%")
        else: # "N/A"
            self.quality_badge.config(bg=COLOR_HEADER, fg=COLOR_TXT_MUTED)
            self.confidence_lbl.config(text="Confidence: N/A")
            


    # Snapping logic
    def track_terminal_geometry(self):
        """Polls the active window to check if it's our CLI. Snaps overlay next to it if found."""
        geom = self.get_active_terminal_geometry()
        if geom:
            tx, ty, tw, th = geom
            
            # Position layout: To the right of the terminal, aligned to top
            padding = 10
            popup_x = tx + tw + padding
            popup_y = ty
            
            # Boundary checks: if snap goes off screen right, place it on the left of the terminal
            screen_w = self.root.winfo_screenwidth()
            if popup_x + self.width > screen_w:
                popup_x = tx - self.width - padding
                if popup_x < 0:
                    popup_x = padding # Fallback to leftmost screen padding
            
            # Smoothly apply snap geometry
            # Check if current position is already close to avoid constant jittering
            cx = self.root.winfo_x()
            cy = self.root.winfo_y()
            if abs(cx - popup_x) > 10 or abs(cy - popup_y) > 10:
                self.root.geometry(f"+{popup_x}+{popup_y}")
                
        # Run snap check again in 2 seconds
        self.root.after(2000, self.track_terminal_geometry)

    def get_active_terminal_geometry(self):
        """Queries X11 active window geometry using xprop and xwininfo."""
        try:
            # 1. Read the active window ID
            out = subprocess.check_output("xprop -root _NET_ACTIVE_WINDOW", shell=True, stderr=subprocess.DEVNULL).decode()
            match = re.search(r"window id # (0x[0-9a-fA-F]+)", out)
            if match and match.group(1) != "0x0":
                win_id = match.group(1)
                
                # Check if it is a terminal or command window running our prompt wrapper
                # We can inspect window class or name using xprop
                name_out = subprocess.check_output(f"xprop -id {win_id} WM_NAME WM_CLASS", shell=True, stderr=subprocess.DEVNULL).decode()
                
                # Check if class/name indicates it's a Terminal or agy CLI
                # Usually terminal windows have 'terminal', 'gnome-terminal', 'xterm', or 'bash' in them
                is_target = any(term in name_out.lower() for term in ["terminal", "term", "agy", "gemini"])
                
                if is_target:
                    # Read dimensions from xwininfo
                    info = subprocess.check_output(f"xwininfo -id {win_id}", shell=True, stderr=subprocess.DEVNULL).decode()
                    x = int(re.search(r"Absolute upper-left X:\s+(-?\d+)", info).group(1))
                    y = int(re.search(r"Absolute upper-left Y:\s+(-?\d+)", info).group(1))
                    width = int(re.search(r"Width:\s+(\d+)", info).group(1))
                    height = int(re.search(r"Height:\s+(\d+)", info).group(1))
                    return x, y, width, height
        except Exception:
            pass
        return None

    def copy_prompt_to_clipboard(self, event=None):
        """Copies the text from the prompt editor to the system clipboard."""
        try:
            content = self.prompt_input.get("1.0", tk.END).strip()
            if content:
                self.root.clipboard_clear()
                self.root.clipboard_append(content)
                self.root.update()
                # Flash success visual feedback
                self.copy_btn.config(text="✓ Copied", fg=COLOR_GREEN)
                self.root.after(1500, lambda: self.copy_btn.config(text="🗐 Copy", fg=COLOR_TXT_PRIMARY))
        except Exception as e:
            messagebox.showerror("Clipboard Error", f"Failed to copy: {e}")

    def on_typing_keyup(self, event):
        """Triggers a rate-limiting debounce before sending prompt updates to the backend."""
        if self.debounce_timer:
            self.root.after_cancel(self.debounce_timer)
        self.debounce_timer = self.root.after(150, self.send_typing_to_backend)

    def send_typing_to_backend(self):
        """Pushes the current editor content to the WebSocket queue."""
        try:
            content = self.prompt_input.get("1.0", tk.END).strip()
            self.typing_queue.put({"prompt": content, "model": self.active_model})
        except Exception:
            pass

    def start_typing_ws(self):
        """Starts the background thread managing the typing sync connection."""
        def ws_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.typing_ws_client())
            
        t = threading.Thread(target=ws_loop, daemon=True)
        t.start()

    async def typing_ws_client(self):
        """WebSocket client for sending active typing buffers from the editor to the backend."""
        loop = asyncio.get_running_loop()
        while True:
            try:
                async with websockets.connect("ws://127.0.0.1:8000/api/ws/typing") as ws:
                    while True:
                        try:
                            item = await loop.run_in_executor(None, lambda: self.typing_queue.get(timeout=0.1))
                            await ws.send(json.dumps(item))
                            await ws.recv() # wait for confirmation
                        except queue.Empty:
                            await asyncio.sleep(0.05)
            except Exception:
                await asyncio.sleep(2.0)



if __name__ == "__main__":
    root = tk.Tk()
    app = OverlayApp(root)
    
    # Set default window coordinates on screen (near right hand side center)
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    default_x = sw - app.width - 50
    default_y = 100
    root.geometry(f"{app.width}x{app.height}+{default_x}+{default_y}")
    
    root.mainloop()
