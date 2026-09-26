import sqlite3
import os
import csv

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.db")
SEED_PATH = "/home/aiml-pragna/Token_evaluator/evaluation.csv"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema and loads the seed dataset from CSV if empty."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Core history table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model TEXT NOT NULL,
        prompt TEXT NOT NULL,
        input_tokens INTEGER NOT NULL,
        output_tokens INTEGER NOT NULL,
        quality TEXT NOT NULL,
        feedback TEXT,
        input_cost REAL DEFAULT 0.0,
        output_cost REAL DEFAULT 0.0,
        total_cost REAL DEFAULT 0.0,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    
    # Run migration check if table already existed without cost columns
    cursor.execute("PRAGMA table_info(history)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if "input_cost" not in columns:
        cursor.execute("ALTER TABLE history ADD COLUMN input_cost REAL DEFAULT 0.0")
    if "output_cost" not in columns:
        cursor.execute("ALTER TABLE history ADD COLUMN output_cost REAL DEFAULT 0.0")
    if "total_cost" not in columns:
        cursor.execute("ALTER TABLE history ADD COLUMN total_cost REAL DEFAULT 0.0")
    conn.commit()
    
    # Check if we need to load seed data
    cursor.execute("SELECT COUNT(*) FROM history WHERE model = 'gemini-3.6-flash'")
    has_new_dataset = cursor.fetchone()[0] > 0
    
    if not has_new_dataset:
        print("Old dataset detected. Resetting database to seed from new Team_6_Gemini_dataset.csv...")
        cursor.execute("DELETE FROM history")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='history'")
        conn.commit()
        
    cursor.execute("SELECT COUNT(*) FROM history")
    count = cursor.fetchone()[0]
    
    if count == 0 and os.path.exists(SEED_PATH):
        try:
            # Local pricing definition for cost seeding
            PRICING = {
                "gemini-1.5-flash": {"input": 0.075 / 1_000_000, "output": 0.30 / 1_000_000},
                "gemini-1.5-pro": {"input": 1.25 / 1_000_000, "output": 5.00 / 1_000_000},
                "gemini-1.0-pro": {"input": 0.50 / 1_000_000, "output": 1.50 / 1_000_000}
            }
            
            with open(SEED_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                
                rows_inserted = 0
                for row in reader:
                    # Clean and extract model
                    model = row.get("model", "gemini-1.5-flash").strip()
                    if not model or model == "gemini":
                        model = "gemini-1.5-flash"
                        
                    prompt = row.get("prompt", "").strip()
                    if not prompt:
                        continue
                        
                    try:
                        input_tokens = int(row.get("input_tokens", 0))
                    except ValueError:
                        input_tokens = 0
                        
                    try:
                        output_tokens = int(row.get("output_tokens", 0))
                    except ValueError:
                        output_tokens = 0
                        
                    # Map quality: good -> Good, avg/average -> Average, bad -> Bad
                    quality_raw = row.get("label", row.get("quality", "average")).strip().lower()
                    if "good" in quality_raw:
                        quality = "Good"
                    elif "bad" in quality_raw:
                        quality = "Bad"
                    else:
                        quality = "Average"
                        
                    feedback = row.get("feedback", "").strip()
                    
                    # Calculate cost mapping
                    pricing = PRICING.get(model, PRICING["gemini-1.5-flash"])
                    input_cost = input_tokens * pricing["input"]
                    output_cost = output_tokens * pricing["output"]
                    total_cost = input_cost + output_cost
                    
                    cursor.execute("""
                    INSERT INTO history (model, prompt, input_tokens, output_tokens, quality, feedback, input_cost, output_cost, total_cost)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (model, prompt, input_tokens, output_tokens, quality, feedback, input_cost, output_cost, total_cost))
                    rows_inserted += 1
                    
            conn.commit()
            print(f"Database seeded with {rows_inserted} prompts from {SEED_PATH}.")
        except Exception as e:
            print(f"Error seeding database: {e}")
            
    conn.close()

def insert_prompt(model: str, prompt: str, input_tokens: int, output_tokens: int, quality: str, feedback: str = "", input_cost: float = 0.0, output_cost: float = 0.0, total_cost: float = 0.0) -> int:
    """Inserts a new prompt entry into history and returns the row ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO history (model, prompt, input_tokens, output_tokens, quality, feedback, input_cost, output_cost, total_cost)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (model, prompt, input_tokens, output_tokens, quality, feedback, input_cost, output_cost, total_cost))
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return row_id

def update_prompt_feedback(prompt_id: int, quality: str, feedback: str):
    """Updates the quality rating and feedback text for an existing prompt run."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    UPDATE history
    SET quality = ?, feedback = ?
    WHERE id = ?
    """, (quality, feedback, prompt_id))
    conn.commit()
    conn.close()

def get_all_history():
    """Retrieves all prompt records sorted by ID descending (newest first)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM history ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
