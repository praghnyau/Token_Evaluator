import sqlite3
import os

DB_PATH = "/home/aiml-pragna/Token_evaluator/prompt_analytics_assistant/backend/history.db"

def reindex_db():
    if not os.path.exists(DB_PATH):
        print("Database file not found.")
        return
        
    print("Connecting to database...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Fetch all rows in order of their current ID
    cursor.execute("SELECT model, prompt, input_tokens, output_tokens, quality, feedback, timestamp, input_cost, output_cost, total_cost FROM history ORDER BY id ASC")
    rows = cursor.fetchall()
    print(f"Retrieved {len(rows)} records.")
    
    # 2. Clear table and reset auto-increment sequence
    cursor.execute("DELETE FROM history")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='history'")
    conn.commit()
    print("Cleared history table and reset auto-increment sequence.")
    
    # 3. Re-insert all rows (they will get IDs 1, 2, 3, ... sequentially)
    cursor.executemany("""
    INSERT INTO history (model, prompt, input_tokens, output_tokens, quality, feedback, timestamp, input_cost, output_cost, total_cost)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    
    # 4. Verify results
    cursor.execute("SELECT MIN(id), MAX(id), COUNT(*) FROM history")
    min_id, max_id, count = cursor.fetchone()
    print(f"Re-indexing complete. New ID Range: {min_id} to {max_id} (Total Count: {count} rows).")
    
    conn.close()

if __name__ == "__main__":
    reindex_db()
