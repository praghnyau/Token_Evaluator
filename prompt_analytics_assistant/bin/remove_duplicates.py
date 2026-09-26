import sqlite3
import os

DB_PATH = "/home/aiml-pragna/Token_evaluator/prompt_analytics_assistant/backend/history.db"

def remove_duplicates():
    if not os.path.exists(DB_PATH):
        print("Database file not found.")
        return
        
    print("Connecting to database...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Fetch all rows in chronological order
    cursor.execute("SELECT id, model, prompt, input_tokens, output_tokens, quality, feedback, timestamp, input_cost, output_cost, total_cost FROM history ORDER BY id ASC")
    rows = cursor.fetchall()
    print(f"Retrieved {len(rows)} records.")
    
    seen_prompts = set()
    unique_rows = []
    duplicates_removed = 0
    
    for row in rows:
        prompt_text = row[2]
        # Normalize: strip and lowercase to detect duplicates reliably
        normalized = (prompt_text or "").strip().lower()
        
        # Guard: if prompt is empty or just whitespace, skip duplicate check or treat as unique
        if not normalized:
            unique_rows.append(row)
            continue
            
        if normalized in seen_prompts:
            duplicates_removed += 1
        else:
            seen_prompts.add(normalized)
            unique_rows.append(row)
            
    print(f"Identified {duplicates_removed} duplicate prompts. Retaining {len(unique_rows)} unique prompts.")
    
    if duplicates_removed > 0:
        # 2. Clear table and reset auto-increment sequence
        cursor.execute("DELETE FROM history")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='history'")
        conn.commit()
        
        # 3. Re-insert the unique rows sequentially (without their original IDs so they auto-increment from 1)
        # We strip the original ID from the tuple
        cleaned_rows = [row[1:] for row in unique_rows]
        
        cursor.executemany("""
        INSERT INTO history (model, prompt, input_tokens, output_tokens, quality, feedback, timestamp, input_cost, output_cost, total_cost)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, cleaned_rows)
        conn.commit()
        
        # 4. Verify results
        cursor.execute("SELECT MIN(id), MAX(id), COUNT(*) FROM history")
        min_id, max_id, count = cursor.fetchone()
        print(f"Duplicates removed. New ID Range: {min_id} to {max_id} (Total Count: {count} rows).")
    else:
        print("No duplicate prompts found. Database is already clean.")
        
    conn.close()

if __name__ == "__main__":
    remove_duplicates()
