import os
import unittest
import sys
import shutil

# Ensure prompt_analytics_assistant is in python path
sys.path.insert(0, "/home/aiml-pragna/Token_evaluator")

import prompt_analytics_assistant.backend.db as db
import prompt_analytics_assistant.backend.tokenizer as tokenizer
import prompt_analytics_assistant.backend.ml as ml

class TestPromptAnalyticsBackend(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Override DB path for isolation during tests
        cls.orig_db_path = db.DB_PATH
        db.DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_history.db")
        if os.path.exists(db.DB_PATH):
            os.remove(db.DB_PATH)
            
    @classmethod
    def tearDownClass(cls):
        # Clean up test DB
        if os.path.exists(db.DB_PATH):
            os.remove(db.DB_PATH)
        db.DB_PATH = cls.orig_db_path

    def setUp(self):
        # Clear database records
        conn = db.get_connection()
        c = conn.cursor()
        c.execute("DROP TABLE IF EXISTS history")
        conn.commit()
        conn.close()
        db.init_db()

    def test_token_counting(self):
        text = "Hello world, testing token calculations."
        tokens = tokenizer.count_tokens(text)
        self.assertGreater(tokens, 0)
        
        # Test empty input
        self.assertEqual(tokenizer.count_tokens(""), 0)

    def test_database_operations(self):
        # Test insert
        row_id = db.insert_prompt("gemini-1.5-flash", "test prompt text", 10, 20, "Good", "Initial feedback")
        self.assertIsNotNone(row_id)
        
        # Test retrieve
        history = db.get_all_history()
        self.assertGreaterEqual(len(history), 1)
        self.assertEqual(history[0]["prompt"], "test prompt text")
        self.assertEqual(history[0]["quality"], "Good")
        
        # Test update
        db.update_prompt_feedback(row_id, "Bad", "New corrected feedback")
        history_updated = db.get_all_history()
        self.assertEqual(history_updated[0]["quality"], "Bad")
        self.assertEqual(history_updated[0]["feedback"], "New corrected feedback")

    def test_ml_pipeline(self):
        # Insert a few sample prompts to support ML model training
        db.insert_prompt("gemini-1.5-flash", "Write a python script to parse CSV data", 10, 100, "Good", "Explicit formatting")
        db.insert_prompt("gemini-1.5-flash", "fix code", 2, 50, "Bad", "Vague request")
        db.insert_prompt("gemini-1.5-flash", "Explain Quantum Mechanics in 3 sentences.", 10, 80, "Good", "Clear constraints")
        db.insert_prompt("gemini-1.5-flash", "tell me about computers", 5, 200, "Bad", "Too generic")
        
        # Trigger model training
        ml.train_models()
        
        # Analyze a new test prompt
        test_prompt = "Write a Python script utilizing standard library to read json"
        results = ml.analyze_prompt(test_prompt, "gemini-1.5-flash", 12)
        
        self.assertIn("quality", results)
        self.assertIn("predicted_output_tokens", results)
        self.assertIn("similar_prompts", results)
        self.assertIn("suggestions", results)
        
        # Check suggestions heuristic rules
        self.assertTrue(len(results["suggestions"]) > 0)
        
        # Test suggestions details for short prompt
        short_sugs = ml.generate_suggestions("hi", "Bad")
        self.assertTrue(any("short" in s.lower() or "context" in s.lower() for s in short_sugs))

if __name__ == "__main__":
    unittest.main()
