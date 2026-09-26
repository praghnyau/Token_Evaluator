#!/usr/bin/env python3
import os
import sys
import sqlite3
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, r2_score

# Configuration
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(WORKSPACE_DIR, "backend", "history.db")

def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def main():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}. Please run the assistant first.", file=sys.stderr)
        sys.exit(1)

    print("Connecting to prompt analytics database...")
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT model, prompt, input_tokens, output_tokens, quality FROM history", conn)
    except Exception as e:
        print(f"Error reading database: {e}", file=sys.stderr)
        conn.close()
        sys.exit(1)
    conn.close()

    if len(df) < 10:
        print(f"Warning: Only {len(df)} samples in database. Need at least 10 for evaluation.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(df)} historical runs in database. Preprocessing...")
    df["clean_prompt"] = df["prompt"].apply(clean_text)

    # 1. TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    X_tfidf = vectorizer.fit_transform(df["clean_prompt"]).toarray()

    lengths = (df["prompt"].apply(len).values.reshape(-1, 1) / 500.0)
    word_counts = (df["prompt"].apply(lambda x: len(x.split())).values.reshape(-1, 1) / 100.0)
    has_code = df["prompt"].apply(lambda x: 1.0 if "```" in x else 0.0).values.reshape(-1, 1)
    has_list = df["prompt"].apply(lambda x: 1.0 if any(char in x for char in ["- ", "* ", "1. "]) else 0.0).values.reshape(-1, 1)

    X_class = np.hstack([X_tfidf, lengths, word_counts, has_code, has_list])
    y_class = df["quality"].values

    # Quality Classifier Train/Test Split
    print("\n--- 1. Quality Classifier Evaluation ---")
    try:
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
        )
        classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
        classifier.fit(X_train_c, y_train_c)
        y_pred_c = classifier.predict(X_test_c)
        
        print(f"Overall Classification Accuracy: {accuracy_score(y_test_c, y_pred_c):.2%}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test_c, y_pred_c, zero_division=0))
    except Exception as e:
        print(f"Could not evaluate classifier (e.g. imbalanced single-class seed data): {e}")

    # 2. Output Token Regressor Evaluation
    print("\n--- 2. Output Token Regressor Evaluation ---")
    is_pro = df["model"].apply(lambda m: 1.0 if "pro" in str(m).lower() else 0.0).values.reshape(-1, 1)
    input_tokens = df["input_tokens"].astype(float).values.reshape(-1, 1) / 100.0

    X_reg = np.hstack([X_tfidf, is_pro, input_tokens])
    y_reg = df["output_tokens"].astype(float).values

    try:
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
        regressor = Ridge(alpha=1.0)
        regressor.fit(X_train_r, y_train_r)
        y_pred_r = regressor.predict(X_test_r)
        
        mae = mean_absolute_error(y_test_r, y_pred_r)
        r2 = r2_score(y_test_r, y_pred_r)
        
        print(f"Mean Absolute Error (MAE): {mae:.2f} tokens")
        print(f"R² (Coefficient of Determination) Score: {r2:.4f}")
    except Exception as e:
        print(f"Could not evaluate regressor: {e}")

    print("\n-------------------------------------------------------------")
    print("To trigger model retraining in the active backend server, run:")
    print("  curl -X POST http://127.0.0.1:8000/api/retrain")
    print("-------------------------------------------------------------")

if __name__ == "__main__":
    main()
