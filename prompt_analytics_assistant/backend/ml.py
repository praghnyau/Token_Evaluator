import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
import prompt_analytics_assistant.backend.db as db

# Attempt to load SentenceTransformers and FAISS for semantic embeddings
SENTENCE_TRANSFORMERS_AVAILABLE = False
FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    FAISS_AVAILABLE = True
except ImportError:
    pass

# Global ML state
_ML_STATE = {
    "vectorizer": None,
    "classifier": None,
    "regressor": None,
    "df": None,
    "tfidf_vectors": None,
    "embedding_model": None,
    "faiss_index": None
}

def clean_text(text: str) -> str:
    """Preprocesses text for vectorization."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\?!\.]', '', text) # keep basic punctuation
    return text

def train_models():
    """Trains the TF-IDF vectorizer, quality classifier, regressor, and builds similarity index."""
    global _ML_STATE
    
    db.init_db()
    history = db.get_all_history()
    
    if not history:
        print("ML warning: No training data available.")
        return
        
    df = pd.DataFrame(history)
    for col in ["prompt", "quality", "output_tokens", "input_tokens", "model", "feedback"]:
        if col not in df.columns:
            df[col] = ""
            
    df["clean_prompt"] = df["prompt"].apply(clean_text)
    _ML_STATE["df"] = df
    
    # 1. TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    X_tfidf = vectorizer.fit_transform(df["clean_prompt"])
    _ML_STATE["vectorizer"] = vectorizer
    _ML_STATE["tfidf_vectors"] = X_tfidf
    
    # Quality classifier not used — rule-based classification is used instead
    _ML_STATE["classifier"] = None
        
    # 3. Output Token Regressor (Ridge Regression)
    is_pro = df["model"].apply(lambda m: 1.0 if "pro" in str(m).lower() else 0.0).values.reshape(-1, 1)
    input_tokens = df["input_tokens"].astype(float).values.reshape(-1, 1) / 100.0
    
    X_reg = np.hstack([X_tfidf.toarray(), is_pro, input_tokens])
    y_reg = df["output_tokens"].astype(float).values
    
    regressor = Ridge(alpha=1.0)
    regressor.fit(X_reg, y_reg)
    _ML_STATE["regressor"] = regressor
    
    # 4. Semantic Similarity Search Indexing
    if SENTENCE_TRANSFORMERS_AVAILABLE and FAISS_AVAILABLE:
        try:
            print("Semantic embeddings: Initializing SentenceTransformer model...")
            if _ML_STATE["embedding_model"] is None:
                # Load a very small, lightweight model
                _ML_STATE["embedding_model"] = SentenceTransformer("all-MiniLM-L6-v2")
            
            embeddings = _ML_STATE["embedding_model"].encode(df["prompt"].tolist())
            embeddings = np.array(embeddings).astype('float32')
            
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension) # Inner Product for Cosine Similarity (with normalized vectors)
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            _ML_STATE["faiss_index"] = index
            print("FAISS semantic index built successfully.")
        except Exception as e:
            print(f"Error building FAISS index: {e}. Falling back to TF-IDF.")
            _ML_STATE["faiss_index"] = None
    else:
        print("SentenceTransformers/FAISS not available. Using TF-IDF fallback for similarity search.")
        _ML_STATE["faiss_index"] = None

def rule_based_quality(prompt: str) -> tuple:
    """Classifies prompt quality using deterministic rules. Returns (quality, confidence)."""
    if not prompt or not prompt.strip():
        return "N/A", 0.0

    words = prompt.split()
    word_count = len(words)
    p = prompt.lower().strip()

    # Bad: single/two words
    if word_count <= 2:
        return "Bad", 0.95

    # Bad: contradictory or impossible constraints
    contradictions = [
        ("one word", ["explain", "describe", "write", "tell"]),
        ("don't explain", ["explain", "tell", "describe"]),
        ("no explanation", ["explain", "describe"]),
    ]
    for constraint, actions in contradictions:
        if constraint in p and any(a in p for a in actions):
            return "Bad", 0.92

    # Bad: vague/meaningless requests
    vague_exact = ["write something", "write something good", "tell me everything",
                   "explain everything", "give me all", "do something", "help me"]
    if any(p == v or p.startswith(v + " ") and word_count < 5 for v in vague_exact):
        return "Bad", 0.90

    # Bad: overly broad with no specific topic (e.g. "explain everything about computers")
    if any(w in p for w in ["everything about", "all about", "all information", "all the information"]):
        return "Bad", 0.88

    # Good: clear, specific, single-topic request
    # Has a concrete subject + action verb + reasonable length
    has_action = any(w in p for w in ["write", "explain", "create", "summarize", "analyze",
                                       "debug", "refactor", "generate", "find", "show",
                                       "describe", "calculate", "compare", "convert",
                                       "what is", "what are", "how does", "how do",
                                       "difference between", "advantages", "disadvantages"])
    is_question = any(p.startswith(w) for w in ["what", "where", "when", "who", "which",
                                                  "how", "why", "can", "is", "are", "do",
                                                  "does", "explain", "tell"]) or p.endswith("?")
    has_specific_topic = word_count >= 4

    if (has_action or is_question) and has_specific_topic:
        # Bonus signals push to Good
        good_signals = 0
        if "```" in prompt: good_signals += 2
        if any(c in prompt for c in ["- ", "* ", "1. "]): good_signals += 1
        if any(w in p for w in ["format", "json", "csv", "bullet", "markdown", "table", "list"]): good_signals += 1
        if any(w in p for w in ["limit", "words", "sentences", "brief", "concise", "short", "simple", "example"]): good_signals += 1
        if word_count >= 10: good_signals += 1

        if good_signals >= 1 or (4 <= word_count <= 12):
            return "Good", 0.85
        return "Average", 0.78

    # Average: has some structure but vague topic
    if word_count >= 6:
        return "Average", 0.70

    return "Bad", 0.75


def analyze_prompt(prompt: str, model_name: str, input_token_count: int) -> dict:
    """Analyzes a prompt and returns its predicted quality, token limits, and optimization recommendations."""
    global _ML_STATE
    
    if _ML_STATE["vectorizer"] is None:
        train_models()
        
    vectorizer = _ML_STATE["vectorizer"]
    classifier = _ML_STATE["classifier"]
    regressor = _ML_STATE["regressor"]
    df = _ML_STATE["df"]
    
    if vectorizer is None or df is None:
        # Fallback values if DB was completely empty and couldn't train
        return {
            "quality": "Average",
            "confidence": 0.5,
            "predicted_output_tokens": 344,  # DB mean output tokens
            "similar_prompts": [],
            "suggestions": generate_suggestions(prompt, "Average")
        }
        
    if not prompt or not prompt.strip() or input_token_count == 0:
        return {
            "quality": "N/A",
            "confidence": 0.0,
            "predicted_output_tokens": 0,
            "similar_prompts": [],
            "suggestions": ["Please enter a prompt to begin analysis."]
        }
        
    clean_p = clean_text(prompt)
    p_tfidf = vectorizer.transform([clean_p])
    
    # 1. Similarity Search
    similar_prompts = []
    
    if _ML_STATE["faiss_index"] is not None and _ML_STATE["embedding_model"] is not None:
        try:
            q_emb = _ML_STATE["embedding_model"].encode([prompt])
            q_emb = np.array(q_emb).astype('float32')
            faiss.normalize_L2(q_emb)
            
            # Query top k
            k = min(4, len(df))
            scores, indices = _ML_STATE["faiss_index"].search(q_emb, k)
            
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(df):
                    row = df.iloc[int(idx)]
                    # Avoid exact match overlap if user is currently typing it
                    if row["prompt"].strip() != prompt.strip() and len(similar_prompts) < 3:
                        similar_prompts.append({
                            "prompt": row["prompt"],
                            "model": row["model"],
                            "input_tokens": int(row["input_tokens"]),
                            "output_tokens": int(row["output_tokens"]),
                            "quality": row["quality"],
                            "feedback": row["feedback"] or "No feedback recorded.",
                            "similarity": float(round(score, 3))
                        })
        except Exception as e:
            print(f"FAISS search failed: {e}. Falling back to TF-IDF.")
            
    # Fallback to TF-IDF Similarity Search
    if not similar_prompts and _ML_STATE["tfidf_vectors"] is not None:
        sim_scores = cosine_similarity(p_tfidf, _ML_STATE["tfidf_vectors"])[0]
        top_indices = np.argsort(sim_scores)[::-1]
        
        for idx in top_indices:
            score = float(sim_scores[idx])
            if score > 0.02 and len(similar_prompts) < 3:
                row = df.iloc[int(idx)]
                if row["prompt"].strip() != prompt.strip():
                    similar_prompts.append({
                        "prompt": row["prompt"],
                        "model": row["model"],
                        "input_tokens": int(row["input_tokens"]),
                        "output_tokens": int(row["output_tokens"]),
                        "quality": row["quality"],
                        "feedback": row["feedback"] or "No feedback recorded.",
                        "similarity": round(score, 3)
                    })
                    
    # Rule-based quality classification (consistent, explainable)
    predicted_quality, confidence = rule_based_quality(prompt)
            
    # 3. Output Token Regression
    # Check if we have a highly similar prompt in our database.
    # If we have a >= 90% match, use its actual output tokens for maximum accuracy.
    # Use DB similar prompt output tokens if similarity >= 0.3, else use regressor
    best_match_tokens = None
    if similar_prompts:
        top_match = similar_prompts[0]
        if top_match["similarity"] >= 0.30:
            best_match_tokens = top_match["output_tokens"]

    if best_match_tokens is not None:
        predicted_output = best_match_tokens
    else:
        # Fallback: use mean output tokens from DB, or regressor
        db_mean_output = int(df["output_tokens"].astype(float).mean()) if df is not None and len(df) > 0 else 344
        predicted_output = db_mean_output
        if regressor is not None:
            try:
                is_pro = 1.0 if "pro" in model_name.lower() else 0.0
                in_tokens = float(input_token_count) / 100.0
                features_reg = np.hstack([p_tfidf.toarray(), [[is_pro]], [[in_tokens]]])
                pred_tokens = regressor.predict(features_reg)[0]
                predicted_output = max(10, int(pred_tokens))
            except Exception as e:
                print(f"Regression failed: {e}")
            
    # 4. Generate recommendations
    suggestions = generate_suggestions(prompt, predicted_quality)
    
    return {
        "quality": predicted_quality,
        "confidence": confidence,
        "predicted_output_tokens": predicted_output,
        "similar_prompts": similar_prompts,
        "suggestions": suggestions
    }

def generate_suggestions(prompt: str, quality: str) -> list:
    """Generates prompt engineering suggestions based on text structures."""
    suggestions = []
    
    if not prompt or not prompt.strip():
        return ["Please enter a prompt to begin analysis."]
        
    words = prompt.split()
    length = len(prompt)
    
    # Rule 1: Length heuristic
    if length < 20:
        suggestions.append("Prompt is short. Specify context, input details, and output requirements to reduce ambiguity.")
        
    # Rule 2: Explicit Action Verbs
    action_verbs = ["write", "explain", "create", "how", "why", "where", "what", "when", "who", "which", "summarize", "analyze", "debug", "refactor", "list", "format", "find", "show", "give", "tell", "describe"]
    has_action = any(verb in prompt.lower() for verb in action_verbs)
    if not has_action:
        suggestions.append("Include an action-oriented verb (e.g., 'Write a python script...', 'Summarize this report...') to set clear intent.")
        
    # Rule 3: Output Formatting Constraints
    format_indicators = ["table", "bullet", "list", "json", "markdown", "csv", "xml", "syntax", "format"]
    has_format = any(fmt in prompt.lower() for fmt in format_indicators)
    if not has_format:
        suggestions.append("Request a specific output format (e.g., bulleted list, JSON structure, or table) to keep response clean.")
        
    # Rule 4: Token Efficiency / Constraints
    constraint_indicators = ["limit", "words", "sentences", "paragraphs", "lines", "short", "brief", "concise"]
    has_constraints = any(c in prompt.lower() for c in constraint_indicators)
    
    if not has_constraints and ("summarize" in prompt.lower() or "explain" in prompt.lower() or length > 200):
        suggestions.append("Set a length boundary (e.g., 'in 2 sentences', 'under 150 words') to minimize unnecessary output tokens.")

    # Rule 5: Quality-based Fallbacks
    if quality == "Bad" and len(suggestions) < 2:
        suggestions.append("Paste the exact source code, trace logs, or inputs instead of generic descriptions like 'fix my code'.")
    elif quality == "Average" and not suggestions:
        suggestions.append("Add a 'few-shot' example demonstrating your desired format to guide the model's output style.")
        
    if not suggestions:
        suggestions.append("Prompt is well-formulated with action verbs, formatting requests, and constraints. Ready to send!")
        
    return suggestions
