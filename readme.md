Build a production-ready desktop overlay application called "Prompt Analytics Assistant" for Antigravity CLI.

This MUST NOT be a web application.

The application should run in the background and monitor text being typed inside the Antigravity CLI in real time. As the user types, display a lightweight floating popup beside the terminal that updates continuously without interrupting typing.

The popup should include:
• Estimated input tokens
• Predicted worst-case output tokens
• Total estimated tokens
• Optional estimated API cost
• Prompt quality classification (Good, Average, Bad)
• Confidence score
• Similar prompts retrieved from a historical dataset
• Feedback from matching prompts
• Suggestions to improve prompt efficiency

Use a dataset containing:
- Model
- Prompt
- Input Tokens
- Output Tokens
- Prompt Quality
- Feedback

Implement:
1. Data preprocessing pipeline
2. Token prediction model
3. Prompt quality classifier
4. Semantic similarity search using embeddings
5. Recommendation engine
6. Desktop overlay UI
7. Integration with Antigravity CLI
8. Continuous learning by storing new prompts and retraining periodically

Recommended stack:
- Python (FastAPI backend)
- Electron or Tauri desktop application
- Sentence Transformers
- FAISS
- SQLite
- tiktoken
- Scikit-learn or XGBoost

The architecture should be modular, scalable, and production-ready. Each component should be independently testable, with clear APIs and documentation. Prioritize low latency, low resource usage, and cross-platform compatibility.
