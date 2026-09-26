import csv
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import numpy as np

# Load data
prompts, labels = [], []
with open('evaluation.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        if len(row) >= 5 and row[4] in ('Good', 'Average', 'Bad'):
            prompts.append(row[1])
            labels.append(row[4])

print(f"Dataset size: {len(prompts)}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    prompts, labels, test_size=0.2, random_state=42, stratify=labels
)

# TF-IDF
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_vec, y_train)
lr_cv = cross_val_score(lr, tfidf.transform(prompts), labels, cv=5).mean()
print(f"\nLogistic Regression CV Accuracy: {lr_cv:.3f}")
print(classification_report(y_test, lr.predict(X_test_vec)))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_vec, y_train)
rf_cv = cross_val_score(rf, tfidf.transform(prompts), labels, cv=5).mean()
print(f"Random Forest CV Accuracy: {rf_cv:.3f}")
print(classification_report(y_test, rf.predict(X_test_vec)))

# Save best model
best_model = lr if lr_cv >= rf_cv else rf
best_name = "LogisticRegression" if lr_cv >= rf_cv else "RandomForest"
print(f"\nBest model: {best_name} (CV: {max(lr_cv, rf_cv):.3f})")

with open('prompt_quality_model.pkl', 'wb') as f:
    pickle.dump({'model': best_model, 'tfidf': tfidf}, f)
print("Model saved to prompt_quality_model.pkl")
