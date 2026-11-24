
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

emails = []
labels = []

for file_path in DATA_DIR.glob("*.json"):
    category = file_path.stem.split("_")[0].upper()

    with open(file_path, "r", encoding="utf-8") as file:
        batch = json.load(file)

    for email in batch:
        subject = email.get("subject") or ""
        body = email.get("body") or ""

        emails.append(subject + " " + body)
        labels.append(category)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(emails, labels)

model_path = MODEL_DIR / "email_classifier.joblib"
joblib.dump(model, model_path)

print("Training emails:", len(emails))
print("Model saved to:", model_path)
