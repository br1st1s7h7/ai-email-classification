
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

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

X_train, X_test, y_train, y_test = train_test_split(
    emails,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("Loading semantic embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

X_train_embeddings = embedding_model.encode(
    X_train,
    show_progress_bar=True
)

X_test_embeddings = embedding_model.encode(
    X_test,
    show_progress_bar=True
)

classifier = GaussianNB()
classifier.fit(X_train_embeddings, y_train)

predictions = classifier.predict(X_test_embeddings)

print("\nTotal emails:", len(emails))
print("Embedding dimensions:", X_train_embeddings.shape[1])
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))

print("\nClassification Report:")
print(classification_report(
    y_test, predictions, zero_division=0
))
