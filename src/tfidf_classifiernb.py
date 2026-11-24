
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
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

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB()
}

for name, classifier in models.items():
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000
        )),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n", "=" * 45)
    print(name)
    print("=" * 45)
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print(classification_report(
        y_test, predictions, zero_division=0
    ))
