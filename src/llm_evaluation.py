
import json
import random
import re
import requests
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

CATEGORIES = ["ADMIN", "HR", "LEGAL", "SALES"]
MODEL_NAME = "qwen2.5:7b"
TEST_PER_CATEGORY = 10
RANDOM_SEED = 42

category_records = {category: [] for category in CATEGORIES}

for file_path in sorted(DATA_DIR.glob("*.json")):
    category = file_path.stem.split("_")[0].upper()

    if category not in CATEGORIES:
        continue

    with open(file_path, "r", encoding="utf-8") as file:
        records = json.load(file)

    for email in records:
        subject = email.get("subject") or ""
        body = email.get("body") or ""

        category_records[category].append({
            "subject": subject,
            "body": body,
            "actual": category
        })

def text_key(email):
    return (
        re.sub(r"\s+", " ", email["subject"]).strip().casefold(),
        re.sub(r"\s+", " ", email["body"]).strip().casefold()
    )

development_keys = set()

for category in CATEGORIES:
    for email in category_records[category][:5]:
        development_keys.add(text_key(email))

grouped = {}

for category in CATEGORIES:
    for email in category_records[category]:
        key = text_key(email)
        grouped.setdefault(key, []).append(email)

unique_emails = []
conflicting_keys = set()

for key, group in grouped.items():
    labels = {email["actual"] for email in group}

    if len(labels) > 1:
        conflicting_keys.add(key)
        continue

    unique_emails.append(group[0])

candidates = [
    email for email in unique_emails
    if text_key(email) not in development_keys
]

rng = random.Random(RANDOM_SEED)
test_emails = []

for category in CATEGORIES:
    category_candidates = [
        email for email in candidates
        if email["actual"] == category
    ]

    if len(category_candidates) < TEST_PER_CATEGORY:
        raise ValueError(
            f"Not enough fresh emails for {category}: "
            f"{len(category_candidates)} available."
        )

    test_emails.extend(
        rng.sample(category_candidates, TEST_PER_CATEGORY)
    )

test_keys = {text_key(email) for email in test_emails}

train_emails = [
    email for email in unique_emails
    if text_key(email) not in development_keys
    and text_key(email) not in test_keys
]

def email_text(email):
    return (
        f"Subject: {email['subject']}\n"
        f"Body: {email['body']}"
    )

baseline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

X_train = [email_text(email) for email in train_emails]
y_train = [email["actual"] for email in train_emails]
X_test = [email_text(email) for email in test_emails]
y_test = [email["actual"] for email in test_emails]

baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)

prompt_template = """
You classify emails for Nexora, an electronics company.

Choose exactly one category based on the email's purpose.

HR:
Job applications, recruitment, interviews, employee records,
attendance, leave, payroll, and ordinary employee matters.

ADMIN:
Office operations, facilities, office supplies, procurement,
vendor offers, courier services, packaging supplies,
business IT services, and internal administration.

LEGAL:
Formal legal notices, lawyer correspondence, legal demands,
contract disputes, cheque dishonour notices, cease-and-desist
letters, regulatory compliance, and legal claims.

SALES:
Customer enquiries, product quotations, product orders,
order tracking, delivery issues, returns, warranty claims,
damaged products, faulty products, and after-sales support.

Priority rules:
- A formal legal notice or legal demand is LEGAL, even if it
  concerns salary, invoices, or another department.
- A customer reporting a fault with a product they purchased,
  especially when an order number is present, is SALES.
- Customer complaints or support requests about Nexora products
  or orders are SALES, not ADMIN.
- Vendor offers for Nexora are generally ADMIN.
- Ordinary recruitment and employee matters are HR.

Consider the complete email's purpose, not just keywords.
Return exactly one category: HR, ADMIN, LEGAL, or SALES.

{email}
"""

session = requests.Session()
llm_predictions = []

for i, email in enumerate(test_emails, start=1):
    prompt = prompt_template.format(email=email_text(email))

    response = session.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0}
        },
        timeout=300
    )
    response.raise_for_status()

    answer = response.json()["response"].strip().upper()
    match = re.search(r"\b(HR|ADMIN|LEGAL|SALES)\b", answer)

    if not match:
        raise ValueError(
            f"Unrecognized answer for email {i}: {answer}"
        )

    prediction = match.group(1)
    llm_predictions.append(prediction)

    print(f"Email {i}/40 | {email['actual']} -> {prediction}")

print("\nTRAINING EMAILS:", len(train_emails))
print("TEST EMAILS:", len(test_emails))
print("CONFLICTING TEXT GROUPS EXCLUDED:", len(conflicting_keys))

for model_name, predictions in [
    ("TF-IDF + Logistic Regression", baseline_predictions),
    ("Qwen2.5 7B", llm_predictions)
]:
    print(f"\n{'=' * 55}")
    print(model_name)
    print(f"{'=' * 55}")
    print(
        f"Accuracy: "
        f"{accuracy_score(y_test, predictions) * 100:.2f}%"
    )
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=CATEGORIES,
        zero_division=0
    ))
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(CATEGORIES)
    print(confusion_matrix(
        y_test,
        predictions,
        labels=CATEGORIES
    ))
