
import json
import requests
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

categories = ["HR", "ADMIN", "LEGAL", "SALES"]
test_emails = []

# Load test emails
for category in categories:
    category_emails = []

    for file_path in sorted(DATA_DIR.glob("*.json")):
        if file_path.stem.split("_")[0].upper() != category:
            continue

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        for email in data:
            category_emails.append({
                "sender_name": email.get("sender_name", ""),
                "sender_email": email.get("sender_email", ""),
                "subject": email.get("subject", ""),
                "body": email.get("body", ""),
                "attachment_names": email.get("attachment_names", []),
                "actual_category": category
            })

    test_emails.extend(category_emails[:5])


url = "http://localhost:11434/api/generate"
results = []

# Classify emails
for i, email in enumerate(test_emails, start=1):

    prompt = f"""
You classify emails for Nexora, an electronics company.

Choose exactly one category based on the email's purpose.

HR:
Job applications, recruitment, interviews, employee records,
attendance, leave, payroll questions, and employee matters
that are not formal legal claims.

ADMIN:
Office operations, facilities, office supplies, procurement,
vendor offers, courier services, packaging supplies,
IT services for business operations, and internal administration.

LEGAL:
Formal legal notices, lawyer correspondence, legal demands,
contract disputes, cheque dishonour notices, cease-and-desist
letters, regulatory compliance, and legal claims.

SALES:
Customer enquiries, product quotations, product orders,
order tracking, delivery issues, returns, warranty claims,
damaged products, faulty products, and after-sales support
for Nexora products.
A customer reporting a fault with a product they purchased,
especially when an order number is present, is SALES.

Priority rules:
- A formal legal notice or legal demand is LEGAL,
  even when it concerns salary, invoices, or another department.
- A customer complaint or support request about a Nexora
  product or order is SALES, not ADMIN.
- A vendor offering goods or services to Nexora is generally ADMIN.
- An ordinary employee or recruitment matter is HR.
- Consider the complete email's purpose, not just keywords.

Return your answer in this JSON format only:

{{
    "category": "HR",
    "reason": "Brief explanation for the classification",
    "confidence": "HIGH",
    "needs_human_review": false
}}

Rules:
- category must be HR, ADMIN, LEGAL, or SALES.
- reason must briefly explain the email's purpose.
- confidence must be HIGH, MEDIUM, or LOW.
- needs_human_review must be true or false.
- Return valid JSON only.
- Do not include Markdown code fences.

Sender name:
{email["sender_name"]}

Sender email:
{email["sender_email"]}

Subject:
{email["subject"]}

Body:
{email["body"]}

Attachment filenames:
{email["attachment_names"]}
"""

    response = requests.post(
        url,
        json={
            "model": "qwen2.5:7b",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0}
        },
        timeout=180
    )
    response.raise_for_status()

    llm_response = response.json()["response"].strip()

    # Validate JSON response
    try:
        prediction_data = json.loads(llm_response)

        prediction = prediction_data["category"].upper()
        reason = prediction_data["reason"]
        confidence = prediction_data["confidence"].upper()
        needs_human_review = prediction_data["needs_human_review"]

        if prediction not in categories:
            raise ValueError("Invalid category")

        if confidence not in ["HIGH", "MEDIUM", "LOW"]:
            raise ValueError("Invalid confidence")

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Invalid reason")

        if not isinstance(needs_human_review, bool):
            raise ValueError("Invalid review value")

    except (json.JSONDecodeError, KeyError, AttributeError, ValueError) as error:
        print("Invalid LLM response:", llm_response)
        print("Error:", error)
        continue

    results.append({
        "actual": email["actual_category"],
        "predicted": prediction,
        "reason": reason,
        "confidence": confidence,
        "needs_human_review": needs_human_review
    })

    print(f"Email {i}")
    print(f"Subject: {email['subject']}")
    print(f"Actual: {email['actual_category']}")
    print(f"Predicted: {prediction}")
    print(f"Reason: {reason}")
    print(f"Confidence: {confidence}")
    print(f"Human review: {needs_human_review}")
    print()


# Evaluation
if results:
    correct = sum(
        result["actual"] == result["predicted"]
        for result in results
    )

    print(f"Correct: {correct}/{len(results)}")
    print(f"Accuracy: {correct / len(results) * 100:.2f}%")
else:
    print("No valid predictions were received.")
