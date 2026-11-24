
import re

CATEGORY_RULES = {
    "HR": {
        "resume": 3,
        "cv": 3,
        "interview": 3,
        "recruitment": 3,
        "internship": 3,
        "candidate": 2,
        "joining": 2,
        "vacancy": 2,
        "ctc": 2,
        "job application": 3,
        "job offer": 3,
        "years of experience": 2,
        "employee termination": 4,
        "final settlement": 4,
        "employment contract": 2,
        "pending salary": 3,
        "employment documents": 3,
        "hr department": 2,
        "pending salary": 4,
        "final salary": 4,
        "employment settlement": 4,
        "experience letter": 3,
        "payroll team": 2,
        "after my resignation": 3,
        "salary":2
},

    "LEGAL": {
        "legal notice": 4,
        "court": 3,
        "advocate": 3,
        "summons": 3,
        "proceedings": 3,
        "hereby": 2,
        "filed": 2,
        "section": 2,
        "penalty": 2,
        "consumer court": 4,
        "legal action": 4,
        "breach of contract": 4
    },

    "ADMIN": {
        "invoice": 3,
        "gst": 3,
        "billing": 3,
        "subscription": 2,
        "renewal": 2,
        "office supplies": 3,
        "office maintenance": 3,
        "payment reminder": 2,
        "vendor registration": 3,
        "cctv": 2,
        "ssl certificate": 2,
        "unsubscribe": 1,
        "outstanding balance": 4,
        "overdue payment": 4,
        "payment immediately": 3,
        "clear this amount": 3,
        "balance for invoice": 4,
        "vendor invoice": 4,
        "unpaid invoice": 4,
        "payment date": 3,
        "transaction reference": 3,
        "payment reminders": 3,
    },

    "SALES": {
        "order": 3,
        "delivery": 3,
        "quotation": 3,
        "warranty": 3,
        "replacement": 3,
        "product": 2,
        "laptop": 2,
        "television": 2,
        "refrigerator": 2,
        "installation": 2,
        "order cancellation": 4,
        "product price": 3
    }
}


def classify_email(subject, body):
    text = f"{subject or ''} {body or ''}".lower()
    scores = {category: 0 for category in CATEGORY_RULES}

    for category, rules in CATEGORY_RULES.items():
        for phrase, weight in rules.items():
            pattern = r"\b" + re.escape(phrase) + r"\b"

            if re.search(pattern, text):
                scores[category] += weight

    predicted_category = max(scores, key=scores.get)

    if all(score == 0 for score in scores.values()):
        predicted_category = "UNKNOWN"

    return predicted_category, scores


if __name__ == "__main__":
    subject = input("Enter email subject: ")
    body = input("Enter email body: ")

    category, scores = classify_email(subject, body)

    print("\nCategory scores:")
    for name, score in scores.items():
        print(f"{name}: {score}")

    print("\nPredicted category:", category)
