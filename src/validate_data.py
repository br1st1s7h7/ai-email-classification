
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

files = [
    "hr_batch_1.json",
    "hr_batch_2.json",
    "hr_batch_3.json",
    "hr_batch_4.json",
    "hr_batch_5.json",
    "legal_batch_1.json",
    "legal_batch_2.json",
    "legal_batch_3.json",
    "legal_batch_4.json",
    "legal_batch_5.json",
    "admin_batch_1.json",
    "admin_batch_2.json",
    "admin_batch_3.json",
    "admin_batch_4.json",
    "admin_batch_5.json",
    "sales_batch_1.json",
    "sales_batch_2.json",
    "sales_batch_3.json",
    "sales_batch_4.json",
    "sales_batch_5.json"
]

required_fields = [
    "sender_name",
    "sender_email",
    "subject",
    "body",
    "attachment_names",
    "difficulty"
]

for filename in files:
    file_path = DATA_DIR / filename

    print(f"\nChecking: {filename}")

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            emails = json.load(file)
    except FileNotFoundError:
        print("File not found.")
        continue
    except json.JSONDecodeError:
        print("Invalid JSON format.")
        continue

    if not isinstance(emails, list):
        print("Error: Expected a list of emails.")
        continue

    print("Total emails:", len(emails))

    for index, email in enumerate(emails, start=1):
        if not isinstance(email, dict):
            print(f"Email {index}: Invalid record format.")
            continue

        missing_fields = [
            field for field in required_fields
            if field not in email
        ]

        empty_fields = [
            field for field in required_fields
            if field in email and (
                email[field] is None
                or email[field] == ""
                or email[field] == []
            )
        ]

        if missing_fields:
            print(f"Email {index}: Missing fields: {missing_fields}")

        if empty_fields:
            print(f"Email {index}: Empty fields: {empty_fields}")

    print("Validation finished.")
