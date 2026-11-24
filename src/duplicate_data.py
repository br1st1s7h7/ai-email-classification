
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

seen_emails = {}
duplicate_count = 0
total_emails = 0

for file_path in sorted(DATA_DIR.glob("*.json")):
    with open(file_path, "r", encoding="utf-8") as file:
        emails = json.load(file)

    for index, email in enumerate(emails, start=1):
        total_emails += 1

        subject = " ".join(email.get("subject", "").lower().split())
        body = " ".join(email.get("body", "").lower().split())

        email_key = (subject, body)

        if email_key in seen_emails:
            duplicate_count += 1
            original_file, original_index = seen_emails[email_key]

            print(f"\nDuplicate email #{duplicate_count}")
            print(f"Original: {original_file}, email {original_index}")
            print(f"Repeated: {file_path.name}, email {index}")
            print(f"Subject: {email.get('subject', '')}")

        else:
            seen_emails[email_key] = (file_path.name, index)

print("\n--- Duplicate Check Summary ---")
print("Total emails checked:", total_emails)
print("Unique subject-body combinations:", len(seen_emails))
print("Duplicate occurrences:", duplicate_count)
