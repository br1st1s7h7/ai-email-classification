
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

fields = [
    "sender_name",
    "sender_email",
    "subject",
    "body",
    "difficulty"
]

missing_counts = {field: 0 for field in fields}
total_emails = 0

for file_path in sorted(DATA_DIR.glob("*.json")):
    with open(file_path, "r", encoding="utf-8") as file:
        emails = json.load(file)

    for email in emails:
        total_emails += 1

        for field in fields:
            value = email.get(field)

            if value is None or (
                isinstance(value, str) and not value.strip()
            ):
                missing_counts[field] += 1
                print(f"Missing {field}: {file_path.name}")

print("\n--- Missing Values Summary ---")
print("Total emails:", total_emails)

for field, count in missing_counts.items():
    print(f"{field}: {count}")
