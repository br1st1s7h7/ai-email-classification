
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

records = []

for file_path in sorted(DATA_DIR.glob("*.json")):
    category = file_path.name.split("_")[0].upper()

    with open(file_path, "r", encoding="utf-8") as file:
        emails = json.load(file)

    for email in emails:
        subject = email.get("subject") or ""
        body = email.get("body") or ""

        records.append({
            "category": category,
            "subject_length": len(subject),
            "body_length": len(body),
            "total_length": len(subject) + len(body)
        })

df = pd.DataFrame(records)

summary = df.groupby("category").agg(
    email_count=("total_length", "count"),
    avg_subject_length=("subject_length", "mean"),
    median_subject_length=("subject_length", "median"),
    avg_body_length=("body_length", "mean"),
    median_body_length=("body_length", "median"),
    avg_total_length=("total_length", "mean")
).round(2)

print("--- Email Length Analysis ---")
print(summary.to_string())
