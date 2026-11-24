
import json
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

category_counts = defaultdict(int)

for file_path in sorted(DATA_DIR.glob("*.json")):
    category = file_path.name.split("_")[0].upper()

    with open(file_path, "r", encoding="utf-8") as file:
        emails = json.load(file)

    category_counts[category] += len(emails)

print("--- Emails per Category ---")

for category, count in sorted(category_counts.items()):
    print(f"{category}: {count}")

print("\nTotal emails:", sum(category_counts.values()))
