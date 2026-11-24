
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

stop_words = {
    "the", "is", "a", "an", "and", "or", "to", "of", "in", "for",
    "on", "with", "this", "that", "it", "i", "we", "you", "your",
    "my", "our", "me", "as", "at", "be", "are", "was", "were",
    "have", "has", "had", "will", "would", "can", "could", "please",
    "hi", "hello", "dear", "regards", "thanks", "thank", "pune",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "mumbai", "delhi", "bengaluru", "hyderabad", "ahmedabad", "india",
    "am", "but", "if", "do", "nov", "oct", "dec", "jan", "apr",
    "feb", "aug", "sept", "sep", "at","nexora"
}

category_words = defaultdict(Counter)
category_email_counts = Counter()

for file_path in sorted(DATA_DIR.glob("*.json")):
    category = file_path.name.split("_")[0].upper()

    with open(file_path, "r", encoding="utf-8") as file:
        emails = json.load(file)

    category_email_counts[category] += len(emails)

    for email in emails:
        text = (
            str(email.get("subject") or "")
            + " "
            + str(email.get("body") or "")
        ).lower()

        words = re.findall(r"\b[a-z]{2,}\b", text)

        words = [
            word for word in words
            if word not in stop_words
        ]

        category_words[category].update(words)

print("--- Category-Specific Word Analysis ---")

for category in sorted(category_words):
    other_words = Counter()

    for other_category, counts in category_words.items():
        if other_category != category:
            other_words.update(counts)

    print(f"\n--- {category} ---")
    print("Emails:", category_email_counts[category])
    print("Most frequent words:")

    for word, count in category_words[category].most_common(15):
        print(f"{word}: {count}")

    print("\nPotentially distinctive words:")

    distinctive = []

    for word, count in category_words[category].most_common():
        other_count = other_words[word]

        if count >= 5 and count > other_count:
            distinctive.append((word, count, other_count))

    distinctive.sort(
        key=lambda item: item[1] / (item[2] + 1),
        reverse=True
    )

    for word, count, other_count in distinctive[:15]:
        print(
            f"{word}: category={count}, "
            f"other_categories={other_count}"
        )
