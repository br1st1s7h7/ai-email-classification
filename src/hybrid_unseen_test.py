
from hybrid_classifier import classify_email
from unseen_test_data import unseen_test_emails

results = []

for index, email in enumerate(unseen_test_emails, start=1):

    result = classify_email(
        email["subject"],
        email["body"]
    )

    results.append({
        "actual": email["actual"],
        "expected_review": email["expected_review"],
        **result
    })

    print(f"\nEmail {index}/{len(unseen_test_emails)}")
    print("Actual:", email["actual"])
    print("LR:", result["lr_prediction"])
    print("Qwen:", result["qwen_prediction"])
    print("Hybrid:", result["final_category"])
    print("Human Review:", result["needs_human_review"])
    print("Reason:", result["decision_reason"])
