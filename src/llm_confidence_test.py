
import json
import requests
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from confidence_test_data import confidence_test_emails

categories = ["HR", "ADMIN", "LEGAL", "SALES"]

url = "http://localhost:11434/api/generate"

results = []

for i, email in enumerate(confidence_test_emails, start=1):

    prompt = f"""
You classify emails for Nexora, an electronics company.

Choose exactly one category based on the complete purpose
of the email.

HR:
Recruitment, interviews, employee records, attendance,
leave, payroll, employee grievances and employee benefits.

ADMIN:
Office operations, facilities, supplies, procurement,
vendor communication, courier services, office IT
and internal administration.

LEGAL:
Formal legal notices, lawyer correspondence, legal demands,
contract disputes, cheque dishonour, cease-and-desist
letters, regulatory compliance and legal claims.

SALES:
Customer enquiries, product quotations, product orders,
order tracking, delivery issues, returns, warranty claims,
damaged products and after-sales support.

PRIORITY RULES:
- Formal legal notices and legal demands are LEGAL.
- Customer product and order complaints are SALES.
- Ordinary employee matters are HR.
- Employee discount eligibility and benefits are HR.
- Customer product discounts and purchase enquiries are SALES.
- Reception forwarding or recording a complaint for
  administrative handling is ADMIN.
- Consider the complete purpose, not just keywords.
- Do not assume the sender is an employee unless the
  email establishes that relationship.
- Do not classify an email as LEGAL merely because it
  mentions further action.

CONFIDENCE AND HUMAN REVIEW RULES:

IMPORTANT: Category prediction and classification uncertainty
are two separate decisions.

First, identify the most likely department.
Then independently decide whether there is enough evidence
to classify the email reliably.

STEP 1: CHECK THE AVAILABLE EVIDENCE

Look for:
- The actual issue or request.
- The purpose of the email.
- The sender's relationship with Nexora, if relevant.
- Any information that distinguishes one department
  from another.

Do not invent missing information.

STEP 2: IDENTIFY CLASSIFICATION UNCERTAINTY

Set needs_human_review = true when:

1. The email does not explain the underlying issue
   or purpose.

2. Two or more departments are reasonably possible
   and the email does not provide enough evidence
   to distinguish between them.

3. The email refers to an attachment as its only source
   of information, but the attachment is unavailable.

4. The sender's relationship with Nexora is important
   for classification but cannot be established.

5. The email contains vague statements such as
   "Please take action", "Please look into this",
   or "We may take further action" without explaining
   the actual problem.

STEP 3: APPLY THESE DECISION RULES

A. ACCESS AND ONBOARDING

An email mentioning a recently joined employee who
cannot access a system may involve HR onboarding
or ADMIN/IT support.

If the email does not establish which department
is responsible, set needs_human_review = true.

Do not automatically classify every onboarding
or access issue as HR.

B. MISSING ATTACHMENTS

If the email only says to review an attached document
and the attachment is unavailable:

- Do not assume the document is legal.
- Do not assume it concerns office administration.
- Set needs_human_review = true.
- Do not claim that the document contains information
  that you cannot see.

C. VAGUE THREATS OR FURTHER ACTION

A statement such as "We may take further action"
does not independently establish a legal dispute.

If the underlying issue is not explained:

- Do not infer LEGAL.
- Do not infer HR.
- Set needs_human_review = true.

D. CLEAR DEPARTMENT, INCOMPLETE DETAILS

If the department is identifiable but some details
are missing, do not automatically request human review.

For example:
- A salary deduction issue is HR.
- A damaged product complaint is SALES.
- An office Wi-Fi problem is ADMIN.
- A formal legal notice is LEGAL.

Set needs_human_review = false when the department
is sufficiently clear.

STEP 4: ASSIGN CONFIDENCE

HIGH:
The email provides clear evidence for one department.

MEDIUM:
One department is more likely, but relevant context
is missing or another department is somewhat plausible.

LOW:
There is insufficient evidence to identify a department,
or multiple departments are equally plausible.

CONSISTENCY RULE:

- If needs_human_review = true, confidence cannot be HIGH.
- If needs_human_review = false, confidence can be
  HIGH or MEDIUM depending on the available evidence.
- Do not use HIGH confidence merely because you can
  produce a plausible category.
- Do not confuse the need for operational investigation
  with uncertainty about the department.

FINAL DECISION:

Always return one provisional category from HR, ADMIN,
LEGAL or SALES.

If the available information is insufficient, choose
the most plausible provisional category, but explicitly
flag the classification for human review.

The human-review decision must reflect uncertainty
about the department, not the urgency or complexity
of resolving the email.

Return valid JSON only:

{{
    "category": "HR",
    "reason": "Brief explanation",
    "confidence": "HIGH",
    "needs_human_review": false
}}

OUTPUT RULES:
- category must be HR, ADMIN, LEGAL or SALES.
- confidence must be HIGH, MEDIUM or LOW.
- needs_human_review must be true or false.
- Return JSON only without Markdown.

Subject:
{email["subject"]}

Body:
{email["body"]}
"""

    print(f"\nTesting email {i}/{len(confidence_test_emails)}...")

    try:
        response = requests.post(
            url,
            json={
                "model": "qwen2.5:7b",
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0}
            },
            timeout=180
        )

        response.raise_for_status()

        llm_response = response.json()["response"].strip()
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

        results.append({
            "subject": email["subject"],
            "actual": email["actual"],
            "predicted": prediction,
            "expected_review": email["expected_review"],
            "predicted_review": needs_human_review,
            "confidence": confidence,
            "reason": reason
        })

        print(f"Subject: {email['subject']}")
        print(f"Actual category: {email['actual']}")
        print(f"Predicted category: {prediction}")
        print(f"Confidence: {confidence}")
        print(f"Expected human review: {email['expected_review']}")
        print(f"Predicted human review: {needs_human_review}")
        print(f"Reason: {reason}")
        print("-" * 60)

    except (
        requests.RequestException,
        ValueError,
        KeyError,
        TypeError,
        AttributeError
    ) as error:
        print(f"Error on email {i}: {error}")


# ==========================================
# EVALUATION
# ==========================================

if results:

    # ======================================
    # 1. CATEGORY EVALUATION
    # ======================================

    valid_results = [
    r for r in results
    if r["actual"] is not None
    and str(r["actual"]).strip().upper() != "NONE"
]

    y_true = [r["actual"] for r in valid_results]
    y_pred = [r["predicted"] for r in valid_results]

    print("\n========== DEBUG CATEGORY CHECK ==========")

    print("Total model results:", len(results))
    print("Valid category results:", len(valid_results))

    print("Actual labels:", Counter(y_true))
    print("Predicted labels:", Counter(y_pred))

    correct_predictions = sum(
        actual == predicted
        for actual, predicted in zip(y_true, y_pred)
    )

    print("Correct predictions:", correct_predictions)
    print(
        "Incorrect predictions:",
        len(valid_results) - correct_predictions
    )

    print("\n========== CATEGORY EVALUATION ==========")

    print(
    f"Valid category predictions: "
    f"{len(valid_results)}/{len(results)}"
)

    if valid_results:

        print(
            f"Category accuracy: "
            f"{accuracy_score(y_true, y_pred) * 100:.2f}%"
        )

        print("\nClassification Report:")

        print(classification_report(
            y_true,
            y_pred,
            labels=categories,
            zero_division=0
        ))

        print("\nCategory Confusion Matrix:")

        print(confusion_matrix(
            y_true,
            y_pred,
            labels=categories
        ))

    else:
        print("No emails with known categories to evaluate.")

    # ======================================
    # 2. HUMAN REVIEW EVALUATION
    # ======================================

    review_true = [
        r["expected_review"] for r in results
    ]

    review_pred = [
        r["predicted_review"] for r in results
    ]

    print("\n========== HUMAN REVIEW EVALUATION ==========")

    print(
        f"Review decision accuracy: "
        f"{accuracy_score(review_true, review_pred) * 100:.2f}%"
    )

    print(
        f"Review precision: "
        f"{precision_score(review_true, review_pred, zero_division=0) * 100:.2f}%"
    )

    print(
        f"Review recall: "
        f"{recall_score(review_true, review_pred, zero_division=0) * 100:.2f}%"
    )

    print(
        f"Review F1-score: "
        f"{f1_score(review_true, review_pred, zero_division=0) * 100:.2f}%"
    )

    print("\nReview Confusion Matrix:")
    print("Order: [No Review, Human Review]")

    print(confusion_matrix(
        review_true,
        review_pred,
        labels=[False, True]
    ))

    # ======================================
    # 3. CONFIDENCE DISTRIBUTION
    # ======================================

    print("\n========== CONFIDENCE DISTRIBUTION ==========")

    print(Counter(
        r["confidence"] for r in results
    ))

    # ======================================
    # 4. HUMAN REVIEW DISTRIBUTION
    # ======================================

    print("\n========== HUMAN REVIEW DISTRIBUTION ==========")

    print(Counter(
        r["predicted_review"] for r in results
    ))

    # ======================================
    # 5. REVIEW DECISION MISMATCHES
    # ======================================

    print("\n========== REVIEW DECISION MISMATCHES ==========")

    mismatches = [
        r for r in results
        if r["expected_review"] != r["predicted_review"]
    ]

    if mismatches:

        for r in mismatches:

            print(f"\nSubject: {r['subject']}")
            print(f"Expected review: {r['expected_review']}")
            print(f"Predicted review: {r['predicted_review']}")
            print(f"Confidence: {r['confidence']}")
            print(f"Reason: {r['reason']}")

    else:
        print("No review decision mismatches.")

else:
    print("No valid predictions received.")
