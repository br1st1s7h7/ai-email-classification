
import json
import requests
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter

categories = ["HR", "ADMIN", "LEGAL", "SALES"]

challenge_emails = [
    # HR
    {
        "subject": "Laptop replacement request during notice period",
        "body": "I am serving my notice period and my company laptop has stopped working. Please arrange a replacement so I can complete my handover.",
        "actual": "HR"
    },
    {
        "subject": "Salary not credited - please resolve",
        "body": "My monthly salary has not been credited. Please check with payroll and update me.",
        "actual": "HR"
    },
    {
        "subject": "Interview postponed due to legal appointment",
        "body": "I have a personal court appointment and cannot attend the scheduled interview. Can we reschedule?",
        "actual": "HR"
    },
    {
        "subject": "Employee attendance correction",
        "body": "My attendance for Monday is marked absent even though I was present in the office. Please correct the record.",
        "actual": "HR"
    },
    {
        "subject": "Grievance regarding manager behaviour",
        "body": "I want to formally report repeated inappropriate behaviour by my reporting manager. Please guide me through the employee grievance process.",
        "actual": "HR"
    },

    # ADMIN
    {
        "subject": "Legal agreement for office cleaning services",
        "body": "Please review and sign the attached service agreement for our office cleaning vendor before the renewal date.",
        "actual": "ADMIN"
    },
    {
        "subject": "Customer complaint received at reception",
        "body": "A visitor has submitted a complaint about a damaged Nexora television. Please forward it to the appropriate customer support team.",
        "actual": "ADMIN"
    },
    {
        "subject": "Office router not working",
        "body": "The office Wi-Fi router has stopped working. Employees on the second floor cannot access the company network. Please arrange IT support.",
        "actual": "ADMIN"
    },
    {
        "subject": "Vendor payment overdue - reminder",
        "body": "Our payment for the office stationery supply invoice is overdue. Kindly confirm the expected payment date.",
        "actual": "ADMIN"
    },
    {
        "subject": "Courier delivery delayed for office documents",
        "body": "The courier containing internal office documents has not arrived. Please coordinate with the delivery provider.",
        "actual": "ADMIN"
    },

    # LEGAL
    {
        "subject": "Employee salary dispute - legal notice attached",
        "body": "We represent a former employee and hereby issue a formal legal notice demanding unpaid salary and compensation within seven days.",
        "actual": "LEGAL"
    },
    {
        "subject": "Customer threatens to file a case",
        "body": "I purchased a Nexora laptop and it stopped working. I am requesting a replacement under warranty. If this is not resolved, I may take legal action.",
        "actual": "SALES"
    },
    {
        "subject": "Contract termination dispute",
        "body": "Our legal counsel disputes the termination of the service agreement and demands compliance with the contractual terms.",
        "actual": "LEGAL"
    },
    {
        "subject": "Cheque dishonour notice from supplier",
        "body": "A formal notice has been issued under Section 138 of the Negotiable Instruments Act regarding a dishonoured payment cheque.",
        "actual": "LEGAL"
    },
    {
        "subject": "Trademark infringement allegation",
        "body": "Our lawyers have identified unauthorised use of our registered trademark and demand that the use stop immediately.",
        "actual": "LEGAL"
    },

    # SALES
    {
        "subject": "Legal action if replacement is not provided",
        "body": "My Nexora washing machine is defective. I have contacted support twice. Please replace it under warranty or I will take legal action.",
        "actual": "SALES"
    },
    {
        "subject": "Employee discount on Nexora laptop",
        "body": "I am a Nexora employee and want to know whether I can purchase a laptop using the employee discount.",
        "actual": "HR"
    },
    {
        "subject": "Refund not received for cancelled order",
        "body": "My order was cancelled last week but the refund has not reached my bank account. Please check the transaction.",
        "actual": "SALES"
    },
    {
        "subject": "Product quotation for 200 laptops",
        "body": "Our company wants to purchase 200 Nexora laptops. Please share the bulk pricing and delivery timeline.",
        "actual": "SALES"
    },
    {
        "subject": "Laptop overheating after purchase",
        "body": "The Nexora laptop I purchased last month becomes extremely hot during normal use. Please help me arrange a warranty inspection.",
        "actual": "SALES"
    }
]


url = "http://localhost:11434/api/generate"
results = []


for i, email in enumerate(challenge_emails, start=1):

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

Priority rules:
- A formal legal notice or legal demand is LEGAL.
- A customer complaint or support request about a Nexora
  product or order is SALES, even if the customer threatens legal action.
- Vendor offers are generally ADMIN.
- Ordinary employee and recruitment matters are HR.
- Consider the complete purpose of the email.

EMPLOYEE DISCOUNT RULE:
- Questions from employees about eligibility, terms, or benefits
  of employee discounts on Nexora products must be classified as HR.
- Do not classify these emails as SALES merely because they
  mention Nexora products or prices.
- Customer discounts, product pricing, and purchase enquiries
  must be classified as SALES.

RECEPTION FORWARDING RULE:
- If reception or an office employee is only forwarding,
  recording, or routing a customer complaint as an
  administrative communication, classify it as ADMIN.
- If the email requests investigation, repair, replacement,
  refund, warranty support, or other resolution of a Nexora
  product or order issue, classify it as SALES.
- Do not classify an email based only on the presence of
  a customer complaint or product-related keywords.

  HUMAN REVIEW RULES:

The purpose of needs_human_review is ONLY to identify
classification uncertainty.

Set needs_human_review = false when:
- The email clearly belongs to one department.
- The email requires investigation or follow-up, but
  its department is identifiable.
- A legal specialist must review a legal document.
- An employee salary, leave, attendance, or benefit
  issue clearly belongs to HR.
- A customer order, refund, warranty, or product issue
  clearly belongs to SALES.
- An office facility, equipment, courier, or supply
  issue clearly belongs to ADMIN.

Set needs_human_review = true when:
- The email contains insufficient information to identify
  a department.
- Two or more departments are equally plausible.
- The email is too vague to determine its purpose.
- Important context is missing and cannot be inferred
  from the subject or body.
- The email refers to an attachment for the only relevant
  information, but that information is unavailable.

IMPORTANT:
Do not request human review merely because an email
requires a person to investigate, approve, or resolve it.

Choose the most likely category when possible, but flag
genuine classification uncertainty for human review.

Do not automatically assign LEGAL just because an email
mentions taking further action or making a complaint.

Return valid JSON only:

{{
    "category": "HR",
    "reason": "Brief explanation",
    "confidence": "HIGH",
    "needs_human_review": false
}}

Rules:
- category must be HR, ADMIN, LEGAL, or SALES.
- confidence must be HIGH, MEDIUM, or LOW.
- needs_human_review must be true or false.
- Return JSON only without Markdown.

Subject:
{email["subject"]}

Body:
{email["body"]}
"""

    print(f"Testing email {i}/20...")

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
            "actual": email["actual"],
            "predicted": prediction,
            "reason": reason,
            "confidence": confidence,
            "needs_human_review": needs_human_review,
            "subject": email["subject"]
        })

        print(f"Subject: {email['subject']}")
        print(f"Actual: {email['actual']}")
        print(f"Predicted: {prediction}")
        print(f"Reason: {reason}")
        print(f"Confidence: {confidence}")
        print(f"Human review: {needs_human_review}")
        print("-" * 60)

    except (requests.RequestException, ValueError, KeyError, TypeError) as error:
        print(f"Error on email {i}: {error}")
        continue


# Evaluation
if results:

    y_true = [result["actual"] for result in results]
    y_pred = [result["predicted"] for result in results]

    accuracy = accuracy_score(y_true, y_pred)

    print("\n========== EVALUATION RESULTS ==========")

    print(f"Valid predictions: {len(results)}/20")
    print(f"Correct predictions: {sum(a == b for a, b in zip(y_true, y_pred))}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=categories,
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(
        y_true,
        y_pred,
        labels=categories
    ))

    print("\nConfidence Distribution:")
    print(Counter(result["confidence"] for result in results))

    print("\nHuman Review Distribution:")
    print(Counter(
        result["needs_human_review"] for result in results
    ))

    print("\nIncorrect Predictions:")

    for result in results:
        if result["actual"] != result["predicted"]:
            print(f"\nSubject: {result['subject']}")
            print(f"Actual: {result['actual']}")
            print(f"Predicted: {result['predicted']}")
            print(f"Reason: {result['reason']}")

else:
    print("No valid predictions received.")
