import requests
import json
import joblib
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "tfidf_lr_pipeline.pkl"
OLLAMA_URL = "http://localhost:11434/api/generate"

model = joblib.load(MODEL_PATH)

SALARY_TERMS = (
    "salary", "salaries", "unpaid wages", "wages", "payroll",
    "payslip", "pay slip", "unpaid salary", "salary slip"
)
COURT_TERMS = (
    "court", "litigation", "advocate", "lawyer", "attorney",
    "legal counsel", "legal representative", "suit", "petition"
)


def is_employee_salary_grievance(text):
    text = text.lower()
    has_salary = any(term in text for term in SALARY_TERMS)
    has_court = any(term in text for term in COURT_TERMS)
    return has_salary and not has_court


ADMIN_FACILITY_TERMS = (
    "air conditioning", "air conditioner", "ac repair",
    "ceiling light", "office maintenance", "meeting room",
    "office premises", "office facilities",
    "office equipment", "facility maintenance"
)

INTERNAL_CONTEXT_TERMS = (
    "our office", "our meeting room", "our premises",
    "internal office", "our workplace",
    "in the office", "office building", "office maintenance",
)

CUSTOMER_CONTEXT_TERMS = (
    "customer", "client requesting", "purchase",
    "quotation", "product warranty", "after-sales"
)


def is_internal_admin_request(text):
    text = text.lower()

    has_facility_issue = any(
        term in text for term in ADMIN_FACILITY_TERMS
    )

    has_internal_context = any(
        term in text for term in INTERNAL_CONTEXT_TERMS
    )

    has_customer_context = any(
        term in text for term in CUSTOMER_CONTEXT_TERMS
    )

    return (
        has_facility_issue
        and has_internal_context
        and not has_customer_context
    )


def classify_email(subject, body):

    email_text = subject + " " + body
    lr_prediction = model.predict([email_text])[0]

    prompt = f"""
You are an email classification assistant for a company

Classify the email into one department:
HR, ADMIN, LEGAL, or SALES.

Department rules:
HR: Employee salaries, leave, attendance, recruitment, benefits,
and employee discount eligibility.

ADMIN: Internal office maintenance, procurement,
vendor quotations, supplier proposals, office equipment,
furniture, stationery, IT equipment, facilities,
courier, and internal business services.
Emails from external vendors offering their products or
services for the company's internal office operations
must be classified as ADMIN.

LEGAL: Formal legal notices, lawyer demands, legal disputes,
and regulatory compliance.

SALES: Customer orders, product enquiries, deliveries,
returns, warranty, and after-sales support.
Emails from customers who want to purchase the company's
products or services must be classified as SALES.

IMPORTANT CLASSIFICATION RULES:

1. Emails about employee salaries, unpaid wages, recruitment,
   payroll, payslips, leave, or employee benefits
   must be classified as HR.

2. Formal language such as "legal notice",
   "formal demand", or "written confirmation"
   does not automatically make an employee
   grievance a LEGAL email.

3. Classify emails as LEGAL when they involve
   court proceedings, litigation documents,
   or formal notices from legal representatives.

4. If the email involves an employee grievance
   with unclear legal escalation, request
   HR.

5. Prioritize the underlying purpose of the
   email over individual keywords.

6. A vendor offering products or services to the company
   for internal business use belongs to ADMIN.

7. A customer requesting products or services
   from the company belongs to SALES.

8. Do not classify an email as SALES simply because
   it contains words such as quotation, price,
   purchase, supplier, vendor, or product.
n
9. Determine whether company is the buyer or seller.
   If this cannot be determined, request HUMAN REVIEW.

10. If the sender contradicts themselves about the
    nature of the request, request HUMAN REVIEW.

Important:
- Classify based on the email's actual purpose.
- Formal legal demands take priority over ordinary HR or SALES issues.
- Vendor quotations for company's internal office needs belong to ADMIN.
- Do not assume missing details.
- Do not infer the sender's role or the type of request without evidence.
- Generic words such as benefit, eligibility, notice, response,
  request, issue, and approval do not establish a department by themselves.
- If essential context is missing, set needs_human_review to true.
- If multiple departments are plausible and the email does not clearly
  distinguish between them, request human review.
- When human review is required, use category NEEDS_HUMAN_REVIEW,
  confidence LOW, and needs_human_review true.
- Do not force a department prediction when the email is ambiguous.
- HIGH confidence must not be used when human review is needed.

Return valid JSON only with:
category, reason, confidence, needs_human_review.

Subject: {subject}
Body: {body}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
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
        llm_result = json.loads(response.json()["response"])

        if not isinstance(llm_result, dict):
            raise ValueError("Qwen response is not a dictionary")

        required_fields = {
            "category",
            "reason",
            "confidence",
            "needs_human_review"
        }

        if not required_fields.issubset(llm_result):
            raise ValueError("Qwen response is missing required fields")

    except (requests.RequestException, ValueError, KeyError, TypeError) as error:
        return {
            "lr_prediction": str(lr_prediction).upper(),
            "qwen_prediction": None,
            "qwen_reason": str(error),
            "qwen_confidence": None,
            "final_category": None,
            "needs_human_review": True,
            "decision_reason": "Qwen request or response failed"
        }

    lr_category = str(lr_prediction).upper()
    qwen_category = str(llm_result["category"]).upper()
    qwen_confidence = str(llm_result["confidence"]).upper()

    valid_categories = {"HR", "ADMIN", "LEGAL", "SALES"}
    qwen_review = llm_result["needs_human_review"]

    if qwen_category == "NEEDS_HUMAN_REVIEW" and isinstance(qwen_review, bool):
        final_category = None
        needs_human_review = True
        decision_reason = "Qwen identified an ambiguous email"

    elif (
        qwen_category not in valid_categories
        or not isinstance(qwen_review, bool)
    ):
        final_category = None
        needs_human_review = True
        decision_reason = "Invalid Qwen response"

    elif qwen_review:
        final_category = None
        needs_human_review = True
        decision_reason = "Qwen requested human review"

    elif (
        lr_category == "LEGAL"
        and qwen_category == "HR"
        and qwen_confidence == "HIGH"
        and is_employee_salary_grievance(email_text)
    ):
        final_category = "HR"
        needs_human_review = False
        decision_reason = "HR salary grievance: Qwen override"

    elif (
        lr_category == "SALES"
        and qwen_category == "ADMIN"
        and qwen_confidence == "HIGH"
        and is_internal_admin_request(email_text)
    ):
        final_category = "ADMIN"
        needs_human_review = False
        decision_reason = "Internal office facilities: Qwen override"

    elif (
        qwen_confidence == "HIGH"
        and lr_category != qwen_category
    ):
        final_category = qwen_category
        needs_human_review = False
        decision_reason = "Qwen override: HIGH confidence"

    elif lr_category != qwen_category:
        final_category = None
        needs_human_review = True
        decision_reason = "Models disagree"

    else:
        final_category = lr_category
        needs_human_review = False
        decision_reason = "Both models agree"

    return {
        "lr_prediction": lr_category,
        "qwen_prediction": qwen_category,
        "qwen_reason": llm_result["reason"],
        "qwen_confidence": llm_result["confidence"],
        "final_category": final_category,
        "needs_human_review": needs_human_review,
        "decision_reason": decision_reason
    }


if __name__ == "__main__":

    from unseen_test_data import unseen_test_emails

    email = unseen_test_emails[33]

    result = classify_email(
        email["subject"],
        email["body"]
    )

    print("Logistic Regression:", result["lr_prediction"])
    print("Qwen:", result["qwen_prediction"])
    print("Final Category:", result["final_category"])
    print("Human Review:", result["needs_human_review"])
    print("Decision Reason:", result["decision_reason"])