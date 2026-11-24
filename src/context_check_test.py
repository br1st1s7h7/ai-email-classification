
import requests

subject = "Request for maternity leave policy"

body = "I am an employee planning my leave and would like a copy of Nexora's maternity leave policy, including the application procedure and eligibility requirements."
prompt = f"""
You are checking whether an email contains enough information
to route it to HR, ADMIN, LEGAL, or SALES.

Do not classify the email.

Check whether the sender's role, the type of benefit or request,
and the relevant business context are actually stated.

Do not assume that a benefit is an employee benefit.
Do not assume that a notice is a legal notice.

Answer these questions:
1. What facts are explicitly stated?
2. What essential information is missing?
3. Can a department be selected without guessing?

Subject: {subject}
Body: {body}
"""

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen2.5:7b",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0}
    },
    timeout=180
)

response.raise_for_status()

print(response.json()["response"])
