# Department Email Routing

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_DIR / ".env")

DEPARTMENT_EMAILS = {
    "HR": os.getenv("HR_EMAIL"),
    "ADMIN": os.getenv("ADMIN_EMAIL"),
    "LEGAL": os.getenv("LEGAL_EMAIL"),
    "SALES": os.getenv("SALES_EMAIL")
}

HUMAN_REVIEW_EMAIL = os.getenv("HUMAN_REVIEW_EMAIL")


def get_recipient(final_category, needs_human_review):
    if needs_human_review:
        return None

    return DEPARTMENT_EMAILS.get(final_category)


if __name__ == "__main__":
    print("Email Routing Configuration")

    for department in DEPARTMENT_EMAILS:
        recipient = get_recipient(department, False)

        if recipient:
            print(department, "-> Configured")
        else:
            print(department, "-> Missing configuration")

    print("HUMAN_REVIEW -> Keep in central inbox")