
import streamlit as st
import sqlite3
from email_tracker import DB_PATH
from gmail_connection import service
import base64

st.set_page_config(
    page_title="Email Dashboard",
    page_icon="📧",
    layout="wide"
)

st.title("📧 AI Email Classification")
st.caption("Department-wise Email Monitoring Dashboard")


def extract_body(payload):
    if payload.get("mimeType") == "text/plain":
        data = payload.get("body", {}).get("data")
        if data:
            return base64.urlsafe_b64decode(data).decode(
                "utf-8", errors="replace"
            )

    for part in payload.get("parts", []):
        body = extract_body(part)
        if body:
            return body

    return ""


def get_classified_emails():
    with sqlite3.connect(DB_PATH) as conn:
        records = conn.execute("""
            SELECT message_id, department, status
            FROM processed_emails
            WHERE department IN (
                'HR', 'ADMIN', 'SALES', 'LEGAL', 'HUMAN_REVIEW'
            )
            ORDER BY processed_at DESC
        """).fetchall()

    emails = []

    for message_id, department, status in records:
        try:
            message = service.users().messages().get(
                userId="me",
                id=message_id,
                format="full"
            ).execute()

            headers = message["payload"].get("headers", [])

            def get_header(name):
                return next(
                    (
                        h["value"]
                        for h in headers
                        if h["name"].lower() == name.lower()
                    ),
                    ""
                )

            emails.append({
                "department": department,
                "subject": get_header("Subject"),
                "sender": get_header("From"),
                "body": extract_body(message["payload"]),
                "status": status
            })

        except Exception as error:
            st.warning(f"Could not load email {message_id}: {error}")

    return emails


if st.button("🔄 Refresh Emails"):
    st.rerun()

with st.spinner("Loading classified emails..."):
    emails = get_classified_emails()

departments = ["HR", "ADMIN", "SALES", "LEGAL", "HUMAN_REVIEW"]

columns = st.columns(5)

for column, department in zip(columns, departments):
    department_emails = [
        email for email in emails
        if email["department"] == department
    ]

    with column:
        st.subheader(department.replace("_", " "))
        st.metric("Emails", len(department_emails))

        for email in department_emails:
            with st.expander(email["subject"] or "(No subject)"):
                st.write("**From:**", email["sender"])
                st.write("**Status:**", email["status"])
                st.write("**Message:**")
                st.text(email["body"] or "(No plain-text body)")
