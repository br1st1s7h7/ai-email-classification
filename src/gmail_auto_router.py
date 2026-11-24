import sqlite3
import base64
from gmail_connection import service

from hybrid_classifier import classify_email
from gmail_forwarder import forward_email
from email_routing_config import get_recipient
from email_tracker import DB_PATH, create_database


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


def extract_attachments(payload, message_id):
    attachments = []

    def scan_part(part):
        filename = part.get("filename", "")
        body = part.get("body", {})
        mime_type = part.get(
            "mimeType", "application/octet-stream"
        )

        if filename:
            attachment_id = body.get("attachmentId")

            if attachment_id:
                attachment_response = (
                    service.users().messages().attachments().get(
                        userId="me",
                        messageId=message_id,
                        id=attachment_id
                    ).execute()
                )

                encoded_data = attachment_response.get("data")

            else:
                encoded_data = body.get("data")

            if encoded_data is None:
                raise ValueError(
                    f"Attachment data missing: {filename}"
                )

            file_data = base64.urlsafe_b64decode(
                encoded_data + "=" * (-len(encoded_data) % 4)
            )

            attachments.append({
                "filename": filename,
                "mime_type": mime_type,
                "data": file_data
            })

        for child in part.get("parts", []):
            scan_part(child)

    scan_part(payload)

    return attachments


create_database()


def reserve_email(message_id):
    with sqlite3.connect(DB_PATH) as conn:
        result = conn.execute(
            """
            INSERT OR IGNORE INTO processed_emails
            (message_id, department, recipient, status)
            VALUES (?, ?, ?, ?)
            """,
            (message_id, "PENDING", None, "PENDING")
        )
        return result.rowcount == 1


def update_status(message_id, department, recipient, status):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            UPDATE processed_emails
            SET department = ?, recipient = ?, status = ?
            WHERE message_id = ?
            """,
            (department, recipient, status, message_id)
        )


def process_inbox():
    response = service.users().messages().list(
        userId="me",
        labelIds=["INBOX"],
        maxResults=10
    ).execute()

    messages = response.get("messages", [])

    for item in messages:
        message_id = item["id"]

        if not reserve_email(message_id):
            print("Already tracked, skipping:", message_id)
            continue

        try:
            message = service.users().messages().get(
                userId="me",
                id=message_id,
                format="full"
            ).execute()

            headers = message["payload"].get("headers", [])

            def header(name):
                return next(
                    (
                        h["value"]
                        for h in headers
                        if h["name"].lower() == name.lower()
                    ),
                    ""
                )

            subject = header("Subject")
            sender = header("From")
            body = extract_body(message["payload"])

            if not body.strip():
                update_status(
                    message_id, "HUMAN_REVIEW", None, "REVIEW"
                )
                print("No readable body; human review:", message_id)
                continue

            result = classify_email(subject, body)

            category = result["final_category"]
            review = result["needs_human_review"]

            recipient = get_recipient(category, review)

            if recipient is None:
                update_status(
                    message_id, "HUMAN_REVIEW", None, "REVIEW"
                )
                print("Human review required:", message_id)
                continue

            attachments = extract_attachments(
                message["payload"],
                message_id
            )

            print(
                f"Attachments found: {len(attachments)} "
                f"for message {message_id}"
            )

            update_status(
                message_id, category, recipient, "SENDING"
            )

            sent = forward_email(
                original_subject=subject,
                original_body=body,
                original_sender=sender,
                final_category=category,
                needs_human_review=False,
                attachments=attachments
            )

            if sent:
                update_status(
                    message_id, category, recipient, "SENT"
                )
                print("Forwarded once:", message_id)

        except Exception as error:
            print("Processing error:", message_id, error)
            print("Held for manual verification; not retried.")


if __name__ == "__main__":
    process_inbox()