import base64
import mimetypes
from email.message import EmailMessage

from gmail_connection import service
from email_routing_config import get_recipient


def forward_email(
    original_subject,
    original_body,
    original_sender,
    final_category,
    needs_human_review=False,
    attachments=None
):
    recipient = get_recipient(
        final_category,
        needs_human_review
    )

    if recipient is None:
        print("Email requires human review. Not forwarded.")
        return False

    message = EmailMessage()

    message["To"] = recipient
    message["Subject"] = f"Fwd: {original_subject}"

    message.set_content(
        f"Department: {final_category}\n\n"
        f"Original sender: {original_sender}\n"
        f"Original subject: {original_subject}\n\n"
        f"---------- Forwarded Message ----------\n"
        f"{original_body}"
    )

    # Attach files to the forwarded email.
    for attachment in attachments or []:

        filename = attachment["filename"]
        file_data = attachment["data"]

        content_type = attachment.get("mime_type")

        if not content_type:
            content_type = (
                mimetypes.guess_type(filename)[0]
                or "application/octet-stream"
            )

        main_type, sub_type = content_type.split("/", 1)

        message.add_attachment(
            file_data,
            maintype=main_type,
            subtype=sub_type,
            filename=filename
        )

    encoded_message = base64.urlsafe_b64encode(
        message.as_bytes()
    ).decode("utf-8")

    service.users().messages().send(
        userId="me",
        body={"raw": encoded_message}
    ).execute()

    print(f"Email forwarded to {recipient}")
    print(f"Attachments forwarded: {len(attachments or [])}")

    return True


if __name__ == "__main__":

    # Controlled test without attachments.
    forward_email(
        original_subject="Test Employee Leave Request",
        original_body="This is a test email for HR routing.",
        original_sender="nexora-test@example.com",
        final_category="HR"
    )