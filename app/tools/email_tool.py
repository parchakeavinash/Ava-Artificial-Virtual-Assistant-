import email
from email.header import decode_header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import imaplib
import smtplib

from app.config.settings import settings

SMTP_HOST = settings.SMTP_HOST
SMTP_PORT = settings.SMTP_PORT
IMAP_HOST = settings.IMAP_HOST
IMAP_PORT = settings.IMAP_PORT
EMAIL_USER = settings.ADMIN_EMAIL
EMAIL_PASSWORD = settings.EMAIL_PASSWORD


def send_email(to: str, subject: str, body: str, is_html: bool = False) -> str:
    """
    Send an email using SMTP.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Email text body.
        is_html: Whether body is HTML (default False).

    Returns:
        Success message or error.
    """
    if not all([SMTP_HOST, SMTP_PORT, EMAIL_USER, EMAIL_PASSWORD]):
        return "Error: SMTP_HOST, SMTP_PORT, EMAIL_USER, or EMAIL_PASSWORD is not configured."

    if "@" not in to or "." not in to.split("@")[-1]:
        return f"Error: Invalid recipient email address: {to}"

    msg = MIMEMultipart("alternative")
    if is_html:
        msg.attach(MIMEText(body, "html", "utf-8"))
    else:
        msg.attach(MIMEText(body, "plain", "utf-8"))

    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = to

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USER, [to], msg.as_string())
        return f"Email successfully sent to {to}."

    except smtplib.SMTPAuthenticationError:
        return "Error: SMTP authentication failed. Check your email or App Password."
    except smtplib.SMTPConnectError:
        return "Error: Could not connect to the SMTP server."
    except Exception as e:
        return f"Error: Failed to send email: {str(e)}"


def read_inbox(limit: int = 5, unread_only: bool = False) -> str:
    """
    Read recent emails from the Gmail inbox.

    Args:
        limit: Maximum number of emails to return (1-20).
        unread_only: If True, return only unread emails.

    Returns:
        Formatted summary of recent emails.
    """
    if not EMAIL_USER or not EMAIL_PASSWORD:
        return "Error: EMAIL_USER or EMAIL_PASSWORD is not configured."

    limit = max(1, min(limit, 20))

    try:
        with imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT) as imap:
            imap.login(EMAIL_USER, EMAIL_PASSWORD)
            status, _ = imap.select("INBOX")
            if status != "OK":
                return "Error: Could not open inbox."

            search_criteria = "UNSEEN" if unread_only else "ALL"
            status, data = imap.search(None, search_criteria)
            if status != "OK":
                return "Error: Failed to search inbox."

            mail_ids = data[0].split()
            if not mail_ids:
                return "No emails found in inbox."

            latest_ids = mail_ids[-limit:][::-1]
            results = []

            for mail_id in latest_ids:
                status, msg_data = imap.fetch(mail_id, "(RFC822)")
                if status != "OK":
                    continue

                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(raw_email)

                raw_subject = msg.get("Subject", "(No Subject)")
                decoded = decode_header(raw_subject)
                subject_parts = []
                for part, encoding in decoded:
                    if isinstance(part, bytes):
                        part = part.decode(encoding or "utf-8", errors="replace")
                    subject_parts.append(part)
                subject = "".join(subject_parts)

                sender = msg.get("From", "Unknown")
                date = msg.get("Date", "Unknown")

                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if "attachment" in str(part.get("Content-Disposition", "")):
                            continue
                        if part.get_content_type() == "text/plain":
                            payload = part.get_payload(decode=True)
                            if payload:
                                body = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
                                break
                else:
                    payload = msg.get_payload(decode=True)
                    if payload:
                        body = payload.decode(msg.get_content_charset() or "utf-8", errors="replace")

                body = body.strip()[:500]
                results.append(f"From: {sender}\nSubject: {subject}\nDate: {date}\nSummary: {body}")

            if not results:
                return "No readable emails found."

            return "\n\n---\n\n".join(results)

    except Exception as e:
        return f"Error: Failed to read inbox: {str(e)}"


def find_email_by_subject(subject_keyword: str) -> str:
    """
    Search the inbox for emails matching a subject keyword and return
    a confirmation prompt before deleting.

    Use this tool FIRST when the user asks to delete an email.
    After calling this, ask the user: "Are you sure you want to delete
    the email titled '<subject>'? Just say yes to confirm."

    Args:
        subject_keyword: Partial or full subject to search for.

    Returns:
        A confirmation message listing found emails, or an error.
    """
    if not EMAIL_USER or not EMAIL_PASSWORD:
        return "Error: EMAIL_USER or EMAIL_PASSWORD is not configured."

    try:
        with imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT) as imap:
            imap.login(EMAIL_USER, EMAIL_PASSWORD)
            status, _ = imap.select("INBOX")
            if status != "OK":
                return "Error: Could not open inbox."

            status, data = imap.search(None, "ALL")
            if status != "OK":
                return "Error: Failed to search inbox."

            mail_ids = data[0].split()
            if not mail_ids:
                return "Your inbox is empty."

            matches = []
            for mail_id in mail_ids[-50:][::-1]:
                status, msg_data = imap.fetch(mail_id, "(RFC822.HEADER)")
                if status != "OK":
                    continue

                raw = msg_data[0][1]
                msg = email.message_from_bytes(raw)

                raw_subject = msg.get("Subject", "")
                decoded = decode_header(raw_subject)
                subject_parts = []
                for part, enc in decoded:
                    if isinstance(part, bytes):
                        part = part.decode(enc or "utf-8", errors="replace")
                    subject_parts.append(part)
                full_subject = "".join(subject_parts)

                if subject_keyword.lower() in full_subject.lower():
                    sender = msg.get("From", "Unknown")
                    date = msg.get("Date", "Unknown")
                    matches.append(f'- Subject: "{full_subject}" | From: {sender} | Date: {date}')

            if not matches:
                return f'I could not find any emails with subject containing "{subject_keyword}".'

            match_list = "\n".join(matches)
            return (
                f"CONFIRMATION_REQUIRED\n"
                f"I found the following email(s) matching '{subject_keyword}':\n"
                f"{match_list}\n\n"
                f"Please confirm: say YES to delete, or NO to cancel."
            )

    except Exception as e:
        return f"Error: Failed to search inbox: {str(e)}"


def delete_email_confirmed(subject_keyword: str) -> str:
    """
    Permanently delete emails matching the subject keyword.

    IMPORTANT: Only call this tool AFTER the user has explicitly confirmed
    deletion (said 'yes', 'confirm', 'go ahead', 'delete it', etc.).
    Never call this without user confirmation.

    Args:
        subject_keyword: Partial or full subject of the email to delete.

    Returns:
        Confirmation message of deletion, or an error.
    """
    if not EMAIL_USER or not EMAIL_PASSWORD:
        return "Error: EMAIL_USER or EMAIL_PASSWORD is not configured."

    try:
        with imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT) as imap:
            imap.login(EMAIL_USER, EMAIL_PASSWORD)
            status, _ = imap.select("INBOX")
            if status != "OK":
                return "Error: Could not open inbox."

            status, data = imap.search(None, "ALL")
            if status != "OK":
                return "Error: Failed to search inbox."

            mail_ids = data[0].split()
            if not mail_ids:
                return "Your inbox is empty — nothing to delete."

            deleted_subjects = []

            for mail_id in mail_ids[-50:][::-1]:
                status, msg_data = imap.fetch(mail_id, "(RFC822.HEADER)")
                if status != "OK":
                    continue

                raw = msg_data[0][1]
                msg = email.message_from_bytes(raw)

                raw_subject = msg.get("Subject", "")
                decoded = decode_header(raw_subject)
                subject_parts = []
                for part, enc in decoded:
                    if isinstance(part, bytes):
                        part = part.decode(enc or "utf-8", errors="replace")
                    subject_parts.append(part)
                full_subject = "".join(subject_parts)

                if subject_keyword.lower() in full_subject.lower():
                    imap.store(mail_id, "+FLAGS", "\\Deleted")
                    deleted_subjects.append(f'"{full_subject}"')

            if not deleted_subjects:
                return f'No emails found matching "{subject_keyword}". Nothing was deleted.'

            imap.expunge()
            deleted_list = ", ".join(deleted_subjects)
            return f"Done! I've permanently deleted {len(deleted_subjects)} email(s): {deleted_list}."

    except Exception as e:
        return f"Error: Failed to delete email: {str(e)}"
