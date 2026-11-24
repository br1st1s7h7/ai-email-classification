
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "email_tracker.db"


def create_database():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_emails (
                message_id TEXT PRIMARY KEY,
                department TEXT NOT NULL,
                recipient TEXT,
                status TEXT NOT NULL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def is_processed(message_id):
    with sqlite3.connect(DB_PATH) as conn:
        result = conn.execute(
            "SELECT 1 FROM processed_emails WHERE message_id = ?",
            (message_id,)
        ).fetchone()

    return result is not None


def mark_processed(message_id, department, recipient, status):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO processed_emails
            (message_id, department, recipient, status)
            VALUES (?, ?, ?, ?)
        """, (message_id, department, recipient, status))


if __name__ == "__main__":
    create_database()
    print("Email tracking database created successfully!")

