
import sqlite3
import os
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "smartllm.db")


def get_connection():
    return sqlite3.connect(DB_PATH)


def initialize_db():
    os.makedirs(BASE_DIR, exist_ok=True)

    conn = get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()
    finally:
        conn.close()


def save_cache_entry(question, answer, embedding):
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()

    initialize_db()

    conn = get_connection()
    try:
        cursor = conn.execute("""
            INSERT INTO cache_entries
            (question, answer, embedding, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            question,
            answer,
            json.dumps(embedding),
            datetime.now().isoformat()
        ))

        conn.commit()

        saved_id = cursor.lastrowid

        # Verify the row exists in this database
        check = conn.execute(
            "SELECT COUNT(*) FROM cache_entries WHERE id = ?",
            (saved_id,)
        ).fetchone()[0]

        if check != 1:
            raise RuntimeError("SQLite insert verification failed.")

        return saved_id

    finally:
        conn.close()


def load_cache_entries():
    initialize_db()

    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT question, answer, embedding, created_at
            FROM cache_entries
            ORDER BY id
        """).fetchall()

        return [
            {
                "question": row[0],
                "answer": row[1],
                "embedding": json.loads(row[2]),
                "created_at": row[3]
            }
            for row in rows
        ]
    finally:
        conn.close()


def get_cache_count():
    initialize_db()

    conn = get_connection()
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM cache_entries"
        ).fetchone()[0]
    finally:
        conn.close()


def clear_cache_database():
    initialize_db()

    conn = get_connection()
    try:
        conn.execute("DELETE FROM cache_entries")
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    test_id = save_cache_entry(
        "SQLITE_TEST_123",
        "Test answer",
        [0.1, 0.2]
    )

    print("Database path:", DB_PATH)
    print("Inserted row ID:", test_id)
    print("Count after save:", get_cache_count())

    conn = get_connection()
    try:
        conn.execute(
            "DELETE FROM cache_entries WHERE id = ?",
            (test_id,)
        )
        conn.commit()
    finally:
        conn.close()

    print("Count after cleanup:", get_cache_count())
