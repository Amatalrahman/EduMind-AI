"""
SQLite database layer: schema creation and CRUD helpers.
"""

import sqlite3
import contextlib
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import DB_PATH


# ── Connection helper ──────────────────────────────────────────────────────────

@contextlib.contextmanager
def get_conn():
    """Yield a thread-safe SQLite connection with WAL mode enabled."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema ─────────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS subjects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id  INTEGER NOT NULL REFERENCES subjects(id),
    filename    TEXT    NOT NULL,
    file_type   TEXT    NOT NULL,          -- 'pdf' or 'pptx'
    file_path   TEXT    NOT NULL,
    upload_date TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      INTEGER NOT NULL REFERENCES documents(id),
    subject_id  INTEGER NOT NULL REFERENCES subjects(id),
    chunk_index INTEGER NOT NULL,
    text        TEXT    NOT NULL,
    page_number INTEGER NOT NULL,
    chroma_id   TEXT    NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS image_chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      INTEGER NOT NULL REFERENCES documents(id),
    page_number INTEGER NOT NULL,
    image_index INTEGER NOT NULL,
    description TEXT    NOT NULL,
    chroma_id   TEXT    NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS quiz_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id  INTEGER NOT NULL REFERENCES subjects(id),
    doc_id      INTEGER,
    score       INTEGER NOT NULL,
    total       INTEGER NOT NULL,
    timestamp   TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS flashcards (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id  INTEGER NOT NULL REFERENCES subjects(id),
    doc_id      INTEGER,
    question    TEXT    NOT NULL,
    answer      TEXT    NOT NULL,
    source_page INTEGER
);

CREATE TABLE IF NOT EXISTS study_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id     INTEGER NOT NULL REFERENCES subjects(id),
    topic          TEXT    NOT NULL,
    times_reviewed INTEGER NOT NULL DEFAULT 0,
    quiz_accuracy  REAL    NOT NULL DEFAULT 0.0,
    last_seen      TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(subject_id, topic)
);
"""


def init_db() -> None:
    """Create all tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript(SCHEMA)


# ── Subjects ───────────────────────────────────────────────────────────────────

def create_subject(name: str) -> int:
    """Insert a new subject; return its id."""
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO subjects (name) VALUES (?) ON CONFLICT(name) DO NOTHING",
            (name,),
        )
        if cur.lastrowid:
            return cur.lastrowid
        row = conn.execute("SELECT id FROM subjects WHERE name = ?", (name,)).fetchone()
        return row["id"]


def get_all_subjects() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM subjects ORDER BY name").fetchall()
        return [dict(r) for r in rows]


def get_subject_by_name(name: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM subjects WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None


# ── Documents ──────────────────────────────────────────────────────────────────

def insert_document(subject_id: int, filename: str, file_type: str, file_path: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO documents (subject_id, filename, file_type, file_path) VALUES (?,?,?,?)",
            (subject_id, filename, file_type, file_path),
        )
        return cur.lastrowid


def get_documents_by_subject(subject_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE subject_id = ? ORDER BY upload_date DESC",
            (subject_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_documents() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT d.*, s.name AS subject_name
               FROM documents d
               JOIN subjects s ON s.id = d.subject_id
               ORDER BY d.upload_date DESC"""
        ).fetchall()
        return [dict(r) for r in rows]


def get_document_by_id(doc_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None


# ── Chunks ─────────────────────────────────────────────────────────────────────

def insert_chunk(
    doc_id: int,
    subject_id: int,
    chunk_index: int,
    text: str,
    page_number: int,
    chroma_id: str,
) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO chunks
               (doc_id, subject_id, chunk_index, text, page_number, chroma_id)
               VALUES (?,?,?,?,?,?)""",
            (doc_id, subject_id, chunk_index, text, page_number, chroma_id),
        )
        return cur.lastrowid


def insert_chunks_batch(rows: list[tuple]) -> None:
    """Bulk insert; each row = (doc_id, subject_id, chunk_index, text, page_number, chroma_id)."""
    with get_conn() as conn:
        conn.executemany(
            """INSERT OR IGNORE INTO chunks
               (doc_id, subject_id, chunk_index, text, page_number, chroma_id)
               VALUES (?,?,?,?,?,?)""",
            rows,
        )


def get_chunks_by_subject(subject_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM chunks WHERE subject_id = ? ORDER BY doc_id, chunk_index",
            (subject_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_chunks_by_document(doc_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
            (doc_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_chunk_by_chroma_id(chroma_id: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM chunks WHERE chroma_id = ?", (chroma_id,)
        ).fetchone()
        return dict(row) if row else None


# ── Image chunks ───────────────────────────────────────────────────────────────

def insert_image_chunk(
    doc_id: int, page_number: int, image_index: int, description: str, chroma_id: str
) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT OR IGNORE INTO image_chunks
               (doc_id, page_number, image_index, description, chroma_id)
               VALUES (?,?,?,?,?)""",
            (doc_id, page_number, image_index, description, chroma_id),
        )
        return cur.lastrowid


# ── Quiz results ───────────────────────────────────────────────────────────────

def insert_quiz_result(subject_id: int, score: int, total: int, doc_id: Optional[int] = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO quiz_results (subject_id, doc_id, score, total) VALUES (?,?,?,?)",
            (subject_id, doc_id, score, total),
        )
        return cur.lastrowid


# ── Flashcards ─────────────────────────────────────────────────────────────────

def insert_flashcard(
    subject_id: int, question: str, answer: str, source_page: Optional[int] = None, doc_id: Optional[int] = None
) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO flashcards (subject_id, doc_id, question, answer, source_page) VALUES (?,?,?,?,?)",
            (subject_id, doc_id, question, answer, source_page),
        )
        return cur.lastrowid


def get_flashcards_by_subject(subject_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM flashcards WHERE subject_id = ? ORDER BY id DESC",
            (subject_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# ── Study log (long-term memory) ───────────────────────────────────────────────

def upsert_study_log(subject_id: int, topic: str, quiz_accuracy: float) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO study_log (subject_id, topic, times_reviewed, quiz_accuracy, last_seen)
               VALUES (?, ?, 1, ?, datetime('now'))
               ON CONFLICT(subject_id, topic) DO UPDATE SET
                   times_reviewed = times_reviewed + 1,
                   quiz_accuracy  = (quiz_accuracy * times_reviewed + excluded.quiz_accuracy)
                                    / (times_reviewed + 1),
                   last_seen      = datetime('now')""",
            (subject_id, topic, quiz_accuracy),
        )


def get_study_log_by_subject(subject_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM study_log WHERE subject_id = ? ORDER BY quiz_accuracy ASC, last_seen ASC",
            (subject_id,),
        ).fetchall()
        return [dict(r) for r in rows]
