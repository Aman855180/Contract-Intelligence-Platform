"""
sqlite_db.py

SQLite connection management and schema initialization for the Contract
Intelligence Platform. Per project scope, SQLite stores ONLY document
metadata, raw page text, and (in later phases) extracted structured
fields — embeddings/vectors are never stored here, they live exclusively
in ChromaDB.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.utils.config_loader import get_project_root, get_settings
from app.utils.logging_setup import get_logger

logger = get_logger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    document_id      TEXT PRIMARY KEY,
    contract_name    TEXT NOT NULL,
    document_type    TEXT NOT NULL CHECK (document_type IN ('pdf', 'docx')),
    upload_date      TEXT NOT NULL,
    page_count       INTEGER NOT NULL CHECK (page_count >= 1),
    file_size_bytes  INTEGER NOT NULL CHECK (file_size_bytes >= 0),
    source_path      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS document_pages (
    document_id  TEXT NOT NULL,
    page_number  INTEGER NOT NULL CHECK (page_number >= 1),
    text         TEXT NOT NULL,
    PRIMARY KEY (document_id, page_number),
    FOREIGN KEY (document_id) REFERENCES documents (document_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_document_pages_document_id
    ON document_pages (document_id);
"""


def get_db_path() -> Path:
    """Resolve the absolute SQLite database path from settings.yaml."""
    settings = get_settings()
    db_path_rel = settings["paths"]["sqlite_db_path"]
    return get_project_root() / db_path_rel


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """
    Context-managed SQLite connection with foreign keys and the configured
    journal mode/busy timeout enabled. Commits on clean exit, rolls back
    and re-raises on exception.

    Yields:
        An open sqlite3.Connection with row_factory set to sqlite3.Row.
    """
    settings = get_settings()
    db_cfg = settings.get("database", {})
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), timeout=db_cfg.get("busy_timeout_ms", 5000) / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    journal_mode = db_cfg.get("journal_mode", "WAL")
    conn.execute(f"PRAGMA journal_mode = {journal_mode};")

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        logger.exception("Database transaction rolled back due to an error.")
        raise
    finally:
        conn.close()


def init_db() -> None:
    """
    Create the documents and document_pages tables if they do not already
    exist. Safe to call on every application startup (idempotent).
    """
    with get_connection() as conn:
        conn.executescript(_SCHEMA_SQL)
    logger.info("SQLite schema initialized at %s.", get_db_path())
