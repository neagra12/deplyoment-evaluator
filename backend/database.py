import sqlite3
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

DB_PATH = os.getenv("DB_PATH", "../data/prox.db")


def get_connection() -> sqlite3.Connection:
    path = Path(__file__).parent.parent / DB_PATH.lstrip("../")
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS manuals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            filename        TEXT    NOT NULL,
            title           TEXT,
            equipment_type  TEXT,
            total_pages     INTEGER,
            ingested_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
            checksum        TEXT    UNIQUE
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            manual_id      INTEGER NOT NULL REFERENCES manuals(id),
            page_number    INTEGER NOT NULL,
            chunk_index    INTEGER NOT NULL,
            text           TEXT    NOT NULL,
            embedding      BLOB,
            section_header TEXT,
            token_count    INTEGER
        );

        CREATE TABLE IF NOT EXISTS eval_questions (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            manual_id             INTEGER NOT NULL REFERENCES manuals(id),
            category              TEXT    NOT NULL,
            question_text         TEXT    NOT NULL,
            ground_truth_answer   TEXT    NOT NULL,
            ground_truth_page     INTEGER,
            ground_truth_chunk_id INTEGER REFERENCES chunks(id),
            generated_at          DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS simulated_dealerships (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            name           TEXT    NOT NULL,
            skew_category  TEXT    NOT NULL,
            skew_weight    REAL    NOT NULL,
            question_count INTEGER NOT NULL,
            created_at     DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS eval_runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            manual_id       INTEGER NOT NULL REFERENCES manuals(id),
            label           TEXT,
            dealership_id   INTEGER REFERENCES simulated_dealerships(id),
            started_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
            completed_at    DATETIME,
            total_questions INTEGER,
            attempted       INTEGER,
            summary_json    TEXT
        );

        CREATE TABLE IF NOT EXISTS eval_results (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id            INTEGER NOT NULL REFERENCES eval_runs(id),
            question_id       INTEGER NOT NULL REFERENCES eval_questions(id),
            agent_answer      TEXT,
            escalated         BOOLEAN DEFAULT 0,
            escalation_reason TEXT,
            cited_page        INTEGER,
            confidence_score  REAL,
            retrieval_correct BOOLEAN,
            citation_correct  BOOLEAN,
            hallucination_flag BOOLEAN,
            groundedness_score REAL,
            latency_ms        INTEGER
        );

        CREATE TABLE IF NOT EXISTS simulated_dealership_questions (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            dealership_id  INTEGER NOT NULL REFERENCES simulated_dealerships(id),
            question_id    INTEGER NOT NULL REFERENCES eval_questions(id),
            run_id         INTEGER REFERENCES eval_runs(id)
        );
    """)
    conn.commit()
    conn.close()
