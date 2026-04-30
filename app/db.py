import sqlite3
from pathlib import Path

DB_PATH = "data/rag_learn.db"


def _connect() -> sqlite3.Connection:
    Path("data").mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # rows behave like dicts
    return conn


def init_db():
    """Create tables if they don't exist. Called once on startup."""
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            question  TEXT NOT NULL,
            answer    TEXT NOT NULL,
            intent    TEXT,
            cached    INTEGER DEFAULT 0,
            created   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS eval_results (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            question           TEXT NOT NULL,
            answer             TEXT NOT NULL,
            faithfulness       REAL,
            answer_relevancy   REAL,
            context_precision  REAL,
            created            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()


def save_chat(question: str, answer: str, intent: str = "", cached: bool = False):
    conn = _connect()
    conn.execute(
        "INSERT INTO chat_history (question, answer, intent, cached) VALUES (?,?,?,?)",
        (question, answer, intent, int(cached))
    )
    conn.commit()
    conn.close()


def save_eval_result(question: str, answer: str, scores: dict):
    conn = _connect()
    conn.execute(
        """INSERT INTO eval_results
           (question, answer, faithfulness, answer_relevancy, context_precision)
           VALUES (?,?,?,?,?)""",
        (
            question, answer,
            scores.get("faithfulness"),
            scores.get("answer_relevancy"),
            scores.get("context_precision"),
        )
    )
    conn.commit()
    conn.close()


def get_history(limit: int = 50) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM chat_history ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_eval_history(limit: int = 100) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM eval_results ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
