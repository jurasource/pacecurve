import sqlite3
import json
from pathlib import Path

DB_PATH = Path(__file__).parent / "race_data.db"

PARTICIPANT_COLS = [
    "bib", "rank", "name", "status", "gender",
    "nationality", "distance_km", "distance_miles", "age", "club",
]


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection):
    participant_col_defs = "\n".join(f"    {col}  TEXT," for col in PARTICIPANT_COLS)
    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS races (
            event_id    INTEGER PRIMARY KEY,
            year        INTEGER,
            date        TEXT,
            event_name  TEXT
        );

        CREATE TABLE IF NOT EXISTS participants (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id    INTEGER NOT NULL REFERENCES races(event_id),
            pid         TEXT NOT NULL,
            {participant_col_defs}
            raw_json    TEXT,
            UNIQUE(event_id, pid)
        );

        CREATE TABLE IF NOT EXISTS laps (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id            INTEGER NOT NULL REFERENCES races(event_id),
            pid                 TEXT NOT NULL,
            lap_number          INTEGER,
            distance_km         REAL,
            elapsed_time_sec    INTEGER,
            split_time_sec      INTEGER,
            raw_json            TEXT
        );
    """)
    conn.commit()


def _existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r["name"] for r in rows}


def ensure_participant_columns(conn: sqlite3.Connection, columns: list[str]):
    existing = _existing_columns(conn, "participants")
    for col in columns:
        if col not in existing:
            conn.execute(f'ALTER TABLE participants ADD COLUMN "{col}" TEXT')
    conn.commit()


def upsert_race(conn: sqlite3.Connection, event_id: int, year: int, date: str | None, event_name: str):
    conn.execute(
        "INSERT OR REPLACE INTO races (event_id, year, date, event_name) VALUES (?, ?, ?, ?)",
        (event_id, year, date, event_name),
    )
    conn.commit()


def upsert_participant(conn: sqlite3.Connection, event_id: int, pid: str, fields: dict):
    raw = json.dumps(fields)
    cols = list(fields.keys())
    vals = list(fields.values())
    col_list = ", ".join(f'"{c}"' for c in cols)
    placeholders = ", ".join("?" * len(vals))
    conn.execute(
        f"""
        INSERT INTO participants (event_id, pid, raw_json, {col_list})
        VALUES (?, ?, ?, {placeholders})
        ON CONFLICT(event_id, pid) DO UPDATE SET
            raw_json=excluded.raw_json,
            {", ".join(f'"{c}"=excluded."{c}"' for c in cols)}
        """,
        [event_id, pid, raw, *vals],
    )


def insert_laps(conn: sqlite3.Connection, event_id: int, pid: str, laps: list[dict]):
    conn.execute("DELETE FROM laps WHERE event_id=? AND pid=?", (event_id, pid))
    for lap in laps:
        conn.execute(
            "INSERT INTO laps (event_id, pid, lap_number, distance_km, elapsed_time_sec, split_time_sec, raw_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                lap["event_id"],
                lap["pid"],
                lap.get("lap_number"),
                lap.get("distance_km"),
                lap.get("elapsed_time_sec"),
                lap.get("split_time_sec"),
                lap.get("raw_json"),
            ),
        )
