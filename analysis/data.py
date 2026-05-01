import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent.parent / "race_data.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def get_races(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM races ORDER BY year", conn)


def get_participants(
    conn: sqlite3.Connection,
    event_ids: list[int] | None = None,
    exclude_dnf: bool = False,
) -> pd.DataFrame:
    query = "SELECT * FROM participants"
    params: list = []
    conditions = []
    if event_ids:
        placeholders = ",".join("?" * len(event_ids))
        conditions.append(f"event_id IN ({placeholders})")
        params.extend(event_ids)
    if exclude_dnf:
        conditions.append("(status IS NULL OR status != '*')")
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    df = pd.read_sql_query(query, conn, params=params)
    df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")
    return df


def get_all_laps(
    conn: sqlite3.Connection,
    event_ids: list[int] | None = None,
    exclude_dnf: bool = True,
) -> pd.DataFrame:
    """
    Returns one row per lap joined with participant and race metadata.

    Columns: year, event_id, pid, name, gender, nationality, age, club,
             lap_number, split_time_sec, distance_km (lap cumulative),
             final_distance_km, status.

    Note: elapsed_time_sec in the DB is NULL for ~94% of rows due to a
    scraper parsing bug. Reconstruct elapsed as split_time_sec.cumsum()
    sorted by lap_number within each (event_id, pid) group.
    """
    dnf_filter = ""
    if exclude_dnf:
        dnf_filter = "AND (p.status IS NULL OR p.status != '*')"

    event_filter = ""
    params: list = []
    if event_ids:
        placeholders = ",".join("?" * len(event_ids))
        event_filter = f"AND l.event_id IN ({placeholders})"
        params.extend(event_ids)

    query = f"""
        SELECT
            r.year,
            l.event_id,
            l.pid,
            p.name,
            p.gender,
            p.nationality,
            p.age,
            p.club,
            p.status,
            l.lap_number,
            l.split_time_sec,
            l.distance_km,
            CAST(p.distance_km AS REAL) AS final_distance_km
        FROM laps l
        JOIN participants p ON l.event_id = p.event_id AND l.pid = p.pid
        JOIN races r ON l.event_id = r.event_id
        WHERE l.split_time_sec IS NOT NULL
        {dnf_filter}
        {event_filter}
        ORDER BY l.event_id, l.pid, l.lap_number
    """
    return pd.read_sql_query(query, conn, params=params)
