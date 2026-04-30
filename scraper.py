"""
Race result scraper — downloads all participants and lap times for every edition
of the Self-Transcendence 24 Hour Track Race from raceresult.com.

Usage:
    python scraper.py
"""

import json
import re
from tqdm import tqdm

import db
import raceresult
from eventrac import get_races

# ---------------------------------------------------------------------------
# Participant list parser
# ---------------------------------------------------------------------------
# DataFields returned by the /results/list endpoint:
#   ['BIB', 'ID', '[OverallRankp]', 'DisplayName', 'if([STATUS]>0;"*";)',
#    'GenderMF', 'NATION.IOCNAME', 'RunnerKM', 'RunnerMiles', 'AGE', 'CLUB']

PARTICIPANT_FIELD_MAP = {
    "BIB":             "bib",
    "ID":              "pid",
    "[OverallRankp]":  "rank",
    "DisplayName":     "name",
    'if([STATUS]>0;"*";)': "status",
    "GenderMF":        "gender",
    "NATION.IOCNAME":  "nationality",
    "RunnerKM":        "distance_km",
    "RunnerMiles":     "distance_miles",
    "AGE":             "age",
    "CLUB":            "club",
}


def parse_participants(raw: dict) -> list[dict]:
    fields = raw.get("DataFields", [])
    rows = raw.get("data", [])
    participants = []
    for row in rows:
        record = {}
        for i, field in enumerate(fields):
            col = PARTICIPANT_FIELD_MAP.get(field, field)
            record[col] = row[i] if i < len(row) else None
        if record.get("pid"):
            participants.append(record)
    return participants


# ---------------------------------------------------------------------------
# Lap parser
# ---------------------------------------------------------------------------
# DataFields returned by the /F846BF/list endpoint:
#   ['BIB', 'ID', '{n}', '[Start_Lap.Read{n}Text]', '[Start_Lap.Lap{n}Text]',
#    'trim(({n}*[LapCentimeter]-[LapOffset])/100000) & " km"']
# Indices:  0      1      2          3                     4                    5
# Meaning: bib   pid  lap_num  elapsed_time          split_time           distance

LAP_FIELD_NAMES = ["bib", "pid", "lap_number", "elapsed_time", "split_time", "distance_raw"]


def parse_laps(raw: dict, event_id: int, pid: str) -> list[dict]:
    rows = raw.get("data", [])
    laps = []
    for row in rows:
        record = dict(zip(LAP_FIELD_NAMES, row))
        laps.append({
            "event_id":        event_id,
            "pid":             pid,
            "lap_number":      _to_int(record.get("lap_number")),
            "elapsed_time_sec": _parse_time(record.get("elapsed_time")),
            "split_time_sec":  _parse_time(record.get("split_time")),
            "distance_km":     _parse_distance(record.get("distance_raw")),
            "raw_json":        json.dumps(record),
        })
    return laps


def _to_int(val) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _parse_distance(s: str | None) -> float | None:
    if not s:
        return None
    m = re.search(r"([\d.,]+)", str(s))
    if m:
        try:
            return float(m.group(1).replace(",", "."))
        except ValueError:
            pass
    return None


def _parse_time(s: str | None) -> int | None:
    """Convert 'HH:MM:SS', 'MM:SS.ss', 'MM:SS' to whole seconds."""
    if not s:
        return None
    s = str(s).strip()
    # HH:MM:SS
    m = re.match(r"^(\d+):(\d{2}):(\d{2})$", s)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
    # MM:SS.ss or MM:SS
    m = re.match(r"^(\d+):(\d{2})(?:\.\d+)?$", s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    conn = db.get_conn()
    db.init_db(conn)

    races = get_races()
    print(f"Found {len(races)} races from eventrac")

    for race in races:
        event_id = race["event_id"]
        year = race["year"]
        print(f"\n=== {year} (event {event_id}) ===")

        try:
            cfg = raceresult.get_config(event_id)
        except Exception as e:
            print(f"  [ERROR] config failed: {e}")
            continue

        event_name = cfg.get("eventname", str(year))
        db.upsert_race(conn, event_id, year, race.get("date"), event_name)
        print(f"  Event: {event_name}")

        try:
            raw_list = raceresult.get_participants(event_id, cfg)
        except Exception as e:
            print(f"  [ERROR] participant list failed: {e}")
            continue

        participants = parse_participants(raw_list)
        print(f"  Participants: {len(participants)}")
        if not participants:
            continue

        # Ensure participant columns exist for any extra fields
        extra_cols = [k for k in participants[0] if k not in ("pid", "raw_json")]
        db.ensure_participant_columns(conn, extra_cols)

        for p in participants:
            pid = p.pop("pid")
            db.upsert_participant(conn, event_id, pid, p)
        conn.commit()

        # Re-build list of pids after saving
        pids = [p.get("pid") or str(p.get("ID") or p.get("id") or "") for p in parse_participants(raw_list)]
        pids = [pid for pid in pids if pid]

        lap_total = 0
        with tqdm(pids, desc=f"  Laps {year}", unit="p") as bar:
            for pid in bar:
                try:
                    raw_laps = raceresult.get_participant_laps_with_delay(event_id, cfg, pid)
                except Exception as e:
                    tqdm.write(f"  [ERROR] laps pid={pid}: {e}")
                    continue
                laps = parse_laps(raw_laps, event_id, pid)
                if laps:
                    db.insert_laps(conn, event_id, pid, laps)
                    lap_total += len(laps)
        conn.commit()
        print(f"  Laps stored: {lap_total}")

    print("\n=== Summary ===")
    for row in conn.execute(
        "SELECT r.year, r.event_id, COUNT(DISTINCT p.pid) as n "
        "FROM races r LEFT JOIN participants p ON r.event_id=p.event_id "
        "GROUP BY r.year ORDER BY r.year"
    ).fetchall():
        n_laps = conn.execute(
            "SELECT COUNT(*) FROM laps WHERE event_id=?", (row["event_id"],)
        ).fetchone()[0]
        print(f"  {row['year']}: {row['n']} participants, {n_laps} lap rows")

    conn.close()


if __name__ == "__main__":
    run()
