import re
import requests
from bs4 import BeautifulSoup
from config import EVENTRAC_URL, OVERRIDES


def get_races() -> list[dict]:
    resp = requests.get(EVENTRAC_URL, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    races = []
    for row in soup.select("table tr, .results-list li, [class*=result] [class*=row]"):
        text = row.get_text(" ", strip=True)
        year_match = re.search(r"\b(20\d{2})\b", text)
        if not year_match:
            continue
        year = int(year_match.group(1))

        date_match = re.search(r"\b(\d{2}/\d{2}/20\d{2})\b", text)
        date = date_match.group(1) if date_match else None

        link = row.find("a", href=re.compile(r"raceresult\.com/\d+"))
        if link:
            event_id_match = re.search(r"raceresult\.com/(\d+)", link["href"])
            event_id = int(event_id_match.group(1)) if event_id_match else None
        else:
            event_id = None

        if year in OVERRIDES:
            event_id = OVERRIDES[year]

        if event_id:
            races.append({"year": year, "date": date, "event_id": event_id})

    # Deduplicate by event_id, keeping first occurrence
    seen = set()
    unique = []
    for r in races:
        if r["event_id"] not in seen:
            seen.add(r["event_id"])
            unique.append(r)

    # Ensure override years that weren't on the page at all are included
    for year, event_id in OVERRIDES.items():
        if event_id not in seen:
            unique.append({"year": year, "date": None, "event_id": event_id})

    return sorted(unique, key=lambda r: r["year"])
