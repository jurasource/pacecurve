import time
import requests
from config import BASE_URL, LIST_ID

DETAILS_TAB = "F846BF"
LAP_LIST_NAME = "Result Lists|Lap Details Online"
MAIN_LIST_NAME = f"Result Lists|Result OverAll"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
}

session = requests.Session()
session.headers.update(HEADERS)


def _referer(event_id: int) -> dict:
    return {"Referer": f"{BASE_URL}/{event_id}/"}


def get_config(event_id: int) -> dict:
    url = f"{BASE_URL}/{event_id}/results/config"
    r = session.get(url, params={"lang": "en"}, headers=_referer(event_id), timeout=30)
    r.raise_for_status()
    return r.json()


def get_participants(event_id: int, cfg: dict) -> dict:
    server = cfg["server"]
    key = cfg["key"]
    url = f"https://{server}/{event_id}/results/list"
    params = {
        "key": key,
        "listname": MAIN_LIST_NAME,
        "page": "results",
        "contest": "0",
        "r": "all",
        "lang": "en",
        "l": "0",
        "openedGroups": "{}",
        "term": "",
    }
    r = session.get(url, params=params, headers=_referer(event_id), timeout=30)
    r.raise_for_status()
    return r.json()


def get_participant_laps(event_id: int, cfg: dict, pid: str) -> dict:
    server = cfg["server"]
    key = cfg["key"]
    url = f"https://{server}/{event_id}/{DETAILS_TAB}/list"
    params = {
        "key": key,
        "listname": LAP_LIST_NAME,
        "page": DETAILS_TAB,
        "r": "pid",
        "pid": pid,
    }
    r = session.get(url, params=params, headers=_referer(event_id), timeout=30)
    r.raise_for_status()
    return r.json()


def get_participant_laps_with_delay(event_id: int, cfg: dict, pid: str, delay: float = 0.1) -> dict:
    result = get_participant_laps(event_id, cfg, pid)
    time.sleep(delay)
    return result
