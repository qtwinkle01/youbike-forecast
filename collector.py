"""Minimal YouBike + weather collector.

Every INTERVAL_SECONDS (default 300) it fetches:
  - YouBike Taipei (one JSON)
  - YouBike New Taipei (paged JSON, one file per page)
  - Open-Meteo forecast + current conditions (every WEATHER_INTERVAL_SECONDS)

Raw response bytes are stored untouched (gzip-compressed) under
  DATA_DIR/raw/<source>/YYYY/MM/DD/<UTC timestamp>[_pN].json.gz
The timestamp in the file name is OUR fetch time (UTC), not the API's.
Every attempt (success or failure) is appended to DATA_DIR/fetch_log.jsonl.
"""
import gzip
import json
import logging
import os
import signal
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import urllib3.util.connection as urllib3_cn

# Force IPv4: IPv6 / default routing to api.open-meteo.com was flaky on this network.
urllib3_cn.allowed_gai_family = lambda: socket.AF_INET

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
INTERVAL = int(os.environ.get("INTERVAL_SECONDS", "300"))
WEATHER_INTERVAL = int(os.environ.get("WEATHER_INTERVAL_SECONDS", "3600"))
RUN_ONCE = os.environ.get("RUN_ONCE") == "1"  # for testing
TIMEOUT = 30
RETRIES = 3
WEATHER_TIMEOUT = 10
WEATHER_RETRIES = 2
WEATHER_RETRY_AFTER = 600  # seconds, after a failed weather round
NTPC_PAGE_SIZE = 1000
NTPC_MAX_PAGES = 20  # safety stop

TAIPEI_URL = "https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json"
NTPC_URL = "https://data.ntpc.gov.tw/api/datasets/010e5b15-3823-4b20-b401-b1cf000550c5/json"
METEO_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_POINTS = {
    "weather_taipei": (25.0478, 121.5319),
    "weather_newtaipei": (25.0143, 121.4672),
}
WEATHER_PARAMS = {
    "current": "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,wind_speed_10m",
    "hourly": "temperature_2m,precipitation_probability,precipitation,weather_code,wind_speed_10m",
    "forecast_days": 2,
    "timezone": "Asia/Taipei",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logging.Formatter.converter = time.gmtime
log = logging.getLogger("collector")

session = requests.Session()
session.headers["User-Agent"] = "youbike-collector/0.1"
stop = threading.Event()
log_lock = threading.Lock()


def utcnow():
    return datetime.now(timezone.utc)


def fetch(url, params=None, timeout=TIMEOUT, retries=RETRIES):
    """GET with retries + backoff. Returns raw bytes. Raises after final failure."""
    last = None
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.content
        except Exception as e:  # noqa: BLE001
            last = e
            log.warning("fetch %s attempt %d/%d failed: %s", url, attempt, retries, e)
            if attempt < retries and not stop.wait(2 ** attempt * 2):
                continue
            break
    raise last


def save_raw(source, ts, body, suffix=""):
    d = DATA_DIR / "raw" / source / ts.strftime("%Y/%m/%d")
    d.mkdir(parents=True, exist_ok=True)
    name = ts.strftime("%Y-%m-%dT%H-%M-%SZ") + suffix + ".json.gz"
    final = d / name
    tmp = d / (name + ".tmp")
    with gzip.open(tmp, "wb") as f:
        f.write(body)
    os.replace(tmp, final)  # atomic: no half-written files
    return final


def count_rows(body):
    try:
        obj = json.loads(body)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(obj, list):
        return len(obj)
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list):
                return len(v)
    return None


def log_event(event):
    with log_lock, open(DATA_DIR / "fetch_log.jsonl", "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def run_source(name, fn):
    ts = utcnow()
    t0 = time.monotonic()
    ev = {"ts": ts.isoformat(), "source": name}
    try:
        ev.update(fn(ts))
        ev["ok"] = True
    except Exception as e:  # noqa: BLE001
        ev["ok"] = False
        ev["error"] = f"{type(e).__name__}: {e}"
        log.error("%s failed: %s", name, ev["error"])
    ev["elapsed_s"] = round(time.monotonic() - t0, 2)
    log_event(ev)
    if ev["ok"]:
        log.info("%s ok rows=%s bytes=%s %.1fs", name, ev.get("rows"), ev.get("bytes"), ev["elapsed_s"])
    return ev["ok"]


def collect_taipei(ts):
    body = fetch(TAIPEI_URL)
    save_raw("youbike_taipei", ts, body)
    return {"rows": count_rows(body), "bytes": len(body)}


def collect_newtaipei(ts):
    total_rows = total_bytes = pages = 0
    try:
        for page in range(NTPC_MAX_PAGES):
            body = fetch(NTPC_URL, {"page": page, "size": NTPC_PAGE_SIZE})
            n = count_rows(body)
            if n is None:
                raise ValueError(f"page {page}: cannot parse response")
            if n == 0:
                if page == 0:
                    raise ValueError("first page is empty")
                break
            save_raw("youbike_newtaipei", ts, body, suffix=f"_p{page}")
            pages += 1
            total_rows += n
            total_bytes += len(body)
            if n < NTPC_PAGE_SIZE:
                break
    except Exception as e:
        raise RuntimeError(f"after {pages} pages saved: {e}") from e
    return {"rows": total_rows, "bytes": total_bytes, "pages": pages}


def make_weather_collector(source, lat, lon):
    def _collect(ts):
        body = fetch(METEO_URL, {"latitude": lat, "longitude": lon, **WEATHER_PARAMS},
                     timeout=WEATHER_TIMEOUT, retries=WEATHER_RETRIES)
        save_raw(source, ts, body)
        return {"bytes": len(body)}
    return _collect


def weather_round():
    """Fetch all weather points once. Returns True if all succeeded."""
    return all([
        run_source(src, make_weather_collector(src, lat, lon))
        for src, (lat, lon) in WEATHER_POINTS.items()
    ])


def weather_loop():
    """Runs in its own thread so a slow/blocked weather API never delays YouBike."""
    while not stop.is_set():
        ok = weather_round()
        if ok:
            wait = WEATHER_INTERVAL - (time.time() % WEATHER_INTERVAL)
        else:
            wait = WEATHER_RETRY_AFTER
        stop.wait(wait)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    log.info("started: interval=%ss weather_interval=%ss data=%s", INTERVAL, WEATHER_INTERVAL, DATA_DIR)

    if not RUN_ONCE:
        threading.Thread(target=weather_loop, name="weather", daemon=True).start()

    while not stop.is_set():
        results = [
            run_source("youbike_taipei", collect_taipei),
            run_source("youbike_newtaipei", collect_newtaipei),
        ]
        if any(results):
            (DATA_DIR / "heartbeat").touch()  # used by Docker healthcheck

        if RUN_ONCE:
            weather_round()
            break
        # sleep until the next wall-clock boundary (:00, :05, :10 ...) to avoid drift
        stop.wait(INTERVAL - (time.time() % INTERVAL))
    log.info("stopped")


if __name__ == "__main__":
    main()