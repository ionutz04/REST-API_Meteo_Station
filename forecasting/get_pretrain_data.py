#!/usr/bin/env python3
import argparse
import csv
import logging
import sys
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import requests

PWS_HISTORY_HOST = "https://api.weather.com/v2/pws/history/all"
API_KEY = "e1f10a1e78da46f5b10a1e78da96f525"
STATION_ID = "IOTOPE31"

CSV_COLUMNS = [
    "station_id",
    "obs_time_utc",
    "obs_time_local",
    "temperature_c",
    "dewpoint_c",
    "humidity_pct",
    "pressure_hpa",
    "wind_speed_ms",
    "wind_speed_kmh",
    "wind_gust_ms",
    "wind_gust_kmh",
    "wind_dir_deg",
    "precip_rate_mm_h",
    "precip_total_mm",
    "solar_radiation_w_m2",
    "uv_index",
    "elevation_m",
    "lat",
    "lon",
    "qc_status",
    "source_url",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.wunderground.com/",
    "Origin": "https://www.wunderground.com",
}


def setup_logger(verbose: bool, log_file: str) -> logging.Logger:
    logger = logging.getLogger("wu-pws-history")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def f_to_c(f: Optional[float]) -> Optional[float]:
    if f is None:
        return None
    return round((f - 32.0) * 5.0 / 9.0, 2)


def inhg_to_hpa(inhg: Optional[float]) -> Optional[float]:
    if inhg is None:
        return None
    return round(inhg * 33.8638866667, 2)


def mph_to_ms(mph: Optional[float]) -> Optional[float]:
    if mph is None:
        return None
    return round(mph * 0.44704, 3)


def mph_to_kmh(mph: Optional[float]) -> Optional[float]:
    if mph is None:
        return None
    return round(mph * 1.609344, 3)


def inch_to_mm(inches: Optional[float]) -> Optional[float]:
    if inches is None:
        return None
    return round(inches * 25.4, 3)


def build_url(station_id: str, day: date) -> str:
    date_str = day.strftime("%Y%m%d")
    return (
        f"{PWS_HISTORY_HOST}"
        f"?stationId={station_id}"
        f"&format=json"
        f"&units=e"
        f"&numericPrecision=decimal"
        f"&date={date_str}"
        f"&apiKey={API_KEY}"
    )


def safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def map_observation(obs: Dict, source_url: str, station_id: str) -> Dict:
    imperial = obs.get("imperial", {}) if isinstance(obs.get("imperial"), dict) else {}

    temp_f = imperial.get("temp")
    dew_f = imperial.get("dewpt")
    pressure_in = imperial.get("pressure")
    wind_mph = imperial.get("windSpeed")
    gust_mph = imperial.get("windGust")
    precip_rate_in = imperial.get("precipRate")
    precip_total_in = imperial.get("precipTotal")

    return {
        "station_id": obs.get("stationID", station_id),
        "obs_time_utc": obs.get("obsTimeUtc"),
        "obs_time_local": obs.get("obsTimeLocal"),
        "temperature_c": f_to_c(temp_f),
        "dewpoint_c": f_to_c(dew_f),
        "humidity_pct": obs.get("humidity"),
        "pressure_hpa": inhg_to_hpa(pressure_in),
        "wind_speed_ms": mph_to_ms(wind_mph),
        "wind_speed_kmh": mph_to_kmh(wind_mph),
        "wind_gust_ms": mph_to_ms(gust_mph),
        "wind_gust_kmh": mph_to_kmh(gust_mph),
        "wind_dir_deg": obs.get("winddir"),
        "precip_rate_mm_h": inch_to_mm(precip_rate_in),
        "precip_total_mm": inch_to_mm(precip_total_in),
        "solar_radiation_w_m2": obs.get("solarRadiation"),
        "uv_index": obs.get("uv"),
        "elevation_m": obs.get("metric", {}).get("elev") if isinstance(obs.get("metric"), dict) else obs.get("elev"),
        "lat": obs.get("lat"),
        "lon": obs.get("lon"),
        "qc_status": obs.get("qcStatus"),
        "source_url": source_url,
    }


def fetch_day(session: requests.Session, station_id: str, day: date, logger: logging.Logger, retries: int = 3) -> List[Dict]:
    url = build_url(station_id, day)

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"[{day}] GET {url} (attempt {attempt}/{retries})")
            r = session.get(url, headers=HEADERS, timeout=40)
            logger.info(f"[{day}] HTTP {r.status_code}, bytes={len(r.text)}")

            if r.status_code != 200:
                logger.warning(f"[{day}] non-200 response")
                time.sleep(attempt)
                continue

            data = r.json()
            observations = data.get("observations", [])

            logger.info(f"[{day}] observations found: {len(observations)}")

            rows = [map_observation(obs, url, station_id) for obs in observations]

            if rows:
                for i, row in enumerate(rows[:3], 1):
                    logger.debug(f"[{day}] sample row {i}: {row}")
                return rows

            logger.warning(f"[{day}] zero observations in payload")
            logger.debug(f"[{day}] response keys: {list(data.keys())}")
            time.sleep(attempt)

        except Exception as e:
            logger.exception(f"[{day}] request/parse failed: {e}")
            time.sleep(attempt)

    logger.error(f"[{day}] FAILED after {retries} attempts")
    return []


def write_csv(path: str, rows: List[Dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station-id", default=STATION_ID)
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--output", default="otopeni_pws_2025_metric.csv")
    ap.add_argument("--log-file", default="otopeni_pws_2025_metric.log")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--limit-days", type=int, default=0)
    args = ap.parse_args()

    logger = setup_logger(args.verbose, args.log_file)
    logger.info(f"Station ID   : {args.station_id}")
    logger.info(f"Date range   : {args.start} -> {args.end}")
    logger.info("Target units : °C, hPa, m/s, km/h, mm")

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    session = requests.Session()

    all_rows = []
    ok_days = 0
    fail_days = 0
    n_days = 0

    for d in daterange(start, end):
        n_days += 1
        rows = fetch_day(session, args.station_id, d, logger)

        if rows:
            ok_days += 1
            all_rows.extend(rows)
            logger.info(f"[{d}] SUCCESS rows={len(rows)} total={len(all_rows)}")
        else:
            fail_days += 1
            logger.warning(f"[{d}] NO DATA")

        if args.limit_days and n_days >= args.limit_days:
            logger.warning(f"Stopping early because --limit-days={args.limit_days}")
            break

        time.sleep(args.sleep)

    all_rows.sort(key=lambda r: (r["obs_time_local"] or "", r["obs_time_utc"] or ""))
    write_csv(args.output, all_rows)

    logger.info("=== FINAL ===")
    logger.info(f"days processed = {n_days}")
    logger.info(f"ok days        = {ok_days}")
    logger.info(f"failed days    = {fail_days}")
    logger.info(f"rows total     = {len(all_rows)}")
    logger.info(f"csv            = {args.output}")


if __name__ == "__main__":
    main()