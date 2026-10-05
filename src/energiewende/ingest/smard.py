"""Electricity market data from SMARD.

Source: Bundesnetzagentur | SMARD.de, license CC BY 4.0.
SMARD stores each series in weekly files. An index file lists the start
time of every weekly file, in milliseconds since 1970.
"""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from energiewende.ingest.download import get_json

BASE_URL = "https://www.smard.de/app/chart_data"
SOURCE = "Bundesnetzagentur | SMARD.de (CC BY 4.0)"
BERLIN = ZoneInfo("Europe/Berlin")
DAY_MS = 24 * 3600 * 1000
MAX_DAYS = 31

# Our name -> (SMARD filter id, region, unit)
SERIES = {
    "price": (4169, "DE-LU", "EUR/MWh"),  # day-ahead wholesale price
    "load": (410, "DE", "MWh"),  # total electricity consumption
    "wind_onshore": (4067, "DE", "MWh"),
    "wind_offshore": (1225, "DE", "MWh"),
    "solar": (4068, "DE", "MWh"),
    "lignite": (1223, "DE", "MWh"),
    "hard_coal": (4069, "DE", "MWh"),
    "natural_gas": (4071, "DE", "MWh"),
}


def to_millis(day):
    """Midnight of the given day in Berlin, as milliseconds since 1970."""
    midnight = datetime(day.year, day.month, day.day, tzinfo=BERLIN)
    return int(midnight.timestamp() * 1000)


def get_series(name, start, end):
    """Hourly values of one series from start to end (both days included)."""
    if name not in SERIES:
        raise ValueError(f"Unknown series '{name}'. Choose from: {', '.join(SERIES)}")
    if end < start:
        raise ValueError("end must not be before start")
    if (end - start).days + 1 > MAX_DAYS:
        raise ValueError(f"At most {MAX_DAYS} days per request")

    filter_id, region, unit = SERIES[name]
    start_ms = to_millis(start)
    end_ms = to_millis(end + timedelta(days=1))  # midnight after the last day

    index = get_json(f"{BASE_URL}/{filter_id}/{region}/index_hour.json", max_age_hours=6)

    points = []
    for file_start in index["timestamps"]:
        # A weekly file covers about 7 days. We use 8 days to be safe
        # around daylight saving time changes.
        if file_start >= end_ms or file_start + 8 * DAY_MS <= start_ms:
            continue
        url = f"{BASE_URL}/{filter_id}/{region}/{filter_id}_{region}_hour_{file_start}.json"
        week = get_json(url, max_age_hours=6)
        for timestamp, value in week["series"]:
            if start_ms <= timestamp < end_ms and value is not None:
                time = datetime.fromtimestamp(timestamp / 1000, BERLIN)
                points.append({"time": time.isoformat(), "value": value})

    return {"series": name, "unit": unit, "source": SOURCE, "points": points}
