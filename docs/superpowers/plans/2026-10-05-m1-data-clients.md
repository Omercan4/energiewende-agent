# M1 Data Clients Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three small, tested Python clients that download electricity data (SMARD), weather data (Open-Meteo) and energy-related Bundestag papers (DIP), with a simple file cache.

**Architecture:** One helper, `get_json`, downloads JSON and caches it in files. Each data source has its own small module that calls `get_json`, checks its inputs, and returns plain Python dicts. Tests replace `get_json` with a fake, so no test touches the internet.

**Tech Stack:** Python 3.12, requests, python-dotenv, pytest.

**Spec:** `docs/superpowers/specs/2026-10-05-energiewende-agent-design.md` (sections 4, 8, 9, milestone M1)

This is plan 1 of 6. M2 (index), M3 (agent and API), M4 (evaluation), M5 (Docker and CI) and M6 (Azure and README) get their own plans after M1 has been reviewed.

## Global Constraints

- Code style: as simple and plain as possible. Plain functions, plain dicts, short comments in simple English. No classes unless a test needs a tiny fake.
- Python 3.12. Pinned versions: `requests==2.34.2`, `python-dotenv==1.2.4`, `pytest==9.1.1`.
- No API key is ever written into a committed file. Keys live only in `.env` (git-ignored).
- No test talks to a real API.
- Attribution strings, used exactly:
  - SMARD: `Bundesnetzagentur | SMARD.de (CC BY 4.0)`
  - Open-Meteo: `Open-Meteo.com, data from DWD (CC BY 4.0)`
  - DIP: `Deutscher Bundestag, DIP`
- Every commit message ends with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Verified facts these tasks rely on (checked 2026-10-05)

- SMARD index: `https://www.smard.de/app/chart_data/{filter}/{region}/index_hour.json` returns `{"timestamps": [ms, ...]}`. Each timestamp is the start of a weekly file.
- SMARD weekly file: `.../{filter}/{region}/{filter}_{region}_hour_{timestamp}.json` returns `{"meta_data": {...}, "series": [[ms, value_or_null], ...]}`.
- SMARD filter ids: 4169 price (region `DE-LU`), 410 grid load, 4067 wind onshore, 1225 wind offshore, 4068 solar, 1223 lignite, 4069 hard coal, 4071 natural gas (region `DE`).
- Open-Meteo returns `{"hourly": {"time": [...], "<variable>": [...]}, "hourly_units": {...}}`. The archive API has complete data up to about 6 days ago; the forecast API accepts `start_date` in the recent past.
- DIP `drucksache-text` returns `{"numFound", "cursor", "documents"}` with 10 documents per page. When there are no more results it returns zero documents and the same cursor. Each document has `id`, `dokumentnummer`, `datum`, `drucksachetyp`, `titel`, `urheber` (list of dicts with `titel`), `fundstelle` (dict with `pdf_url`), `text`.
- DIP auth: header `Authorization: ApiKey <key>`.

## File Structure

```
pyproject.toml                        project name + pytest settings
requirements.txt                      pinned packages
.env.example                          which settings exist (no real values)
src/energiewende/__init__.py          empty, marks the package
src/energiewende/config.py            reads settings from the environment / .env
src/energiewende/ingest/__init__.py   empty
src/energiewende/ingest/download.py   get_json: download + file cache
src/energiewende/ingest/smard.py      get_series: electricity data
src/energiewende/ingest/openmeteo.py  get_weather: weather data
src/energiewende/ingest/dip.py        search_papers, get_energy_papers: Bundestag papers
scripts/check_sources.py              calls each real API once, prints a summary
tests/test_download.py
tests/test_smard.py
tests/test_openmeteo.py
tests/test_dip.py
```

---

### Task 1: Project setup and the cached download helper

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `.env.example`, `src/energiewende/__init__.py`, `src/energiewende/config.py`, `src/energiewende/ingest/__init__.py`, `src/energiewende/ingest/download.py`
- Test: `tests/test_download.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `energiewende.config.CACHE_DIR: str` (default `"data/cache"`), `energiewende.config.DIP_API_KEY: str` (default `""`)
  - `energiewende.ingest.download.get_json(url: str, params: dict | None = None, headers: dict | None = None, max_age_hours: float = 24) -> dict | list`

- [ ] **Step 1: Create the project files**

`pyproject.toml`:

```toml
[project]
name = "energiewende-agent"
version = "0.1.0"
requires-python = ">=3.12"

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

`requirements.txt`:

```
requests==2.34.2
python-dotenv==1.2.4
pytest==9.1.1
```

`.env.example`:

```
# Copy this file to .env and fill in the values. Never commit .env.

# Bundestag DIP API key. A public key is listed at
# https://dip.bundestag.de/über-dip/hilfe/api
DIP_API_KEY=

# Folder for cached API responses
CACHE_DIR=data/cache
```

`src/energiewende/__init__.py` and `src/energiewende/ingest/__init__.py`: empty files.

`src/energiewende/config.py`:

```python
"""All settings in one place. Values come from environment variables or the .env file."""

import os

from dotenv import load_dotenv

load_dotenv()  # reads .env in the current folder, if it exists

# Folder where downloaded API responses are cached.
CACHE_DIR = os.environ.get("CACHE_DIR", "data/cache")

# Key for the Bundestag DIP API.
DIP_API_KEY = os.environ.get("DIP_API_KEY", "")
```

- [ ] **Step 2: Create the virtual environment and install packages**

Run:

```bash
cd ~/Desktop/energiewende-agent
python3.12 -m venv .venv || python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python --version
```

Expected: the last line prints `Python 3.12.x`.

- [ ] **Step 3: Write the failing tests**

`tests/test_download.py`:

```python
from energiewende import config
from energiewende.ingest import download


class FakeResponse:
    """Looks like a requests response, but holds fixed data."""

    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def use_fake_internet(monkeypatch, tmp_path):
    """Send all downloads to a fake, and the cache to a temporary folder.
    Returns the list of downloaded URLs."""
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path))
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        return FakeResponse({"call": len(calls)})

    monkeypatch.setattr(download.requests, "get", fake_get)
    return calls


def test_second_call_comes_from_cache(monkeypatch, tmp_path):
    calls = use_fake_internet(monkeypatch, tmp_path)

    first = download.get_json("https://example.com/data")
    second = download.get_json("https://example.com/data")

    assert first == {"call": 1}
    assert second == {"call": 1}  # same data, no new download
    assert len(calls) == 1


def test_old_cache_is_downloaded_again(monkeypatch, tmp_path):
    calls = use_fake_internet(monkeypatch, tmp_path)

    download.get_json("https://example.com/data")
    result = download.get_json("https://example.com/data", max_age_hours=0)

    assert result == {"call": 2}
    assert len(calls) == 2


def test_different_params_are_cached_separately(monkeypatch, tmp_path):
    calls = use_fake_internet(monkeypatch, tmp_path)

    download.get_json("https://example.com/data", params={"day": 1})
    download.get_json("https://example.com/data", params={"day": 2})

    assert len(calls) == 2
    assert len(list(tmp_path.iterdir())) == 2
```

- [ ] **Step 4: Run the tests to see them fail**

Run: `.venv/bin/pytest tests/test_download.py -v`
Expected: FAIL with `ImportError: cannot import name 'download'`.

- [ ] **Step 5: Write the download helper**

`src/energiewende/ingest/download.py`:

```python
"""Download JSON from a URL and keep a copy in a cache folder."""

import hashlib
import json
import os
import time

import requests

from energiewende import config


def get_json(url, params=None, headers=None, max_age_hours=24):
    """Return the JSON at url. Use the cached copy if it is younger than max_age_hours."""
    # The file name is a hash of the URL and the parameters.
    # Headers are not part of it, so API keys never end up in file names.
    key = url + json.dumps(params or {}, sort_keys=True)
    file_name = hashlib.sha256(key.encode()).hexdigest() + ".json"
    path = os.path.join(config.CACHE_DIR, file_name)

    if os.path.exists(path):
        age_hours = (time.time() - os.path.getmtime(path)) / 3600
        if age_hours < max_age_hours:
            with open(path, encoding="utf-8") as f:
                return json.load(f)

    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()  # stop with an error on 4xx / 5xx
    data = response.json()

    os.makedirs(config.CACHE_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data
```

- [ ] **Step 6: Run the tests to see them pass**

Run: `.venv/bin/pytest tests/test_download.py -v`
Expected: 3 passed.

- [ ] **Step 7: Create your local .env (not committed)**

Run: `cp .env.example .env`, then put the public key from https://dip.bundestag.de/über-dip/hilfe/api after `DIP_API_KEY=`.
Check: `git status --short` must NOT list `.env`.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml requirements.txt .env.example src tests/test_download.py
git commit -m "feat: project setup and cached download helper" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: SMARD electricity data client

**Files:**
- Create: `src/energiewende/ingest/smard.py`
- Test: `tests/test_smard.py`

**Interfaces:**
- Consumes: `get_json(url, params=None, headers=None, max_age_hours=24)` from Task 1.
- Produces:
  - `smard.SERIES: dict[str, tuple[int, str, str]]` (name -> filter id, region, unit). Names: `price, load, wind_onshore, wind_offshore, solar, lignite, hard_coal, natural_gas`.
  - `smard.to_millis(day: date) -> int`
  - `smard.get_series(name: str, start: date, end: date) -> dict` returning `{"series": str, "unit": str, "source": str, "points": [{"time": str (ISO, Berlin time), "value": float}]}`. Raises `ValueError` for an unknown name, `end < start`, or more than 31 days.

- [ ] **Step 1: Write the failing tests**

`tests/test_smard.py`:

```python
from datetime import date

import pytest

from energiewende.ingest import smard

HOUR = 3600 * 1000
WEEK1 = smard.to_millis(date(2026, 9, 21))  # start of the first weekly file
WEEK2 = smard.to_millis(date(2026, 9, 28))  # start of the second weekly file


def use_fake_smard(monkeypatch):
    """Replace downloads with two small weekly files shaped like the real ones.
    Returns the list of requested URLs."""
    urls = []

    def fake_get_json(url, params=None, headers=None, max_age_hours=24):
        urls.append(url)
        if url.endswith("index_hour.json"):
            return {"timestamps": [WEEK1, WEEK2]}
        if url.endswith(f"_{WEEK1}.json"):
            return {"meta_data": {}, "series": [[WEEK1, 50.0], [WEEK1 + HOUR, 60.0]]}
        if url.endswith(f"_{WEEK2}.json"):
            return {"meta_data": {}, "series": [[WEEK2, 70.0], [WEEK2 + HOUR, None]]}
        raise AssertionError("unexpected url " + url)

    monkeypatch.setattr(smard, "get_json", fake_get_json)
    return urls


def test_returns_only_points_inside_the_range(monkeypatch):
    use_fake_smard(monkeypatch)

    result = smard.get_series("price", date(2026, 9, 28), date(2026, 9, 28))

    assert result["series"] == "price"
    assert result["unit"] == "EUR/MWh"
    assert result["source"] == "Bundesnetzagentur | SMARD.de (CC BY 4.0)"
    # Points from the first week are outside the range, and None is skipped.
    assert result["points"] == [{"time": "2026-09-28T00:00:00+02:00", "value": 70.0}]


def test_uses_the_real_smard_urls(monkeypatch):
    urls = use_fake_smard(monkeypatch)

    smard.get_series("price", date(2026, 9, 28), date(2026, 9, 28))

    assert urls[0] == "https://www.smard.de/app/chart_data/4169/DE-LU/index_hour.json"
    assert f"https://www.smard.de/app/chart_data/4169/DE-LU/4169_DE-LU_hour_{WEEK2}.json" in urls


def test_unknown_series_is_rejected():
    with pytest.raises(ValueError, match="Unknown series"):
        smard.get_series("coffee", date(2026, 9, 1), date(2026, 9, 2))


def test_end_before_start_is_rejected():
    with pytest.raises(ValueError, match="before start"):
        smard.get_series("price", date(2026, 9, 2), date(2026, 9, 1))


def test_too_long_range_is_rejected():
    with pytest.raises(ValueError, match="At most 31 days"):
        smard.get_series("price", date(2026, 1, 1), date(2026, 3, 1))
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/pytest tests/test_smard.py -v`
Expected: FAIL with `ImportError: cannot import name 'smard'`.

- [ ] **Step 3: Write the SMARD client**

`src/energiewende/ingest/smard.py`:

```python
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
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/pytest tests/test_smard.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/energiewende/ingest/smard.py tests/test_smard.py
git commit -m "feat: SMARD electricity data client" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Open-Meteo weather client

**Files:**
- Create: `src/energiewende/ingest/openmeteo.py`
- Test: `tests/test_openmeteo.py`

**Interfaces:**
- Consumes: `get_json(url, params=None, headers=None, max_age_hours=24)` from Task 1.
- Produces:
  - `openmeteo.REGIONS: dict[str, tuple[float, float]]` with names `north_sea, berlin, cologne, munich`.
  - `openmeteo.VARIABLES: list[str]` = `["wind_speed_100m", "shortwave_radiation", "temperature_2m"]`.
  - `openmeteo.get_weather(region: str, start: date, end: date, today: date | None = None) -> dict` returning `{"region": str, "units": dict[str, str], "source": str, "points": [{"time": str, "wind_speed_100m": float, "shortwave_radiation": float, "temperature_2m": float}]}`. Raises `ValueError` for an unknown region, `end < start`, or more than 31 days. `today` exists only so tests can fix the date.

- [ ] **Step 1: Write the failing tests**

`tests/test_openmeteo.py`:

```python
from datetime import date

import pytest

from energiewende.ingest import openmeteo

TODAY = date(2026, 10, 5)

FAKE_RESPONSE = {
    "hourly_units": {
        "time": "iso8601",
        "wind_speed_100m": "km/h",
        "shortwave_radiation": "W/m²",
        "temperature_2m": "°C",
    },
    "hourly": {
        "time": ["2026-09-01T00:00", "2026-09-01T01:00"],
        "wind_speed_100m": [33.3, 30.1],
        "shortwave_radiation": [0.0, 0.0],
        "temperature_2m": [14.2, 13.9],
    },
}


def use_fake_openmeteo(monkeypatch):
    """Replace downloads with a fixed response. Returns a list of (url, params)."""
    calls = []

    def fake_get_json(url, params=None, headers=None, max_age_hours=24):
        calls.append((url, params))
        return FAKE_RESPONSE

    monkeypatch.setattr(openmeteo, "get_json", fake_get_json)
    return calls


def test_turns_columns_into_points(monkeypatch):
    use_fake_openmeteo(monkeypatch)

    result = openmeteo.get_weather("north_sea", date(2026, 9, 1), date(2026, 9, 1), today=TODAY)

    assert result["region"] == "north_sea"
    assert result["source"] == "Open-Meteo.com, data from DWD (CC BY 4.0)"
    assert result["units"]["wind_speed_100m"] == "km/h"
    assert result["points"][0] == {
        "time": "2026-09-01T00:00",
        "wind_speed_100m": 33.3,
        "shortwave_radiation": 0.0,
        "temperature_2m": 14.2,
    }
    assert len(result["points"]) == 2


def test_old_days_use_the_archive(monkeypatch):
    calls = use_fake_openmeteo(monkeypatch)

    openmeteo.get_weather("berlin", date(2026, 9, 1), date(2026, 9, 2), today=TODAY)

    url, params = calls[0]
    assert url == "https://archive-api.open-meteo.com/v1/archive"
    assert params["latitude"] == 52.52
    assert params["start_date"] == "2026-09-01"
    assert params["end_date"] == "2026-09-02"
    assert params["timezone"] == "Europe/Berlin"


def test_recent_days_use_the_forecast_api(monkeypatch):
    calls = use_fake_openmeteo(monkeypatch)

    openmeteo.get_weather("berlin", date(2026, 10, 3), date(2026, 10, 6), today=TODAY)

    url, params = calls[0]
    assert url == "https://api.open-meteo.com/v1/forecast"


def test_unknown_region_is_rejected():
    with pytest.raises(ValueError, match="Unknown region"):
        openmeteo.get_weather("atlantis", date(2026, 9, 1), date(2026, 9, 1), today=TODAY)


def test_too_long_range_is_rejected():
    with pytest.raises(ValueError, match="At most 31 days"):
        openmeteo.get_weather("berlin", date(2026, 1, 1), date(2026, 3, 1), today=TODAY)
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/pytest tests/test_openmeteo.py -v`
Expected: FAIL with `ImportError: cannot import name 'openmeteo'`.

- [ ] **Step 3: Write the Open-Meteo client**

`src/energiewende/ingest/openmeteo.py`:

```python
"""Weather data from Open-Meteo.

Source: Open-Meteo.com, using data from Deutscher Wetterdienst (DWD).
License CC BY 4.0. The free API is for non-commercial use.
"""

from datetime import date, timedelta

from energiewende.ingest.download import get_json

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
SOURCE = "Open-Meteo.com, data from DWD (CC BY 4.0)"
MAX_DAYS = 31
VARIABLES = ["wind_speed_100m", "shortwave_radiation", "temperature_2m"]

# A few fixed places, so nobody can ask for random coordinates.
REGIONS = {
    "north_sea": (53.55, 8.58),  # Bremerhaven, near offshore wind farms
    "berlin": (52.52, 13.41),
    "cologne": (50.94, 6.96),
    "munich": (48.14, 11.58),
}


def get_weather(region, start, end, today=None):
    """Hourly weather for one region from start to end (both days included)."""
    if region not in REGIONS:
        raise ValueError(f"Unknown region '{region}'. Choose from: {', '.join(REGIONS)}")
    if end < start:
        raise ValueError("end must not be before start")
    if (end - start).days + 1 > MAX_DAYS:
        raise ValueError(f"At most {MAX_DAYS} days per request")

    # The archive is complete only up to about 6 days ago. Newer days come
    # from the forecast API, which also accepts dates in the recent past.
    today = today or date.today()
    url = ARCHIVE_URL if end <= today - timedelta(days=6) else FORECAST_URL

    latitude, longitude = REGIONS[region]
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(VARIABLES),
        "timezone": "Europe/Berlin",
    }
    data = get_json(url, params=params, max_age_hours=6)

    # The API returns one list per variable. We turn that into one dict per hour.
    hourly = data["hourly"]
    points = []
    for i, time in enumerate(hourly["time"]):
        point = {"time": time}
        for name in VARIABLES:
            point[name] = hourly[name][i]
        points.append(point)

    units = {name: data["hourly_units"][name] for name in VARIABLES}
    return {"region": region, "units": units, "source": SOURCE, "points": points}
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/pytest tests/test_openmeteo.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/energiewende/ingest/openmeteo.py tests/test_openmeteo.py
git commit -m "feat: Open-Meteo weather client" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Bundestag DIP papers client

**Files:**
- Create: `src/energiewende/ingest/dip.py`
- Test: `tests/test_dip.py`

**Interfaces:**
- Consumes: `get_json(...)` from Task 1, `config.DIP_API_KEY` from Task 1.
- Produces:
  - `dip.ENERGY_KEYWORDS: list[str]` = `["Energie", "Strom", "Erneuerbare", "Wasserstoff", "Wärme"]`
  - `dip.to_paper(doc: dict) -> dict` returning `{"id": str, "number": str, "date": str, "type": str, "title": str, "authors": list[str], "url": str, "text": str}`. M2 builds the index from these dicts.
  - `dip.search_papers(keyword: str, wahlperiode: int = 21, max_docs: int = 50) -> list[dict]` (paper dicts as above). Raises `RuntimeError` if `DIP_API_KEY` is empty.
  - `dip.get_energy_papers(wahlperiode: int = 21, max_docs: int = 200) -> list[dict]` (all keywords, no duplicate ids).

- [ ] **Step 1: Write the failing tests**

`tests/test_dip.py`:

```python
import pytest

from energiewende import config
from energiewende.ingest import dip


def make_doc(n):
    """One document, shaped like the real DIP response."""
    return {
        "id": str(n),
        "dokumentnummer": f"21/{n}",
        "datum": "2026-09-30",
        "drucksachetyp": "Antrag",
        "titel": f"Paper {n}",
        "urheber": [{"titel": "Bundesregierung"}],
        "fundstelle": {"pdf_url": f"https://example.com/{n}.pdf"},
        "text": f"Text of paper {n}",
    }


# Three pages, found by cursor. The last page is empty and repeats its cursor.
PAGES = {
    None: {"numFound": 3, "cursor": "c1", "documents": [make_doc(1), make_doc(2)]},
    "c1": {"numFound": 3, "cursor": "c2", "documents": [make_doc(3)]},
    "c2": {"numFound": 3, "cursor": "c2", "documents": []},
}


def use_fake_dip(monkeypatch):
    """Replace downloads with the pages above. Returns the list of params sent."""
    monkeypatch.setattr(config, "DIP_API_KEY", "test-key")
    sent = []

    def fake_get_json(url, params=None, headers=None, max_age_hours=24):
        assert url == "https://search.dip.bundestag.de/api/v1/drucksache-text"
        assert headers == {"Authorization": "ApiKey test-key"}
        sent.append(params)
        return PAGES[params.get("cursor")]

    monkeypatch.setattr(dip, "get_json", fake_get_json)
    return sent


def test_follows_the_cursor_until_the_end(monkeypatch):
    sent = use_fake_dip(monkeypatch)

    papers = dip.search_papers("Strom")

    assert [p["number"] for p in papers] == ["21/1", "21/2", "21/3"]
    assert sent[0] == {"f.wahlperiode": 21, "f.titel": "Strom"}
    assert sent[1]["cursor"] == "c1"


def test_keeps_only_the_fields_we_need(monkeypatch):
    use_fake_dip(monkeypatch)

    paper = dip.search_papers("Strom")[0]

    assert paper == {
        "id": "1",
        "number": "21/1",
        "date": "2026-09-30",
        "type": "Antrag",
        "title": "Paper 1",
        "authors": ["Bundesregierung"],
        "url": "https://example.com/1.pdf",
        "text": "Text of paper 1",
    }


def test_max_docs_limits_the_result(monkeypatch):
    use_fake_dip(monkeypatch)

    papers = dip.search_papers("Strom", max_docs=2)

    assert len(papers) == 2


def test_missing_key_gives_a_clear_error(monkeypatch):
    monkeypatch.setattr(config, "DIP_API_KEY", "")

    with pytest.raises(RuntimeError, match="DIP_API_KEY"):
        dip.search_papers("Strom")


def test_energy_papers_have_no_duplicates(monkeypatch):
    use_fake_dip(monkeypatch)

    # The fake returns the same three papers for every keyword.
    papers = dip.get_energy_papers()

    assert sorted(p["id"] for p in papers) == ["1", "2", "3"]
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/pytest tests/test_dip.py -v`
Expected: FAIL with `ImportError: cannot import name 'dip'`.

- [ ] **Step 3: Write the DIP client**

`src/energiewende/ingest/dip.py`:

```python
"""Printed papers (Drucksachen) from the Bundestag DIP API.

The API needs a key, sent in the Authorization header. We only download
papers whose title contains an energy keyword, because full texts are long
and we want a small corpus.
"""

from energiewende import config
from energiewende.ingest.download import get_json

URL = "https://search.dip.bundestag.de/api/v1/drucksache-text"
SOURCE = "Deutscher Bundestag, DIP"
ENERGY_KEYWORDS = ["Energie", "Strom", "Erneuerbare", "Wasserstoff", "Wärme"]


def to_paper(doc):
    """Keep only the fields we need from one DIP document."""
    return {
        "id": doc["id"],
        "number": doc["dokumentnummer"],
        "date": doc["datum"],
        "type": doc.get("drucksachetyp", ""),
        "title": doc["titel"],
        "authors": [u["titel"] for u in doc.get("urheber", [])],
        "url": doc.get("fundstelle", {}).get("pdf_url", ""),
        "text": doc.get("text") or "",
    }


def search_papers(keyword, wahlperiode=21, max_docs=50):
    """Papers of one electoral period whose title contains the keyword."""
    if not config.DIP_API_KEY:
        raise RuntimeError("DIP_API_KEY is not set. See .env.example.")

    headers = {"Authorization": "ApiKey " + config.DIP_API_KEY}
    params = {"f.wahlperiode": wahlperiode, "f.titel": keyword}
    papers = []

    while len(papers) < max_docs:
        page = get_json(URL, params=params, headers=headers)
        if not page["documents"]:
            break  # no more results
        papers += [to_paper(doc) for doc in page["documents"]]
        # Each page gives the cursor for the next page. When there are
        # no more pages, the API returns the same cursor again.
        if page["cursor"] == params.get("cursor"):
            break
        params = {**params, "cursor": page["cursor"]}

    return papers[:max_docs]


def get_energy_papers(wahlperiode=21, max_docs=200):
    """Papers for all energy keywords, without duplicates."""
    papers = {}  # id -> paper, so the same paper is stored only once
    for keyword in ENERGY_KEYWORDS:
        for paper in search_papers(keyword, wahlperiode, max_docs):
            papers[paper["id"]] = paper
    return list(papers.values())[:max_docs]
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/pytest tests/test_dip.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/energiewende/ingest/dip.py tests/test_dip.py
git commit -m "feat: Bundestag DIP papers client" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Live check against the real APIs

This is the moment to see real data. It is a script, not a test, so CI never calls the real APIs.

**Files:**
- Create: `scripts/check_sources.py`

**Interfaces:**
- Consumes: `smard.get_series`, `openmeteo.get_weather`, `dip.search_papers` from Tasks 2-4.
- Produces: nothing other tasks depend on.

- [ ] **Step 1: Write the script**

`scripts/check_sources.py`:

```python
"""Call each real API once and print a short summary. Not part of the tests.

Run from the project folder:  PYTHONPATH=src .venv/bin/python scripts/check_sources.py
"""

from datetime import date, timedelta

from energiewende.ingest import dip, openmeteo, smard

day = date.today() - timedelta(days=2)

price = smard.get_series("price", day, day)
print(f"SMARD price on {day}: {len(price['points'])} hours, first: {price['points'][:1]}")

wind = smard.get_series("wind_onshore", day, day)
print(f"SMARD wind onshore on {day}: {len(wind['points'])} hours, first: {wind['points'][:1]}")

weather = openmeteo.get_weather("north_sea", day, day)
print(f"Open-Meteo north_sea on {day}: {len(weather['points'])} hours, first: {weather['points'][:1]}")

papers = dip.search_papers("Strom", max_docs=3)
for paper in papers:
    print(f"DIP: {paper['number']} {paper['date']} {paper['title'][:60]} ({len(paper['text'])} characters)")
```

- [ ] **Step 2: Run it**

Run: `PYTHONPATH=src .venv/bin/python scripts/check_sources.py`
Expected: four kinds of lines. SMARD shows 23-25 hours with a value, Open-Meteo shows 24 hours, DIP shows three papers with thousands of characters each. If DIP fails with 401, the key in `.env` is wrong or expired.

- [ ] **Step 3: Run the whole test suite**

Run: `.venv/bin/pytest -v`
Expected: 18 passed.

- [ ] **Step 4: Commit**

```bash
git add scripts/check_sources.py
git commit -m "feat: live check script for the three data sources" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
