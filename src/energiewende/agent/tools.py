"""The tools the agent can use.

Each tool returns a small dict. The data tools compute mean, min, max and total
in Python, because LLMs make mistakes when they add up many numbers.
Every result has a "sources" list, so the answer can name where the data is from.
"""

import functools
from datetime import date

from langchain_core.tools import tool

from energiewende.index import search, store
from energiewende.ingest import openmeteo, smard

SEARCH_K = 4  # how many text passages bundestag_search returns
CHUNK_SIZE = 400  # which search index bundestag_search uses (there is one per chunk size)


def summarize(points, key="value"):
    """Mean, min, max and total of all points, and the same per day."""
    points = [p for p in points if p[key] is not None]
    if not points:
        raise ValueError("No data for this time range. It may be in the future or not published yet.")
    values = [p[key] for p in points]

    days = {}  # "2026-10-04" -> list of values
    for p in points:
        days.setdefault(p["time"][:10], []).append(p[key])
    daily = [
        {"date": day, "mean": round(sum(v) / len(v), 2), "min": min(v), "max": max(v), "total": round(sum(v), 2)}
        for day, v in days.items()
    ]

    lowest = min(points, key=lambda p: p[key])
    highest = max(points, key=lambda p: p[key])
    return {
        "mean": round(sum(values) / len(values), 2),
        "min": {"time": lowest["time"], "value": lowest[key]},
        "max": {"time": highest["time"], "value": highest[key]},
        "total": round(sum(values), 2),
        "daily": daily,
    }


def data_result(raw, start, end, name):
    """The common shape of the price and generation results."""
    result = {
        "series": raw["series"],
        "unit": raw["unit"],
        "start": start,
        "end": end,
        "summary": summarize(raw["points"]),
        "sources": [f"{raw['source']}, {name}, {start} to {end}"],
    }
    if (date.fromisoformat(end) - date.fromisoformat(start)).days < 2:
        result["hourly"] = raw["points"]  # short range: hourly values are small enough to send
    return result


@tool
def price_series(start: str, end: str) -> dict:
    """Day-ahead wholesale electricity price in Germany/Luxembourg in EUR/MWh.
    Dates as YYYY-MM-DD, both included, at most 31 days.
    Returns mean, min, max per day and overall (hourly values for up to 2 days)."""
    raw = smard.get_series("price", date.fromisoformat(start), date.fromisoformat(end))
    return data_result(raw, start, end, "price")


@tool
def generation_load(series: str, start: str, end: str) -> dict:
    """Electricity generation by source or total consumption in Germany, in MWh per hour.
    series is one of: load, wind_onshore, wind_offshore, solar, lignite, hard_coal, natural_gas.
    Dates as YYYY-MM-DD, both included, at most 31 days.
    Returns mean, min, max and total per day and overall (hourly values for up to 2 days)."""
    if series == "price":
        raise ValueError("For prices use the price_series tool.")
    raw = smard.get_series(series, date.fromisoformat(start), date.fromisoformat(end))
    return data_result(raw, start, end, series)


@tool
def weather(region: str, start: str, end: str) -> dict:
    """Hourly weather: wind speed at 100 m, solar radiation, temperature.
    region is one of: north_sea, berlin, cologne, munich.
    Dates as YYYY-MM-DD, both included, at most 31 days.
    Returns mean, min, max per day and overall for each variable."""
    raw = openmeteo.get_weather(region, date.fromisoformat(start), date.fromisoformat(end))
    return {
        "region": region,
        "units": raw["units"],
        "start": start,
        "end": end,
        "summary_by_variable": {name: summarize(raw["points"], key=name) for name in openmeteo.VARIABLES},
        "sources": [f"{raw['source']}, {region}, {start} to {end}"],
    }


@functools.cache
def get_index(chunk_size):
    """Load the search index for this chunk size once, on first use."""
    return store.load_index(chunk_size)


@tool
def bundestag_search(question: str) -> dict:
    """Search Bundestag printed papers (Drucksachen) on energy topics: laws, motions,
    committee reports. Use it for what parties, the government or the Bundestag
    want, plan, criticize or decided. Returns the best matching text passages."""
    hits = search.search(get_index(CHUNK_SIZE), question, k=SEARCH_K)
    sources = [f"Bundestag Drucksache {h['number']} ({h['date']}): {h['url']}" for h in hits]
    return {"hits": hits, "sources": sorted(set(sources))}


DATA_TOOLS = [price_series, generation_load, weather]
TOOLS = DATA_TOOLS + [bundestag_search]
