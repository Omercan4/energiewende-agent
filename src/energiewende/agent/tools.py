"""The tools the agent can use.

Each tool returns a small dict. The data tools compute mean, min, max and total
in Python (with explicit names: per hour, per complete day), because LLMs make mistakes when they add up many numbers.
Every result has a "sources" list, so the answer can name where the data is from.
"""

import functools
from datetime import date
from typing import Optional

from langchain_core.tools import tool

from energiewende import config
from energiewende.index import search, store
from energiewende.ingest import openmeteo, smard

SEARCH_K = 4  # how many text passages bundestag_search returns
CHUNK_SIZE = 400  # which search index bundestag_search uses (there is one per chunk size)
USE_FILTERS = True  # False: ignore the party / paper_type arguments (for the evaluation)
USE_RERANKER = bool(config.RERANKER_MODEL)  # False: no second sorting step (for the evaluation)


COMPLETE_DAY_HOURS = (23, 24, 25)  # 23 and 25 hours happen on the two daylight saving days

SUMMARY_NOTE = (
    "mean_per_hour is the average of the hourly values, NOT a daily value. "
    "total is the sum of all hours (energy series only). "
    "mean_total_per_complete_day is the average daily total over complete days (use it for 'per day'). "
    "Days with complete=false have missing hours (for example today): say so, and do not treat them as full days."
)


def summarize(points, key="value"):
    """Statistics over the whole range and per day.

    Names say what they are: mean_per_hour is the average of the hourly values, total is their sum.
    Each day knows how many hours it has and whether it is complete, so a partial day (today, or a
    day with a gap in the data) can never be mistaken for a full day.
    """
    points = [p for p in points if p[key] is not None]
    if not points:
        raise ValueError("No data for this time range. It may be in the future or not published yet.")
    values = [p[key] for p in points]

    days = {}  # "2026-10-04" -> list of values
    for p in points:
        days.setdefault(p["time"][:10], []).append(p[key])
    daily = [
        {
            "date": day,
            "hours": len(v),
            "complete": len(v) in COMPLETE_DAY_HOURS,
            "mean_per_hour": round(sum(v) / len(v), 2),
            "min": min(v),
            "max": max(v),
            "total": round(sum(v), 2),
        }
        for day, v in days.items()
    ]
    complete = [day for day in daily if day["complete"]]

    lowest = min(points, key=lambda p: p[key])
    highest = max(points, key=lambda p: p[key])
    return {
        "mean_per_hour": round(sum(values) / len(values), 2),
        "min": {"time": lowest["time"], "value": lowest[key]},
        "max": {"time": highest["time"], "value": highest[key]},
        "total": round(sum(values), 2),
        "hours": len(values),
        "complete_days": len(complete),
        "incomplete_days": [day["date"] for day in daily if not day["complete"]],
        "mean_total_per_complete_day": round(sum(day["total"] for day in complete) / len(complete), 2) if complete else None,
        "daily": daily,
        "note": SUMMARY_NOTE,
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
    Returns mean_per_hour (the average price), min, max per day and overall (hourly values for up to 2 days)."""
    raw = smard.get_series("price", date.fromisoformat(start), date.fromisoformat(end))
    return data_result(raw, start, end, "price")


@tool
def generation_load(series: str, start: str, end: str) -> dict:
    """Electricity generation by source or total consumption in Germany, in MWh per hour.
    series is one of: load, wind_onshore, wind_offshore, solar, lignite, hard_coal, natural_gas.
    Dates as YYYY-MM-DD, both included, at most 31 days.
    Returns mean_per_hour, min, max and total per day and overall, and mean_total_per_complete_day for
    questions 'per day' (hourly values for up to 2 days). Partial days are marked complete=false."""
    if series == "price":
        raise ValueError("For prices use the price_series tool.")
    raw = smard.get_series(series, date.fromisoformat(start), date.fromisoformat(end))
    return data_result(raw, start, end, series)


@tool
def weather(region: str, start: str, end: str) -> dict:
    """Hourly weather: wind speed at 100 m, solar radiation, temperature.
    region is one of: north_sea, berlin, cologne, munich.
    Dates as YYYY-MM-DD, both included, at most 31 days.
    Returns mean_per_hour, min, max per day and overall for each variable."""
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


@functools.cache
def get_reranker():
    """The reranker model, loaded once on first use."""
    from sentence_transformers import CrossEncoder  # imported here: it is heavy and optional

    return CrossEncoder(config.RERANKER_MODEL, max_length=512)


@tool
def bundestag_search(question: str, party: Optional[str] = None, paper_type: Optional[str] = None) -> dict:
    """Search Bundestag printed papers (Drucksachen) on energy topics: laws, motions,
    committee reports. Use it for what parties, the government or the Bundestag
    want, plan, criticize or decided. Returns the best matching text passages.
    If the question is about one party's position, set party to one of: afd, gruene, linke,
    bundesregierung, ausschuss. If it asks for a certain kind of paper, set paper_type to one of:
    Gesetzentwurf, Antrag, Kleine Anfrage, Antwort, Beschlussempfehlung und Bericht, Unterrichtung."""
    hits = search.search(
        get_index(CHUNK_SIZE),
        question,
        k=SEARCH_K,
        party=party if USE_FILTERS else None,
        paper_type=paper_type if USE_FILTERS else None,
        reranker=get_reranker() if USE_RERANKER else None,
    )
    # Citation as required by the DIP terms of use: paper type and number, and the source "Deutscher Bundestag/Bundesrat – DIP".
    sources = [f"Bundestags-Drucksache {h['number']} ({h['date']}), Deutscher Bundestag/Bundesrat – DIP: {h['url']}" for h in hits]
    return {"hits": hits, "sources": sorted(set(sources))}


DATA_TOOLS = [price_series, generation_load, weather]
TOOLS = DATA_TOOLS + [bundestag_search]
