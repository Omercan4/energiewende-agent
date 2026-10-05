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
