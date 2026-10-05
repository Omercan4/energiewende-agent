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
