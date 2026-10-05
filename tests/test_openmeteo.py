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
