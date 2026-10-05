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
