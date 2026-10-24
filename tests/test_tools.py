from energiewende.agent import tools

POINTS = [
    {"time": "2026-10-04T00:00:00+02:00", "value": 100.0},
    {"time": "2026-10-04T01:00:00+02:00", "value": 200.0},
    {"time": "2026-10-05T00:00:00+02:00", "value": 60.0},
    {"time": "2026-10-05T01:00:00+02:00", "value": None},  # missing hour
]


def test_summarize_computes_the_statistics():
    result = tools.summarize(POINTS)

    assert result["mean_per_hour"] == 120.0
    assert result["min"] == {"time": "2026-10-05T00:00:00+02:00", "value": 60.0}
    assert result["max"] == {"time": "2026-10-04T01:00:00+02:00", "value": 200.0}
    assert result["total"] == 360.0
    assert result["hours"] == 3
    assert result["daily"] == [
        {"date": "2026-10-04", "hours": 2, "complete": False, "mean_per_hour": 150.0, "min": 100.0, "max": 200.0, "total": 300.0},
        {"date": "2026-10-05", "hours": 1, "complete": False, "mean_per_hour": 60.0, "min": 60.0, "max": 60.0, "total": 60.0},
    ]


def full_day(day, value):
    return [{"time": f"{day}T{hour:02d}:00:00+02:00", "value": value} for hour in range(24)]


def test_a_full_day_is_complete_and_has_a_daily_total():
    result = tools.summarize(full_day("2026-10-04", 10.0) + full_day("2026-10-05", 20.0))

    assert result["mean_per_hour"] == 15.0
    assert result["complete_days"] == 2
    assert result["incomplete_days"] == []
    assert result["mean_total_per_complete_day"] == 360.0  # (240 + 480) / 2, the average DAY, not the average hour
    assert result["daily"][0]["hours"] == 24 and result["daily"][0]["complete"] is True
    assert result["daily"][0]["total"] == 240.0


def test_a_partial_day_is_flagged_and_left_out_of_the_daily_average():
    today = full_day("2026-10-05", 20.0)[:9]  # only 9 hours are published so far
    result = tools.summarize(full_day("2026-10-04", 10.0) + today)

    assert result["incomplete_days"] == ["2026-10-05"]
    assert result["complete_days"] == 1
    assert result["mean_total_per_complete_day"] == 240.0  # only the complete day counts
    assert result["daily"][1]["complete"] is False
    assert "complete" in result["note"].lower()


def test_daylight_saving_days_with_23_or_25_hours_count_as_complete():
    short = full_day("2026-03-29", 1.0)[:23]
    long = full_day("2026-10-25", 1.0) + [{"time": "2026-10-25T24:00:00+01:00", "value": 1.0}]

    assert tools.summarize(short)["daily"][0]["complete"] is True
    assert tools.summarize(long)["daily"][0]["complete"] is True


def test_no_complete_day_means_no_daily_average():
    result = tools.summarize(POINTS)

    assert result["mean_total_per_complete_day"] is None


def test_summarize_explains_when_there_is_no_data():
    try:
        tools.summarize([{"time": "2026-10-04T00:00", "value": None}])
        assert False, "should have raised"
    except ValueError as error:
        assert "No data" in str(error)


def test_summarize_can_use_another_key():
    points = [{"time": "2026-10-04T00:00", "wind_speed_100m": 10.0}, {"time": "2026-10-04T01:00", "wind_speed_100m": 20.0}]

    assert tools.summarize(points, key="wind_speed_100m")["mean_per_hour"] == 15.0


def fake_series(name, start, end):
    """Two days of made-up hourly values, shaped like smard.get_series."""
    return {"series": name, "unit": "EUR/MWh", "source": "SMARD", "points": POINTS}


def test_price_series_returns_statistics_and_source(monkeypatch):
    monkeypatch.setattr(tools.smard, "get_series", fake_series)

    result = tools.price_series.invoke({"start": "2026-10-04", "end": "2026-10-05"})

    assert result["summary"]["mean_per_hour"] == 120.0
    assert result["hourly"] == POINTS  # short range, so hourly values are included
    assert result["sources"] == ["SMARD, price, 2026-10-04 to 2026-10-05"]


def test_long_ranges_leave_out_the_hourly_values(monkeypatch):
    monkeypatch.setattr(tools.smard, "get_series", fake_series)

    result = tools.price_series.invoke({"start": "2026-09-01", "end": "2026-09-30"})

    assert "hourly" not in result


def test_generation_load_rejects_price(monkeypatch):
    monkeypatch.setattr(tools.smard, "get_series", fake_series)

    try:
        tools.generation_load.invoke({"series": "price", "start": "2026-10-04", "end": "2026-10-04"})
        assert False, "should have raised"
    except ValueError as error:
        assert "price_series" in str(error)


def test_weather_summarizes_each_variable(monkeypatch):
    def fake_weather(region, start, end):
        return {
            "region": region,
            "units": {"wind_speed_100m": "km/h"},
            "source": "Open-Meteo",
            "points": [{"time": "2026-10-04T00:00", "wind_speed_100m": 10.0}, {"time": "2026-10-04T01:00", "wind_speed_100m": 20.0}],
        }

    monkeypatch.setattr(tools.openmeteo, "get_weather", fake_weather)
    monkeypatch.setattr(tools.openmeteo, "VARIABLES", ["wind_speed_100m"])

    result = tools.weather.invoke({"region": "north_sea", "start": "2026-10-04", "end": "2026-10-04"})

    assert result["summary_by_variable"]["wind_speed_100m"]["mean_per_hour"] == 15.0
    assert result["sources"] == ["Open-Meteo, north_sea, 2026-10-04 to 2026-10-04"]


def test_bundestag_search_returns_hits_and_sources(monkeypatch):
    hit = {"number": "21/7", "date": "2026-09-30", "title": "Titel", "url": "https://x/7.pdf", "score": 0.8, "text": "Text"}
    monkeypatch.setattr(tools, "get_index", lambda chunk_size: "fake index")
    monkeypatch.setattr(tools.search, "search", lambda index, question, k, **kwargs: [hit])

    result = tools.bundestag_search.invoke({"question": "Wasserstoff"})

    assert result["hits"] == [hit]
    # Attribution as required by the DIP terms of use (4c).
    assert result["sources"] == ["Bundestags-Drucksache 21/7 (2026-09-30), Deutscher Bundestag/Bundesrat – DIP: https://x/7.pdf"]


def test_bundestag_search_uses_the_chosen_index_and_k(monkeypatch):
    used = {}

    def fake_get_index(chunk_size):
        used["chunk_size"] = chunk_size
        return "fake index"

    def fake_search(index, question, k, **kwargs):
        used["k"] = k
        return []

    monkeypatch.setattr(tools, "get_index", fake_get_index)
    monkeypatch.setattr(tools.search, "search", fake_search)
    monkeypatch.setattr(tools, "CHUNK_SIZE", 300)
    monkeypatch.setattr(tools, "SEARCH_K", 6)

    tools.bundestag_search.invoke({"question": "Wasserstoff"})

    assert used == {"chunk_size": 300, "k": 6}


def test_bundestag_search_passes_party_and_paper_type_on(monkeypatch):
    seen = {}

    def fake_search(index, question, k, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(tools, "get_index", lambda chunk_size: "fake index")
    monkeypatch.setattr(tools.search, "search", fake_search)
    monkeypatch.setattr(tools, "USE_FILTERS", True)
    monkeypatch.setattr(tools, "USE_RERANKER", False)

    tools.bundestag_search.invoke({"question": "EEG", "party": "afd", "paper_type": "Antrag"})

    assert seen == {"party": "afd", "paper_type": "Antrag", "reranker": None}


def test_filters_can_be_switched_off_for_the_evaluation(monkeypatch):
    seen = {}

    def fake_search(index, question, k, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(tools, "get_index", lambda chunk_size: "fake index")
    monkeypatch.setattr(tools.search, "search", fake_search)
    monkeypatch.setattr(tools, "USE_FILTERS", False)
    monkeypatch.setattr(tools, "USE_RERANKER", False)

    tools.bundestag_search.invoke({"question": "EEG", "party": "afd"})

    assert seen["party"] is None and seen["paper_type"] is None


FORECAST = {
    "target_day": "2026-10-25",
    "unit": "EUR/MWh",
    "summary": {"mean_q50": 22.5, "min_q50": -2.3, "max_q50": 53.6, "cheapest_hour": 5, "most_expensive_hour": 18},
    "hours": [{"time": "t", "hour": 5, "q10": -10.0, "q50": -2.3, "q90": 80.0, "temp": 1.0}],
    "model": {"backtest_mae": 19.58, "trained_until": "x"},
    "note": "Forecast made with information known at 12:00 the day before. Not trading advice.",
}


def test_price_forecast_returns_the_summary_the_interval_and_a_source(monkeypatch):
    monkeypatch.setattr(tools, "get_json", lambda url, max_age_hours: FORECAST)

    result = tools.price_forecast.invoke({})

    assert result["summary"]["cheapest_hour"] == 5
    assert result["hours"] == [{"hour": 5, "q10": -10.0, "q50": -2.3, "q90": 80.0}]  # only what the model needs
    assert "Not trading advice" in result["note"]
    assert "2026-10-25" in result["sources"][0] and "SMARD" in result["sources"][0]


def test_price_forecast_is_one_of_the_data_tools():
    assert tools.price_forecast in tools.DATA_TOOLS
