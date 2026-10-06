from energiewende.agent import tools

POINTS = [
    {"time": "2026-10-04T00:00:00+02:00", "value": 100.0},
    {"time": "2026-10-04T01:00:00+02:00", "value": 200.0},
    {"time": "2026-10-05T00:00:00+02:00", "value": 60.0},
    {"time": "2026-10-05T01:00:00+02:00", "value": None},  # missing hour
]


def test_summarize_computes_the_statistics():
    result = tools.summarize(POINTS)

    assert result["mean"] == 120.0
    assert result["min"] == {"time": "2026-10-05T00:00:00+02:00", "value": 60.0}
    assert result["max"] == {"time": "2026-10-04T01:00:00+02:00", "value": 200.0}
    assert result["total"] == 360.0
    assert result["daily"] == [
        {"date": "2026-10-04", "mean": 150.0, "min": 100.0, "max": 200.0, "total": 300.0},
        {"date": "2026-10-05", "mean": 60.0, "min": 60.0, "max": 60.0, "total": 60.0},
    ]


def test_summarize_explains_when_there_is_no_data():
    try:
        tools.summarize([{"time": "2026-10-04T00:00", "value": None}])
        assert False, "should have raised"
    except ValueError as error:
        assert "No data" in str(error)


def test_summarize_can_use_another_key():
    points = [{"time": "2026-10-04T00:00", "wind_speed_100m": 10.0}, {"time": "2026-10-04T01:00", "wind_speed_100m": 20.0}]

    assert tools.summarize(points, key="wind_speed_100m")["mean"] == 15.0


def fake_series(name, start, end):
    """Two days of made-up hourly values, shaped like smard.get_series."""
    return {"series": name, "unit": "EUR/MWh", "source": "SMARD", "points": POINTS}


def test_price_series_returns_statistics_and_source(monkeypatch):
    monkeypatch.setattr(tools.smard, "get_series", fake_series)

    result = tools.price_series.invoke({"start": "2026-10-04", "end": "2026-10-05"})

    assert result["summary"]["mean"] == 120.0
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

    assert result["summary_by_variable"]["wind_speed_100m"]["mean"] == 15.0
    assert result["sources"] == ["Open-Meteo, north_sea, 2026-10-04 to 2026-10-04"]


def test_bundestag_search_returns_hits_and_sources(monkeypatch):
    hit = {"number": "21/7", "date": "2026-09-30", "title": "Titel", "url": "https://x/7.pdf", "score": 0.8, "text": "Text"}
    monkeypatch.setattr(tools, "get_index", lambda chunk_size: "fake index")
    monkeypatch.setattr(tools.search, "search", lambda index, question, k: [hit])

    result = tools.bundestag_search.invoke({"question": "Wasserstoff"})

    assert result["hits"] == [hit]
    # Attribution as required by the DIP terms of use (4c).
    assert result["sources"] == ["Bundestags-Drucksache 21/7 (2026-09-30), Deutscher Bundestag/Bundesrat – DIP: https://x/7.pdf"]


def test_bundestag_search_uses_the_chosen_index_and_k(monkeypatch):
    used = {}

    def fake_get_index(chunk_size):
        used["chunk_size"] = chunk_size
        return "fake index"

    def fake_search(index, question, k):
        used["k"] = k
        return []

    monkeypatch.setattr(tools, "get_index", fake_get_index)
    monkeypatch.setattr(tools.search, "search", fake_search)
    monkeypatch.setattr(tools, "CHUNK_SIZE", 300)
    monkeypatch.setattr(tools, "SEARCH_K", 6)

    tools.bundestag_search.invoke({"question": "Wasserstoff"})

    assert used == {"chunk_size": 300, "k": 6}
