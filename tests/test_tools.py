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


def test_summarize_can_use_another_key():
    points = [{"time": "2026-10-04T00:00", "wind_speed_100m": 10.0}, {"time": "2026-10-04T01:00", "wind_speed_100m": 20.0}]

    assert tools.summarize(points, key="wind_speed_100m")["mean"] == 15.0
