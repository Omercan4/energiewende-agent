"""The tools the agent can use.

Each tool returns a small dict. The data tools compute mean, min, max and total
in Python, because LLMs make mistakes when they add up many numbers.
Every result has a "sources" list, so the answer can name where the data is from.
"""


def summarize(points, key="value"):
    """Mean, min, max and total of all points, and the same per day."""
    points = [p for p in points if p[key] is not None]
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
