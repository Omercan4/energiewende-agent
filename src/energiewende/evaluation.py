"""Score the agent's answers for the evaluation in eval/.

For each question we check four things:
- tools:   did the agent call the expected tools?
- hit:     did the Bundestag search find one of the gold papers?
- numbers: does the answer contain the right number (computed by calling the tool ourselves)?
- text:    does the answer agree with the reference facts? (an LLM judge decides)
"""

import re

NUMBER = re.compile(r"-?\d[\d.,]*\d|-?\d")


def numbers_in(text):
    """All numbers in a text. German (1.234,5) and English (1,234.5) style are both tried."""
    found = []
    for token in NUMBER.findall(text):
        german = token.replace(".", "").replace(",", ".")
        english = token.replace(",", "")
        for candidate in {german, english}:
            try:
                found.append(float(candidate))
            except ValueError:
                pass  # e.g. "15.09.2026" is not a number in English style
    return found


def number_found(expected, text, tolerance=0.01):
    """True if the text contains the expected number (within 1 %).
    It may also be written in thousands or millions, e.g. "1.168 GWh" or "1,17 Mio. MWh"."""
    for number in numbers_in(text):
        for scale in (1, 1_000, 1_000_000):
            if abs(number * scale - expected) <= tolerance * abs(expected):
                return True
    return False


def lookup(data, path):
    """lookup({"a": {"b": 1}}, "a.b") returns 1."""
    for key in path.split("."):
        data = data[key]
    return data


def mean(values):
    """Average of the values that are not None, or None if there are none."""
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def metrics(rows):
    """The scores of one run, from the per-question rows."""
    result = {
        "tool_accuracy": mean([r["tools_ok"] for r in rows]),
        "hit_rate": mean([r["hit"] for r in rows]),
        "number_accuracy": mean([r["number_ok"] for r in rows]),
        "text_accuracy": mean([r["text_ok"] for r in rows]),
        "answer_accuracy": mean([r["correct"] for r in rows]),
        "latency_ms": mean([r["latency_ms"] for r in rows]),
        "tokens": mean([r["tokens"] for r in rows]),
    }
    return {name: value for name, value in result.items() if value is not None}  # MLflow cannot log None
