from langchain_core.messages import AIMessage

from energiewende import evaluation
from energiewende.agent import tools


def test_numbers_in_reads_german_and_english_style():
    numbers = evaluation.numbers_in("Im Mittel 168,29 EUR/MWh, insgesamt 1.168.081,41 MWh, Minimum -5.2")

    assert 168.29 in numbers
    assert 1168081.41 in numbers
    assert -5.2 in numbers


def test_number_found_allows_thousands_and_millions():
    assert evaluation.number_found(1168081.41, "Es wurden 1.168.081 MWh erzeugt.")
    assert evaluation.number_found(1168081.41, "Es wurden rund 1,17 Mio. MWh erzeugt.")
    assert evaluation.number_found(1168081.41, "Das sind 1.168 GWh.")


def test_number_found_rejects_wrong_numbers():
    assert not evaluation.number_found(168.29, "Der Preis lag bei 155,78 EUR/MWh.")


def test_lookup_walks_a_path():
    assert evaluation.lookup({"summary": {"max": {"value": 7}}}, "summary.max.value") == 7


def test_metrics_skip_missing_values():
    rows = [
        {"tools_ok": True, "hit": True, "number_ok": None, "text_ok": True, "correct": True, "latency_ms": 100, "tokens": 1000},
        {"tools_ok": False, "hit": None, "number_ok": False, "text_ok": None, "correct": False, "latency_ms": 300, "tokens": 3000},
    ]

    result = evaluation.metrics(rows)

    assert result == {
        "tool_accuracy": 0.5,
        "hit_rate": 1.0,
        "number_accuracy": 0.0,
        "text_accuracy": 1.0,
        "answer_accuracy": 0.5,
        "latency_ms": 200.0,
        "tokens": 2000.0,
    }


class FakeJudge:
    """Returns a prepared verdict instead of calling a real model."""

    def __init__(self, reply):
        self.reply = reply

    def invoke(self, messages):
        return self.reply


def verdict(correct):
    return AIMessage(content="", tool_calls=[{"name": "grade", "args": {"correct": correct, "reason": "because"}, "id": "1"}])


MIXED = {
    "id": "m01",
    "type": "mixed",
    "question": "Strompreis am 15.09.2026 und was will die Linke zur Stromsteuer?",
    "tools": ["price_series", "bundestag_search"],
    "check": {"tool": "price_series", "args": {"start": "2026-09-15", "end": "2026-09-15"}, "value": "summary.mean"},
    "gold": ["21/4273"],
    "facts": "Die Linke will die Stromsteuer auf das EU-Minimum senken.",
}

RESULT = {
    "answer": "Der Preis lag bei 120,50 EUR/MWh. Die Linke will die Stromsteuer auf das EU-Minimum senken.",
    "sources": ["SMARD, price, 2026-09-15 to 2026-09-15", "Bundestag Drucksache 21/4273 (2026-02-24): https://x/4273.pdf"],
    "tool_calls": [{"name": "price_series", "args": {}}, {"name": "bundestag_search", "args": {}}],
    "latency_ms": 100,
    "tokens": 1000,
}


def test_judge_reads_the_grade_tool_call():
    assert evaluation.judge("Frage", "Fakten", "Antwort", FakeJudge(verdict(True))) == (True, "because")


def test_judge_without_verdict_counts_as_wrong():
    ok, reason = evaluation.judge("Frage", "Fakten", "Antwort", FakeJudge(AIMessage(content="Looks fine")))

    assert ok is False
    assert "no verdict" in reason


def test_score_question_checks_tools_hit_number_and_text(monkeypatch):
    monkeypatch.setattr(evaluation, "expected_number", lambda check: 120.5)

    row = evaluation.score_question(MIXED, RESULT, rag=True, judge_llm=FakeJudge(verdict(True)))

    assert row["tools_ok"] is True
    assert row["hit"] is True
    assert row["number_ok"] is True
    assert row["text_ok"] is True
    assert row["correct"] is True


def test_without_rag_the_search_is_not_expected(monkeypatch):
    monkeypatch.setattr(evaluation, "expected_number", lambda check: 120.5)
    result = {**RESULT, "tool_calls": [{"name": "price_series", "args": {}}], "sources": []}

    row = evaluation.score_question(MIXED, result, rag=False, judge_llm=FakeJudge(verdict(False)))

    assert row["tools_ok"] is True  # bundestag_search was not available
    assert row["hit"] is None
    assert row["correct"] is False  # the text part is wrong


def test_questions_file_is_complete():
    questions = evaluation.load_questions()

    assert len(questions) == 30
    assert len({q["id"] for q in questions}) == 30
    assert sorted(q["type"] for q in questions) == ["mixed"] * 10 + ["number"] * 10 + ["text"] * 10
    for q in questions:
        assert set(q["tools"]) <= {t.name for t in tools.TOOLS}
        if q["type"] in ("number", "mixed"):
            assert q["check"]["tool"] in q["tools"]
            assert all(isinstance(value, str) for value in q["check"]["args"].values())  # dates must be quoted
        if q["type"] in ("text", "mixed"):
            assert q["gold"] and q["facts"]
            assert "bundestag_search" in q["tools"]


def test_judge_knows_todays_date_and_ignores_numbers():
    class RecordingJudge(FakeJudge):
        def invoke(self, messages):
            self.messages = messages
            return self.reply

    recording = RecordingJudge(verdict(True))
    evaluation.judge("Frage", "Fakten", "Antwort", recording, today="2026-10-05")

    prompt = recording.messages[0].content
    assert "2026-10-05" in prompt
    assert "numbers" in prompt
