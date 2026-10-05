from datetime import date

from langchain_core.messages import AIMessage

from energiewende.agent import graph, tools


class FakeLLM:
    """Plays back prepared replies instead of calling a real model."""

    def __init__(self, replies):
        self.replies = list(replies)

    def bind_tools(self, tool_list):
        return self

    def invoke(self, messages):
        return self.replies.pop(0)


def usage(total):
    return {"input_tokens": total - 10, "output_tokens": 10, "total_tokens": total}


def fake_series(name, start, end):
    points = [{"time": "2026-10-04T00:00:00+02:00", "value": 100.0}, {"time": "2026-10-04T01:00:00+02:00", "value": 200.0}]
    return {"series": name, "unit": "EUR/MWh", "source": "SMARD", "points": points}


def test_agent_calls_a_tool_and_answers_with_sources(monkeypatch):
    monkeypatch.setattr(tools.smard, "get_series", fake_series)
    llm = FakeLLM([
        AIMessage(content="", tool_calls=[{"name": "price_series", "args": {"start": "2026-10-04", "end": "2026-10-04"}, "id": "1"}], usage_metadata=usage(100)),
        AIMessage(content="Im Mittel 150 EUR/MWh.", usage_metadata=usage(300)),
    ])
    agent = graph.build_agent(llm, tools.TOOLS)

    result = graph.ask("Wie hoch war der Strompreis gestern?", agent=agent, today=date(2026, 10, 5))

    assert result["answer"] == "Im Mittel 150 EUR/MWh."
    assert result["tool_calls"] == [{"name": "price_series", "args": {"start": "2026-10-04", "end": "2026-10-04"}}]
    assert result["sources"] == ["SMARD, price, 2026-10-04 to 2026-10-04"]
    assert result["tokens"] == 400
    assert result["latency_ms"] >= 0


def test_tool_errors_go_back_to_the_model(monkeypatch):
    def broken_series(name, start, end):
        raise ValueError("At most 31 days per request")

    monkeypatch.setattr(tools.smard, "get_series", broken_series)
    llm = FakeLLM([
        AIMessage(content="", tool_calls=[{"name": "price_series", "args": {"start": "2026-01-01", "end": "2026-06-30"}, "id": "1"}]),
        AIMessage(content="Ich kann höchstens 31 Tage abfragen."),
    ])

    result = graph.ask("Preis im ersten Halbjahr?", agent=graph.build_agent(llm, tools.TOOLS))

    assert result["answer"] == "Ich kann höchstens 31 Tage abfragen."
    assert result["sources"] == []  # the failed call has no sources


def test_too_many_steps_gives_a_clear_message(monkeypatch):
    monkeypatch.setattr(tools.smard, "get_series", fake_series)
    call = {"name": "price_series", "args": {"start": "2026-10-04", "end": "2026-10-04"}}
    llm = FakeLLM([AIMessage(content="", tool_calls=[{**call, "id": str(i)}]) for i in range(20)])

    result = graph.ask("Endlosschleife", agent=graph.build_agent(llm, tools.TOOLS))

    assert "too many steps" in result["answer"]
