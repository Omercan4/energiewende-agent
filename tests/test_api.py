from fastapi.testclient import TestClient

from energiewende import config
from energiewende.agent import graph
from energiewende.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_the_agent_result(monkeypatch):
    result = {"answer": "150 EUR/MWh", "sources": ["SMARD"], "tool_calls": [], "latency_ms": 5, "tokens": 10}
    monkeypatch.setattr(graph, "ask", lambda question: result)

    response = client.post("/ask", json={"question": "Strompreis gestern?"})

    assert response.status_code == 200
    assert response.json() == result


def test_empty_question_is_rejected():
    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422


def test_ask_needs_the_key_when_one_is_set(monkeypatch):
    monkeypatch.setattr(config, "APP_API_KEY", "secret")
    monkeypatch.setattr(graph, "ask", lambda question: {"answer": "x", "sources": [], "tool_calls": [], "latency_ms": 1, "tokens": 1})

    assert client.post("/ask", json={"question": "Hallo"}).status_code == 401
    assert client.post("/ask", json={"question": "Hallo"}, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/ask", json={"question": "Hallo"}, headers={"X-API-Key": "secret"}).status_code == 200


def test_health_needs_no_key(monkeypatch):
    monkeypatch.setattr(config, "APP_API_KEY", "secret")

    assert client.get("/health").status_code == 200


def test_main_page_is_the_chat_page():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Zugangscode" in response.text


def test_check_access_accepts_only_the_right_key(monkeypatch):
    monkeypatch.setattr(config, "APP_API_KEY", "secret")

    assert client.get("/check-access").status_code == 401
    assert client.get("/check-access", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/check-access", headers={"X-API-Key": "secret"}).json() == {"ok": True}


def test_check_access_is_open_when_no_key_is_set(monkeypatch):
    monkeypatch.setattr(config, "APP_API_KEY", "")

    assert client.get("/check-access").json() == {"ok": True}
