"""The web API. Start it with:

    PYTHONPATH=src .venv/bin/uvicorn energiewende.api.main:app --reload

Then open http://127.0.0.1:8000/docs to try it in the browser.
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from energiewende.agent import graph

app = FastAPI(title="Energiewende Agent")


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class Answer(BaseModel):
    answer: str
    sources: list[str]
    tool_calls: list[dict]
    latency_ms: int
    tokens: int


@app.get("/health")
def health():
    """For Docker and Azure to check that the service is running."""
    return {"status": "ok"}


@app.post("/ask", response_model=Answer)
def ask(body: Question):
    """Ask the agent one question."""
    return graph.ask(body.question)
