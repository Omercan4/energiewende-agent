"""The web API. Start it with:

    PYTHONPATH=src .venv/bin/uvicorn energiewende.api.main:app --reload

Then open http://127.0.0.1:8000/docs to try it in the browser.
"""

import secrets

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from energiewende import config
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
def ask(body: Question, x_api_key: str | None = Header(default=None)):
    """Ask the agent one question. Needs the header X-API-Key if the server has APP_API_KEY set."""
    if config.APP_API_KEY and not secrets.compare_digest(x_api_key or "", config.APP_API_KEY):
        raise HTTPException(status_code=401, detail="Missing or wrong X-API-Key header")
    return graph.ask(body.question)
