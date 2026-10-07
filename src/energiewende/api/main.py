"""The web API. Start it with:

    PYTHONPATH=src .venv/bin/uvicorn energiewende.api.main:app --reload

Then open http://127.0.0.1:8000 for the chat page, or /docs for the API.
"""

import secrets
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from energiewende import config
from energiewende.agent import graph

app = FastAPI(title="Energiewende Agent")

CHAT_PAGE = Path(__file__).parent / "static" / "index.html"


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class Answer(BaseModel):
    answer: str
    sources: list[str]
    tool_calls: list[dict]
    latency_ms: int
    tokens: int


@app.get("/", include_in_schema=False)
def chat_page():
    """The chat page for users. Developers use /docs."""
    return FileResponse(CHAT_PAGE)


@app.get("/health")
def health():
    """For Docker and Azure to check that the service is running."""
    return {"status": "ok"}


def check_key(x_api_key):
    """Stop the request with 401 if the server has APP_API_KEY set and the given key is wrong."""
    if config.APP_API_KEY and not secrets.compare_digest(x_api_key or "", config.APP_API_KEY):
        raise HTTPException(status_code=401, detail="Missing or wrong X-API-Key header")


@app.get("/check-access")
def check_access(x_api_key: str | None = Header(default=None)):
    """Check an access code without asking the LLM (used by the start screen of the chat page)."""
    check_key(x_api_key)
    return {"ok": True}


@app.post("/ask", response_model=Answer)
def ask(body: Question, x_api_key: str | None = Header(default=None)):
    """Ask the agent one question. Needs the header X-API-Key if the server has APP_API_KEY set."""
    check_key(x_api_key)
    return graph.ask(body.question)
