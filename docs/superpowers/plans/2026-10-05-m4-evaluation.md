# M4 Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how good the agent is on 30 fixed questions, for several settings, and keep every run in MLflow so the settings can be compared.

**Architecture:** `eval/questions.yaml` holds 30 questions (10 number, 10 text, 10 mixed) with the expected tools, a number check (which tool call gives the right number), the gold Bundestag papers and reference facts. `src/energiewende/evaluation.py` scores one answer: right tools? gold paper found? number in the answer? an LLM judge for the text part. `eval/run_eval.py` runs all questions for one setting and logs parameters, metrics and the per-question table to MLflow. `eval/compare.py` turns all runs into a Markdown table.

**Tech Stack:** MLflow (local SQLite store), PyYAML, LangChain `ChatOpenAI` for the judge.

**Spec:** `docs/superpowers/specs/2026-10-05-energiewende-agent-design.md` (section 6, milestone M4)

This is plan 4 of 6. It builds on M2 (`build_index.py`, one index per chunk size) and M3 (`tools`, `graph.build_agent`, `graph.ask`).

## Global Constraints

- Code style: as simple and plain as possible. Plain functions, plain dicts, short comments in simple English.
- Pinned versions (installed and checked on 2026-10-05): `mlflow==3.16.1`, `pyyaml==6.0.3`.
- Settings from environment variables: `JUDGE_MODEL` (default `claude-haiku-4.5`), `MLFLOW_TRACKING_URI` (default `sqlite:///mlflow.db`). `mlflow.db` and `mlruns/` stay git-ignored.
- Every reference fact in `questions.yaml` was read in the paper itself (2026-10-05). Do not add facts that were not checked in the text.
- Expected numbers are not typed in by hand: they are computed by calling the tool with the arguments in the question's `check`.
- Tests never call a real LLM or a real API. Evaluation runs are manual, not part of CI.
- Every commit message ends with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Verified facts these tasks rely on (checked 2026-10-05)

- MLflow 3.16.1: `mlflow.set_tracking_uri("sqlite:///mlflow.db")`, `set_experiment`, `start_run(run_name=...)`, `log_params`, `log_metrics`, `log_table(dict_of_lists, artifact_file="x.json")` and `search_runs(experiment_names=[...])` (columns `tags.mlflow.runName`, `params.k`, `metrics.<name>`) all work. It creates `mlflow.db` and an `mlruns/` folder for artifacts.
- **The gateway adds about 110,000 hidden input tokens to any request that carries no tools** (measured: gemini-2.5-flash 113,193 input tokens for a yes/no question; with one tool bound: 25). The agent always sends tools, so it is not affected. The judge therefore also gets a tool: it answers by calling `grade(correct, reason)`.
- The gateway only allows `tool_choice="auto"` (forcing a tool gives HTTP 400). With `auto`, `claude-haiku-4.5` called `grade` every time in the test, judged a correct and a wrong answer correctly, and used about 800 tokens.
- The judge model (`claude-haiku-4.5`) is from a different family than the agent model (`gemini-2.5-flash`), so the agent's model does not grade itself.
- These follow-up papers refer to the gold papers (checked in their title or first 3,000 characters) and count as hits too: 21/3203 and 21/4326 → 21/2506; 21/2075 and 21/3078 → 21/1491; 21/6563 and 21/6998 → 21/6279; 21/4873 → 21/4461; 21/468 → 21/224; 21/6679 → 21/6332 and 21/6333.
- Number questions use September 2026, so the data is final and the Open-Meteo archive is used.
- YAML reads `2026-09-15` as a date object; the tools need strings, so dates in `questions.yaml` are quoted.

## File Structure

```
requirements.txt                    + mlflow, pyyaml
.gitignore                          + mlflow.db
.env.example, config.py             + JUDGE_MODEL, MLFLOW_TRACKING_URI
src/energiewende/agent/tools.py     + CHUNK_SIZE; get_index(chunk_size)
src/energiewende/evaluation.py      number check, judge, scoring, metrics
eval/questions.yaml                 the 30 questions
eval/run_eval.py                    one run for one setting -> MLflow
eval/compare.py                     all runs -> eval/results.md
tests/test_evaluation.py
```

---

### Task 1: Settings and choosing the index by chunk size

**Files:**
- Modify: `requirements.txt`, `.gitignore`, `.env.example`, `src/energiewende/config.py`, `src/energiewende/agent/tools.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Produces: `config.JUDGE_MODEL: str`, `config.MLFLOW_TRACKING_URI: str`; `tools.CHUNK_SIZE = 400`; `tools.get_index(chunk_size) -> VectorStoreIndex` (cached per chunk size). `bundestag_search` uses `get_index(CHUNK_SIZE)` and `SEARCH_K`.

- [ ] **Step 1: Add the packages and settings**

Append to `requirements.txt`:

```
mlflow==3.16.1
pyyaml==6.0.3
```

Append to `.gitignore`:

```
mlflow.db
```

Append to `.env.example`:

```

# Evaluation: the model that grades text answers, and where MLflow stores runs
JUDGE_MODEL=claude-haiku-4.5
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

Append to `src/energiewende/config.py`:

```python

# The model that grades text answers in the evaluation (another model family than the agent).
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-haiku-4.5")

# Where MLflow saves evaluation runs.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
```

Run: `.venv/bin/pip install -r requirements.txt`
Expected: finishes without errors.

- [ ] **Step 2: Write the failing test**

In `tests/test_tools.py`, in `test_bundestag_search_returns_hits_and_sources`, change the line

```python
    monkeypatch.setattr(tools, "get_index", lambda: "fake index")
```

to

```python
    monkeypatch.setattr(tools, "get_index", lambda chunk_size: "fake index")
```

Append to `tests/test_tools.py`:

```python
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
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `.venv/bin/pytest tests/test_tools.py -v`
Expected: the two Bundestag tests FAIL (`TypeError` from calling `get_index` without/with an argument).

- [ ] **Step 4: Change `tools.py`**

In `src/energiewende/agent/tools.py`, below the line `SEARCH_K = 4  # how many text passages bundestag_search returns`, add:

```python
CHUNK_SIZE = 400  # which search index bundestag_search uses (there is one per chunk size)
```

Replace the `get_index` function with:

```python
@functools.cache
def get_index(chunk_size):
    """Load the search index for this chunk size once, on first use."""
    return store.load_index(chunk_size)
```

In `bundestag_search`, replace `search.search(get_index(), question, k=SEARCH_K)` with `search.search(get_index(CHUNK_SIZE), question, k=SEARCH_K)`.

- [ ] **Step 5: Run the tests to see them pass**

Run: `.venv/bin/pytest -q`
Expected: 39 passed.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore .env.example src/energiewende/config.py src/energiewende/agent/tools.py tests/test_tools.py
git commit -m "feat: evaluation settings and index choice by chunk size" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Number check and metrics

**Files:**
- Create: `src/energiewende/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Produces:
  - `evaluation.numbers_in(text: str) -> list[float]`
  - `evaluation.number_found(expected: float, text: str, tolerance: float = 0.01) -> bool`
  - `evaluation.lookup(data: dict, path: str)` (`"summary.max.value"` walks nested dicts)
  - `evaluation.metrics(rows: list[dict]) -> dict[str, float]` with keys `tool_accuracy, hit_rate, number_accuracy, text_accuracy, answer_accuracy, latency_ms, tokens`; a key is left out when it has no values (MLflow cannot log `None`).

- [ ] **Step 1: Write the failing tests**

`tests/test_evaluation.py`:

```python
from energiewende import evaluation


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
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/pytest tests/test_evaluation.py -v`
Expected: FAIL with `ImportError: cannot import name 'evaluation'`.

- [ ] **Step 3: Write the helpers**

`src/energiewende/evaluation.py`:

```python
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
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/pytest tests/test_evaluation.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/energiewende/evaluation.py tests/test_evaluation.py
git commit -m "feat: number check and metrics for the evaluation" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: The judge and scoring one question

**Files:**
- Modify: `src/energiewende/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `tools.TOOLS` (to compute expected numbers), `config.LLM_BASE_URL`, `config.LLM_API_KEY`, `config.JUDGE_MODEL`.
- Produces:
  - `evaluation.expected_number(check: dict) -> float`, where `check = {"tool": str, "args": dict, "value": "summary.mean"}`
  - `evaluation.get_judge()` (chat model with the `grade` tool bound)
  - `evaluation.judge(question, facts, answer, judge_llm) -> (bool, str)`
  - `evaluation.score_question(question: dict, result: dict, rag: bool, judge_llm) -> dict` (one row: `id, type, question, answer, tools_called, tools_ok, hit, number_ok, text_ok, judge_reason, correct, latency_ms, tokens`)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evaluation.py`:

```python
from langchain_core.messages import AIMessage


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
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/pytest tests/test_evaluation.py -v`
Expected: 5 passed, 4 failed with `AttributeError: module 'energiewende.evaluation' has no attribute ...`.

- [ ] **Step 3: Write the judge and the scoring**

In `src/energiewende/evaluation.py`, replace the line `import re` with:

```python
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from energiewende import config
from energiewende.agent import tools

TOOLS_BY_NAME = {t.name: t for t in tools.TOOLS}

JUDGE_PROMPT = """You grade the answers of an assistant for questions about German energy policy.
Compare the answer with the reference facts.
correct = true if the answer contains the main point of the reference facts and does not contradict them.
Extra details are fine. The answer may be in German.
Always reply by calling the grade tool."""
```

Append to `src/energiewende/evaluation.py`:

```python


def expected_number(check):
    """The right number for a question: call the tool ourselves and read the value."""
    result = TOOLS_BY_NAME[check["tool"]].invoke(check["args"])
    return lookup(result, check["value"])


@tool
def grade(correct: bool, reason: str) -> str:
    """Report the verdict: correct=true if the answer agrees with the reference facts."""
    return reason


def get_judge():
    """The judge model. It answers by calling the grade tool, so we get a clear yes or no.
    (Sending a tool also keeps the request small on gateways that add a long default prompt otherwise.)"""
    llm = ChatOpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        model=config.JUDGE_MODEL,
        temperature=0,
        timeout=60,
    )
    return llm.bind_tools([grade])


def judge(question, facts, answer, judge_llm):
    """Ask the judge if the answer agrees with the reference facts. Returns (ok, reason)."""
    reply = judge_llm.invoke([
        SystemMessage(JUDGE_PROMPT),
        HumanMessage(f"Question: {question}\nReference facts: {facts}\nAnswer: {answer}"),
    ])
    if not reply.tool_calls:
        return False, "The judge gave no verdict."
    args = reply.tool_calls[0]["args"]
    return bool(args["correct"]), args.get("reason", "")


def score_question(question, result, rag, judge_llm):
    """One row of the results table: what the agent did and what was right."""
    called = {call["name"] for call in result["tool_calls"]}
    expected = set(question["tools"])
    if not rag:
        expected.discard("bundestag_search")  # the agent did not have this tool

    row = {
        "id": question["id"],
        "type": question["type"],
        "question": question["question"],
        "answer": result["answer"],
        "tools_called": ", ".join(sorted(called)),
        "tools_ok": expected <= called,
        "hit": None,
        "number_ok": None,
        "text_ok": None,
        "judge_reason": "",
        "latency_ms": result["latency_ms"],
        "tokens": result["tokens"],
    }
    if rag and question.get("gold"):
        row["hit"] = any(f"Drucksache {gold} (" in source for gold in question["gold"] for source in result["sources"])
    if question.get("check"):
        row["number_ok"] = number_found(expected_number(question["check"]), result["answer"])
    if question.get("facts"):
        row["text_ok"], row["judge_reason"] = judge(question["question"], question["facts"], result["answer"], judge_llm)
    row["correct"] = all(ok for ok in (row["number_ok"], row["text_ok"]) if ok is not None)
    return row
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/pytest tests/test_evaluation.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/energiewende/evaluation.py tests/test_evaluation.py
git commit -m "feat: LLM judge and per-question scoring" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The question set and the run script

**Files:**
- Create: `eval/questions.yaml`, `eval/run_eval.py`, `eval/compare.py`
- Modify: `src/energiewende/evaluation.py` (add `load_questions`)
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Produces: `evaluation.load_questions() -> list[dict]` (reads `eval/questions.yaml` relative to the project folder); `eval/run_eval.py --rag on|off --k N --chunk-size N [--limit N]`; `eval/compare.py` writes `eval/results.md`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_evaluation.py`:

```python
from energiewende.agent import tools


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
```

- [ ] **Step 2: Run the test to see it fail**

Run: `.venv/bin/pytest tests/test_evaluation.py -v`
Expected: 1 failed with `AttributeError: module 'energiewende.evaluation' has no attribute 'load_questions'`.

- [ ] **Step 3: Write `load_questions` and the question file**

In `src/energiewende/evaluation.py`, add below `import re`:

```python
from pathlib import Path

import yaml
```

and below `TOOLS_BY_NAME = ...`:

```python
QUESTIONS_FILE = Path(__file__).parents[2] / "eval" / "questions.yaml"
```

Append to `src/energiewende/evaluation.py`:

```python


def load_questions():
    """The evaluation questions from eval/questions.yaml."""
    return yaml.safe_load(QUESTIONS_FILE.read_text())
```

`eval/questions.yaml`:

```yaml
# The 30 evaluation questions.
#
# type    number = answered with data tools, text = answered with the Bundestag search, mixed = both
# tools   the tools the agent should call
# check   how we get the right number: call this tool with these args and read this value
# gold    the Bundestag papers that contain the answer (follow-up papers on the same bill count too)
# facts   the main point of the answer, read in the paper itself; an LLM judge compares the answer with it
#
# Dates are quoted, so YAML keeps them as text.

# ---------- number questions ----------

- id: n01
  type: number
  question: Wie hoch war der durchschnittliche Day-Ahead-Strompreis am 15. September 2026?
  tools: [price_series]
  check: {tool: price_series, args: {start: "2026-09-15", end: "2026-09-15"}, value: summary.mean}

- id: n02
  type: number
  question: Wie hoch war der höchste Strompreis am 20. September 2026?
  tools: [price_series]
  check: {tool: price_series, args: {start: "2026-09-20", end: "2026-09-20"}, value: summary.max.value}

- id: n03
  type: number
  question: Was war der niedrigste Strompreis in der Woche vom 7. bis 13. September 2026?
  tools: [price_series]
  check: {tool: price_series, args: {start: "2026-09-07", end: "2026-09-13"}, value: summary.min.value}

- id: n04
  type: number
  question: Wie viel Solarstrom wurde am 1. September 2026 in Deutschland erzeugt?
  tools: [generation_load]
  check: {tool: generation_load, args: {series: solar, start: "2026-09-01", end: "2026-09-01"}, value: summary.total}

- id: n05
  type: number
  question: Wie viel Strom haben Windkraftanlagen an Land am 10. September 2026 erzeugt?
  tools: [generation_load]
  check: {tool: generation_load, args: {series: wind_onshore, start: "2026-09-10", end: "2026-09-10"}, value: summary.total}

- id: n06
  type: number
  question: Wie hoch war der durchschnittliche stündliche Stromverbrauch in Deutschland am 16. September 2026?
  tools: [generation_load]
  check: {tool: generation_load, args: {series: load, start: "2026-09-16", end: "2026-09-16"}, value: summary.mean}

- id: n07
  type: number
  question: Wie viel Strom wurde vom 21. bis 27. September 2026 aus Braunkohle erzeugt?
  tools: [generation_load]
  check: {tool: generation_load, args: {series: lignite, start: "2026-09-21", end: "2026-09-27"}, value: summary.total}

- id: n08
  type: number
  question: Wie hoch war die höchste stündliche Erzeugung aus Offshore-Wind am 5. September 2026?
  tools: [generation_load]
  check: {tool: generation_load, args: {series: wind_offshore, start: "2026-09-05", end: "2026-09-05"}, value: summary.max.value}

- id: n09
  type: number
  question: Wie hoch war die mittlere Windgeschwindigkeit in 100 m Höhe an der Nordsee am 12. September 2026?
  tools: [weather]
  check: {tool: weather, args: {region: north_sea, start: "2026-09-12", end: "2026-09-12"}, value: summary_by_variable.wind_speed_100m.mean}

- id: n10
  type: number
  question: Wie hoch war die Höchsttemperatur in München am 2. September 2026?
  tools: [weather]
  check: {tool: weather, args: {region: munich, start: "2026-09-02", end: "2026-09-02"}, value: summary_by_variable.temperature_2m.max.value}

# ---------- text questions ----------

- id: t01
  type: text
  question: Wie will die Bundesregierung den Hochlauf von Wasserstoff beschleunigen?
  tools: [bundestag_search]
  gold: ["21/2506", "21/3203", "21/4326"]
  facts: >-
    Draft law 21/2506 (Bundesregierung): many acceleration rules are bundled in one Wasserstoffbeschleunigungsgesetz;
    planning and approval procedures for hydrogen infrastructure become simpler and faster (e.g. digitalisation,
    shorter deadlines); hydrogen projects are given an "überragendes öffentliches Interesse".

- id: t02
  type: text
  question: Was schlägt der Bundesrat vor, um den Ausbau der Windenergie an Land zu beschleunigen?
  tools: [bundestag_search]
  gold: ["21/5920"]
  facts: >-
    Draft law 21/5920 (Bundesrat): a one-off additional special tender (Sonderausschreibung) for onshore wind
    in 2026 with 5,000 MW, not counted against the regular tender volumes, because a record number of projects
    was approved in 2025 and a backlog of approved projects is feared.

- id: t03
  type: text
  question: Was fordert die AfD-Fraktion zum Erneuerbare-Energien-Gesetz?
  tools: [bundestag_search]
  gold: ["21/8108"]
  facts: >-
    Draft law 21/8108 (AfD): the EEG should be abolished (EEG-Abschaffungsgesetz); the EEG would cease to apply
    the day after the law is promulgated. The saved subsidies could pay for lowering the electricity tax.

- id: t04
  type: text
  question: Welche Kritik übt die Linke an der geplanten EEG-Novelle 2027?
  tools: [bundestag_search]
  gold: ["21/7907"]
  facts: >-
    Motion 21/7907 (Die Linke): the EEG-Novelle 2027 cuts support for renewables, which hurts jobs, the pace of
    expansion and the climate targets. Die Linke rejects abolishing the fixed feed-in tariff for small rooftop
    PV and wants support for PV on buildings to be kept.

- id: t05
  type: text
  question: Wie steht die AfD-Fraktion zur Kernenergie?
  tools: [bundestag_search]
  gold: ["21/4461", "21/4873"]
  facts: >-
    Motion 21/4461 (AfD): calls the nuclear phase-out a serious mistake; nuclear energy should be recognised as
    clean and environmentally friendly (EU taxonomy) and promoted; more research on new reactors such as small
    modular reactors (SMR), molten salt and dual-fluid reactors.

- id: t06
  type: text
  question: Was fordert die Linke für die Stromübertragungsnetze?
  tools: [bundestag_search]
  gold: ["21/3911"]
  facts: >-
    Motion 21/3911 (Die Linke): a federal infrastructure company should operate the transmission grids, and the
    transmission grids should be transferred into public ownership. Context: the federal government buys 25.1 %
    of TenneT Deutschland for 7.6 billion euros.

- id: t07
  type: text
  question: Wie stark will die AfD die Energie- und Stromsteuer senken, und wie hoch wäre die Entlastung?
  tools: [bundestag_search]
  gold: ["21/6332", "21/6679"]
  facts: >-
    Draft law 21/6332 (AfD): lower energy and electricity tax rates to the EU minimum rates (Energy Taxation
    Directive 2003/96/EC). Relief of about 21 billion euros per year, of which about 12.3 billion on fuels,
    about 6.2 billion on the electricity tax and 2.5 billion on heating.

- id: t08
  type: text
  question: Welchen Umsatzsteuersatz will die AfD für Gas einführen?
  tools: [bundestag_search]
  gold: ["21/6333", "21/6679"]
  facts: >-
    Draft law 21/6333 (AfD): the supply of gas should get the reduced VAT rate of 7 %. Tax revenue would fall
    by about 9 billion euros per year (full year).

- id: t09
  type: text
  question: Was fordern die Grünen in Bezug auf die Nord-Stream-Pipelines?
  tools: [bundestag_search]
  gold: ["21/224", "21/468"]
  facts: >-
    Motion 21/224 (Bündnis 90/Die Grünen): explicitly rule out putting the Nord Stream pipelines back into
    operation; keep critical infrastructure in European hands; end Russian energy imports by the end of 2027
    at the latest (REPowerEU); present a gas independence strategy.

- id: t10
  type: text
  question: Wie will die Bundesregierung genug gesicherte Kraftwerksleistung für das Jahr 2031 sicherstellen?
  tools: [bundestag_search]
  gold: ["21/6279", "21/6563", "21/6998"]
  facts: >-
    Draft law 21/6279 (Bundesregierung): a capacity market as investment framework. In tenders, bidders compete
    for a payment from the transmission system operators for providing electrical capacity; the cheapest win.
    Limited to the target year 2031; a comprehensive capacity market from 2032 is to follow. The first tenders
    are for new plants.

# ---------- mixed questions ----------

- id: m01
  type: mixed
  question: Wie hoch war der durchschnittliche Strompreis am 15. September 2026, und was fordert die Linke zur Stromsteuer?
  tools: [price_series, bundestag_search]
  check: {tool: price_series, args: {start: "2026-09-15", end: "2026-09-15"}, value: summary.mean}
  gold: ["21/4273"]
  facts: >-
    Motion 21/4273 (Die Linke): lower the electricity tax from 2.05 cent/kWh to the EU minimum of 0.1 cent/kWh
    for private persons and 0.05 cent/kWh for companies.

- id: m02
  type: mixed
  question: Wie viel Solarstrom wurde am 1. September 2026 erzeugt, und was fordern die Grünen zur Einspeisevergütung für Solaranlagen auf Dächern?
  tools: [generation_load, bundestag_search]
  check: {tool: generation_load, args: {series: solar, start: "2026-09-01", end: "2026-09-01"}, value: summary.total}
  gold: ["21/4457"]
  facts: >-
    Motion 21/4457 (Bündnis 90/Die Grünen): keep the feed-in tariff for rooftop PV and avoid unnecessary
    bureaucratic hurdles; keep adequate support for all kinds of rooftop solar; a right to solar, e.g. through
    energy sharing and tenant electricity models.

- id: m03
  type: mixed
  question: Wie viel Windstrom an Land wurde am 10. September 2026 erzeugt, und welche zusätzliche Ausschreibung schlägt der Bundesrat vor?
  tools: [generation_load, bundestag_search]
  check: {tool: generation_load, args: {series: wind_onshore, start: "2026-09-10", end: "2026-09-10"}, value: summary.total}
  gold: ["21/5920"]
  facts: >-
    Draft law 21/5920 (Bundesrat): a one-off additional special tender for onshore wind in 2026 with 5,000 MW,
    not counted against the regular tender volumes.

- id: m04
  type: mixed
  question: Wie viel Strom aus Offshore-Wind wurde vom 21. bis 27. September 2026 erzeugt, und was regelt der Gesetzentwurf zur Windenergie auf See und zu Stromnetzen?
  tools: [generation_load, bundestag_search]
  check: {tool: generation_load, args: {series: wind_offshore, start: "2026-09-21", end: "2026-09-27"}, value: summary.total}
  gold: ["21/1491", "21/2075", "21/3078"]
  facts: >-
    Draft law 21/1491 (Bundesregierung): implements the planning and permitting rules of the EU Renewable Energy
    Directive (EU) 2018/2001 for offshore wind and power grids, with changes to the WindSeeG, the EnWG and the
    NABEG; the offshore area development plan will define acceleration areas with simpler permitting.

- id: m05
  type: mixed
  question: Wie viel Strom wurde vom 21. bis 27. September 2026 aus Braunkohle erzeugt, und wie will die Bundesregierung Investitionen in neue gesicherte Kraftwerksleistung anreizen?
  tools: [generation_load, bundestag_search]
  check: {tool: generation_load, args: {series: lignite, start: "2026-09-21", end: "2026-09-27"}, value: summary.total}
  gold: ["21/6279", "21/6563", "21/6998"]
  facts: >-
    Draft law 21/6279 (Bundesregierung): a capacity market with tenders, in which bidders get a payment from the
    transmission system operators for providing secured capacity, first for the target year 2031.

- id: m06
  type: mixed
  question: Wie hoch war der höchste Strompreis am 20. September 2026, und was fordern die Grünen zum Netzanschlusspaket?
  tools: [price_series, bundestag_search]
  check: {tool: price_series, args: {start: "2026-09-20", end: "2026-09-20"}, value: summary.max.value}
  gold: ["21/8121"]
  facts: >-
    Motion 21/8121 (Bündnis 90/Die Grünen): withdraw the Netzanschlusspaket and present a new grid package for
    a 100 % renewable energy system; reject the Redispatch-Vorbehalt (regional steering of renewables in
    congested areas); modernise and digitalise the grids.

- id: m07
  type: mixed
  question: Wie hoch war die mittlere Windgeschwindigkeit an der Nordsee am 12. September 2026, und welche Maßnahmen gegen Netzengpässe fordert die Linke?
  tools: [weather, bundestag_search]
  check: {tool: weather, args: {region: north_sea, start: "2026-09-12", end: "2026-09-12"}, value: summary_by_variable.wind_speed_100m.mean}
  gold: ["21/7906"]
  facts: >-
    Motion 21/7906 (Die Linke): no Redispatch-Vorbehalt for renewables in capacity-limited grid areas; instead
    e.g. overbuilding grid connections, curtailment at the feed-in point, standardised flexible grid connection
    agreements, temporarily higher grid utilisation and a battery storage push in congested areas.

- id: m08
  type: mixed
  question: Wie hoch war der durchschnittliche stündliche Stromverbrauch am 16. September 2026, und was fordern die Grünen zur 65-Prozent-Regel beim Heizen?
  tools: [generation_load, bundestag_search]
  check: {tool: generation_load, args: {series: load, start: "2026-09-16", end: "2026-09-16"}, value: summary.mean}
  gold: ["21/2724"]
  facts: >-
    Motion 21/2724 (Bündnis 90/Die Grünen): keep the 65 % renewable heating rule of § 71 GEG and secure it
    legally; implement the EU buildings directive (EPBD) by May 2026 with minimum energy performance standards
    so the worst buildings are renovated first; raise the renovation rate to at least 2 % per year.

- id: m09
  type: mixed
  question: Wie hoch war die Höchsttemperatur in München am 2. September 2026, und was fordert die Linke für eine soziale Wärmewende?
  tools: [weather, bundestag_search]
  check: {tool: weather, args: {region: munich, start: "2026-09-02", end: "2026-09-02"}, value: summary_by_variable.temperature_2m.max.value}
  gold: ["21/3910"]
  facts: >-
    Motion 21/3910 (Die Linke): the heating transition should be a mandatory municipal task; raise the 65 % rule
    to 100 % renewable energy for one- and two-family houses and 75 % for multi-family houses; renovations should
    be neutral for the total rent (warmmietenneutral) and help low-income households.

- id: m10
  type: mixed
  question: Was war der niedrigste Strompreis in der Woche vom 7. bis 13. September 2026, und warum wollen die Grünen eine Rückkehr zu Nord Stream ausschließen?
  tools: [price_series, bundestag_search]
  check: {tool: price_series, args: {start: "2026-09-07", end: "2026-09-13"}, value: summary.min.value}
  gold: ["21/224", "21/468"]
  facts: >-
    Motion 21/224 (Bündnis 90/Die Grünen): the Nord Stream pipelines endanger security; Russian fossil exports
    finance Russia's war against Ukraine; Russian energy imports should end by the end of 2027 and dependence on
    autocracies should be reduced.
```

- [ ] **Step 4: Run the test to see it pass**

Run: `.venv/bin/pytest tests/test_evaluation.py -v`
Expected: 10 passed.

- [ ] **Step 5: Write `eval/run_eval.py`**

```python
"""Run the agent on all evaluation questions and log the run to MLflow.

Run from the project folder, for example:
    PYTHONPATH=src .venv/bin/python eval/run_eval.py --rag on --k 6 --chunk-size 300
Look at the runs in the browser:
    .venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse

import mlflow

from energiewende import config, evaluation
from energiewende.agent import graph, tools

parser = argparse.ArgumentParser()
parser.add_argument("--rag", choices=["on", "off"], default="on", help="give the agent the Bundestag search or not")
parser.add_argument("--k", type=int, default=4, help="how many text passages the search returns")
parser.add_argument("--chunk-size", type=int, default=400, help="which search index to use")
parser.add_argument("--limit", type=int, help="only the first N questions, for a quick try")
args = parser.parse_args()

tools.SEARCH_K = args.k
tools.CHUNK_SIZE = args.chunk_size
rag = args.rag == "on"
agent = graph.build_agent(graph.get_llm(), tools.TOOLS if rag else tools.DATA_TOOLS)
judge_llm = evaluation.get_judge()
questions = evaluation.load_questions()[: args.limit]

run_name = f"rag-k{args.k}-chunk{args.chunk_size}" if rag else "rag-off"
if args.limit:
    run_name = "try-" + run_name  # compare.py leaves out these short test runs

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.set_experiment("energiewende-agent")
with mlflow.start_run(run_name=run_name):
    mlflow.log_params({
        "model": config.LLM_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "rag": args.rag,
        "k": args.k,
        "chunk_size": args.chunk_size,
        "questions": len(questions),
    })

    rows = []
    for question in questions:
        try:
            result = graph.ask(question["question"], agent=agent)
        except Exception as error:  # e.g. a timeout: the question counts as wrong
            result = {"answer": f"ERROR: {error}", "sources": [], "tool_calls": [], "latency_ms": 0, "tokens": 0}
        row = evaluation.score_question(question, result, rag, judge_llm)
        rows.append(row)
        print(f"{row['id']}  correct={row['correct']}  tools={row['tools_called']}")

    scores = evaluation.metrics(rows)
    mlflow.log_metrics(scores)
    mlflow.log_table({key: [row[key] for row in rows] for key in rows[0]}, artifact_file="questions.json")

print(run_name, scores)
```

- [ ] **Step 6: Write `eval/compare.py`**

```python
"""Show all evaluation runs as one Markdown table and save it to eval/results.md.

Run:  PYTHONPATH=src .venv/bin/python eval/compare.py
"""

from pathlib import Path

import mlflow

from energiewende import config

COLUMNS = [
    ("Run", "tags.mlflow.runName"),
    ("Tools", "metrics.tool_accuracy"),
    ("Hit@k", "metrics.hit_rate"),
    ("Numbers", "metrics.number_accuracy"),
    ("Text", "metrics.text_accuracy"),
    ("All correct", "metrics.answer_accuracy"),
    ("Latency (ms)", "metrics.latency_ms"),
    ("Tokens", "metrics.tokens"),
]


def cell(value):
    """Scores with 2 decimals, big numbers without decimals, '-' when missing."""
    if isinstance(value, float):
        if value != value:  # NaN: this run has no such metric
            return "-"
        return f"{value:.0f}" if value > 1 else f"{value:.2f}"
    return str(value)


mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
runs = mlflow.search_runs(experiment_names=["energiewende-agent"], order_by=["start_time ASC"])
runs = runs[~runs["tags.mlflow.runName"].str.startswith("try-")]

lines = ["| " + " | ".join(name for name, _ in COLUMNS) + " |", "|" + " --- |" * len(COLUMNS)]
for _, run in runs.iterrows():
    lines.append("| " + " | ".join(cell(run.get(column, float("nan"))) for _, column in COLUMNS) + " |")

table = "\n".join(lines)
print(table)
Path("eval/results.md").write_text("# Evaluation results\n\n30 questions per run. Scores are shares from 0 to 1.\n\n" + table + "\n")
```

- [ ] **Step 7: Quick try with 3 questions**

Run: `PYTHONPATH=src .venv/bin/python eval/run_eval.py --limit 3`
Expected: three lines `n01 correct=True tools=price_series` (and n02, n03), then `try-rag-k4-chunk400 {...}`. `mlflow.db` exists and `git status` does not list it.

- [ ] **Step 8: Commit**

```bash
git add eval/questions.yaml eval/run_eval.py eval/compare.py src/energiewende/evaluation.py tests/test_evaluation.py
git commit -m "feat: 30 evaluation questions, run script and comparison table" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Run the comparison

This calls the real LLM: about 6 runs × 30 questions, plus judge calls (roughly 1 million tokens in total, about 30-40 minutes).

**Files:**
- Create: `eval/results.md` (generated)

- [ ] **Step 1: Build the two extra indexes**

Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_index.py 300
PYTHONPATH=src .venv/bin/python scripts/build_index.py 500
```

Expected: about 8,000 chunks for size 300 and about 5,000 for size 500, each in 1-2 minutes.

- [ ] **Step 2: Run the six settings**

```bash
PYTHONPATH=src .venv/bin/python eval/run_eval.py --rag off
PYTHONPATH=src .venv/bin/python eval/run_eval.py --rag on --k 4 --chunk-size 400
PYTHONPATH=src .venv/bin/python eval/run_eval.py --rag on --k 3 --chunk-size 300
PYTHONPATH=src .venv/bin/python eval/run_eval.py --rag on --k 6 --chunk-size 300
PYTHONPATH=src .venv/bin/python eval/run_eval.py --rag on --k 3 --chunk-size 500
PYTHONPATH=src .venv/bin/python eval/run_eval.py --rag on --k 6 --chunk-size 500
```

Expected: each ends with a line like `rag-k6-chunk300 {'tool_accuracy': ..., ...}`. `rag-off` has no `hit_rate` and a low text score.

- [ ] **Step 3: Make the table and look at the failures**

Run: `PYTHONPATH=src .venv/bin/python eval/compare.py`
Expected: a table with 6 rows, saved to `eval/results.md`.

Open `.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db`, go to `http://127.0.0.1:5000`, open a run, and check the `questions.json` table for wrong answers. For each wrong one, decide: agent mistake, judge mistake, or question problem. Write down what you find; do not change a question just to make the score higher.

- [ ] **Step 4: Run the whole test suite**

Run: `.venv/bin/pytest -q`
Expected: 49 passed.

- [ ] **Step 5: Commit**

```bash
git add eval/results.md
git commit -m "docs: evaluation results for six settings" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
