"""Score the agent's answers for the evaluation in eval/.

For each question we check four things:
- tools:   did the agent call the expected tools?
- hit:     did the Bundestag search find one of the gold papers?
- numbers: does the answer contain the right number (computed by calling the tool ourselves)?
- text:    does the answer agree with the reference facts? (an LLM judge decides)
"""

import re
from datetime import date
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from energiewende import config
from energiewende.agent import tools

TOOLS_BY_NAME = {t.name: t for t in tools.TOOLS}
QUESTIONS_FILE = Path(__file__).parents[2] / "eval" / "questions.yaml"

JUDGE_PROMPT = """You grade the answers of an assistant for questions about German energy policy.
Today is {today}. The dates in the questions are real and in the past.
Grade only the text part: compare it with the reference facts. Ignore numbers like prices or amounts
of electricity, they are checked separately.
correct = true if the answer contains the main point of the reference facts and does not contradict them.
Extra details are fine. The answer may be in German.
Always reply by calling the grade tool."""

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


def judge(question, facts, answer, judge_llm, today=None):
    """Ask the judge if the answer agrees with the reference facts. Returns (ok, reason)."""
    reply = judge_llm.invoke([
        SystemMessage(JUDGE_PROMPT.format(today=today or date.today())),
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


def load_questions():
    """The evaluation questions from eval/questions.yaml."""
    return yaml.safe_load(QUESTIONS_FILE.read_text())
