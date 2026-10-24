"""One-off: ask the three 'per hour / per day / total' questions with the code BEFORE the summarize() fix.

Saves the answers to eval/bugfix/before.json so they can be scored later against the fixed tools.
Run: PYTHONPATH=src .venv/bin/python eval/bugfix/ask_before.py
"""

import json
from pathlib import Path

from energiewende import evaluation
from energiewende.agent import graph, tools

QUESTIONS = [
    ("n11", "Wie viel Windstrom an Land wurde in der Woche vom 7. bis 13. September 2026 im Durchschnitt pro Tag erzeugt?"),
    ("n12", "Wie hoch war der durchschnittliche Stromverbrauch pro Stunde am 15. September 2026?"),
    ("n13", "Wie viel Strom wurde am 15. September 2026 insgesamt in Deutschland verbraucht?"),
]

# expected values, computed from the OLD tool output (every day in these ranges is complete)
wind = tools.generation_load.invoke({"series": "wind_onshore", "start": "2026-09-07", "end": "2026-09-13"})["summary"]
load = tools.generation_load.invoke({"series": "load", "start": "2026-09-15", "end": "2026-09-15"})["summary"]
expected = {
    "n11": sum(day["total"] for day in wind["daily"]) / len(wind["daily"]),
    "n12": load["mean"],
    "n13": load["total"],
}

rows = []
for qid, text in QUESTIONS:
    result = graph.ask(text)
    ok = evaluation.number_found(expected[qid], result["answer"])
    rows.append({"id": qid, "question": text, "expected": round(expected[qid], 2), "answer": result["answer"], "number_correct": ok})
    print(qid, "expected", round(expected[qid], 2), "correct:", ok)
Path(__file__).with_name("before.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
