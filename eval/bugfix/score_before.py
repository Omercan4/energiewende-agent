"""Score the saved 'before the fix' answers with the fixed tools, and print them next to the 'after' run.

Run: PYTHONPATH=src .venv/bin/python eval/bugfix/score_before.py
"""

import json
from pathlib import Path

from energiewende import evaluation

rows = json.loads(Path(__file__).with_name("before.json").read_text())
questions = {q["id"]: q for q in evaluation.load_questions()}
for row in rows:
    expected = evaluation.expected_number(questions[row["id"]]["check"])
    row["expected_fixed_tools"] = round(expected, 2)
    row["number_correct"] = evaluation.number_found(expected, row["answer"])
    print(row["id"], "expected", round(expected, 2), "correct:", row["number_correct"])
print("before the fix:", sum(r["number_correct"] for r in rows), "of", len(rows), "number answers correct")
