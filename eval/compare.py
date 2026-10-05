"""Show all evaluation runs as one Markdown table and save it to eval/results.md.

Run:  PYTHONPATH=src .venv/bin/python eval/compare.py
"""

import math
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
        if math.isnan(value):  # this run has no such metric
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
