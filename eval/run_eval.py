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

run_name = f"v{graph.PROMPT_VERSION}-" + (f"rag-k{args.k}-chunk{args.chunk_size}" if rag else "rag-off")
if args.limit:
    run_name = "try-" + run_name  # compare.py leaves out these short test runs

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.set_experiment("energiewende-agent")
with mlflow.start_run(run_name=run_name):
    mlflow.log_params({
        "model": config.LLM_MODEL,
        "prompt_version": graph.PROMPT_VERSION,
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
