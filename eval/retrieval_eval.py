"""Measure the search alone (no LLM): is a gold paper among the first k passages?

For every text and mixed question in eval/questions.yaml we know which Bundestag papers contain the answer ("gold").
We run the search with and without the reranker and report:
  hit@k  share of questions where one of the first k passages comes from a gold paper
  MRR    mean of 1 / (rank of the first passage from a gold paper), 0 if there is none among the passages

This is fast, costs no LLM calls, and gives the same numbers every time, so small differences are meaningful here
(unlike in the end-to-end evaluation, where one question is worth 3 points).

Run: RERANKER_MODEL=BAAI/bge-reranker-v2-m3 PYTHONPATH=src .venv/bin/python eval/retrieval_eval.py
"""

import json
import os
from pathlib import Path

from energiewende import config, evaluation
from energiewende.index import search, store

CHUNK_SIZE = 300
KS = (1, 3, 6)
DEPTH = 6  # passages looked at per question
OUT = Path(__file__).with_name("retrieval_results.json")


def first_gold_rank(hits, gold):
    for rank, hit in enumerate(hits, start=1):
        if hit["number"] in gold:
            return rank
    return None


def evaluate(index, questions, reranker=None):
    ranks = []
    for question in questions:
        hits = search.search(index, question["question"], k=DEPTH, reranker=reranker)
        ranks.append(first_gold_rank(hits, set(question["gold"])))
    result = {f"hit@{k}": round(sum(1 for r in ranks if r and r <= k) / len(ranks), 3) for k in KS}
    result["MRR"] = round(sum(1 / r for r in ranks if r) / len(ranks), 3)
    result["questions"] = len(ranks)
    return result


def main():
    questions = [q for q in evaluation.load_questions() if q.get("gold")]
    index = store.load_index(CHUNK_SIZE)
    results = {"baseline (vector search only)": evaluate(index, questions)}
    model_name = config.RERANKER_MODEL or os.environ.get("RERANKER_MODEL")
    if model_name:
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder(model_name, max_length=512)
        results["+ reranker (30 candidates re-sorted)"] = evaluate(index, questions, reranker)
    OUT.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
