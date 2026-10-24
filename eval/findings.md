# What the evaluation showed (first round, 2026-10-05)

Six runs with 30 questions each: agent `gemini-2.5-flash`, judge `claude-haiku-4.5`.
The table is in `results.md`. Below is what the wrong answers had in common.

## What works

- **Numbers: 100 % correct in every run.** The tools compute mean, min, max and total in Python, and the
  model only copies them. (Before this design, the model calculated an average itself and was off by 7 %.)
- **Without the Bundestag search, no text question is answered** (text score 0.00). The search is what
  makes the text answers possible.
- **More context helps:** 6 passages instead of 3 raise the text score (chunk 300: 0.65 → 0.75,
  chunk 500: 0.70 → 0.80) and the search hit rate. The cost is more tokens and time
  (k6/chunk500 is the most accurate, 0.87, but also the slowest: 6.9 s and 4,700 tokens per question).

## Why answers were wrong

1. **The agent stops halfway on two-part questions (m04, m10).** It answers the number part and then
   writes "für den zweiten Teil benötige ich eine weitere Abfrage" without calling the search.
   Fix to try: tell the agent in the system prompt to call all tools a question needs before answering.
2. **The search finds a paper on the right topic but from the wrong party or bill (t03, m07, m08).**
   Example t03: "Was fordert die AfD zum EEG?" finds the AfD paper on nuclear energy instead of the
   EEG abolition bill. The small embedding model matches the topic words well but not who says what.
   Ideas: a reranker, or filters on party and paper type.
3. **The right paper is found, but not the passage with the detail (t07, m01).** For example the relief
   of 21 billion euros is in the paper, but not in the retrieved passages, so the agent says it is "not
   mentioned". More or larger passages help here.
4. **The judge thinks September 2026 is in the future** and sometimes mentions this in its reason.
   The judge should only grade the text part (numbers are checked in code) and should be told the date.

## Corrections to the question set

- m08: 21/4391 (the committee recommendation on the Greens' motion 21/2724) was added to the gold papers
  after this round. The hit rates in `results.md` were computed before; there, finding 21/4391 counts as a miss.

# Round 2 (2026-10-05): two fixes, two settings re-run

Changes (prompt version 2, runs named `v2-...`):
- Agent prompt: "If the question has several parts, call the tools for every part before you answer."
- Judge prompt: knows today's date and grades only the text part (numbers are checked in code).
- m08 has the extra gold paper 21/4391 (see above).

| Setting | Round | Tools | Hit@k | Text | All correct |
| --- | --- | --- | --- | --- | --- |
| k4, chunk 400 | 1 | 0.97 | 0.75 | 0.70 | 0.80 |
| k4, chunk 400 | 2 | **1.00** | 0.80 | 0.70 | 0.80 |
| k6, chunk 300 | 1 | 0.97 | 0.90 | 0.75 | 0.83 |
| k6, chunk 300 | 2 | **1.00** | **1.00** | 0.75 | 0.83 |

What changed:
- **The agent fix worked for what it targeted.** Tool accuracy is 1.00 in both runs, every mixed question
  now uses the search (10/10, before 9/10), and no answer stops halfway any more.
- **The overall score did not move.** The questions that are still wrong (t03, t04, m01, m07, m08 and a few
  others) fail because the search returns the wrong passage or the wrong party's paper (problems 2 and 3
  above). The prompt fix cannot help there; better retrieval can.
- **The judge itself is noisy.** Re-grading the round-1 answers with the new judge gave text scores of
  0.75 (k4/chunk400, before 0.70) and 0.70 (k6/chunk300, before 0.75). One text question is 0.05 of the
  score, so differences of 0.05 between runs are within this noise. Bigger question sets or several
  judge votes per answer would make the numbers more stable.

Next step for a round 3: improve retrieval (a reranker, or filters on party and paper type), then
re-run the same settings and compare.


# Round 3 (2026-10-24): number names, search filters, reranker, forecast tool

**Setup changed, so the numbers are not comparable with rounds 1 and 2.** The gateway removed `gemini-2.5-flash`
and `claude-haiku-4.5` answered with server errors. The agent is now `claude-sonnet-5.5`, the judge `gemini-3.8-flash`.

## What was changed

- `summarize()` now returns `mean_per_hour`, `mean_total_per_complete_day` and a `complete` flag per day (and says so in a `note`).
  Before, one field `mean` sat next to the totals, and a partial day looked like a full day.
- `bundestag_search` can filter by party (`afd`, `gruene`, `linke`, `bundesregierung`, `ausschuss`) and paper type, and can rerank.
- New tool `price_forecast` (reads the forecast of the energy-price-forecast project). Prompt version 4.
- Five new questions: n11 to n13 (per hour / per day / total) and n14, n15 (forecast).

## What the evaluation showed

- **A bug in my own scoring.** Negative prices written as "‑0,24" with a non-breaking hyphen (U+2011) were read as +0.24. Two correct
  answers (n03, m10) were scored wrong. Fixed in `numbers_in` with a test, and all runs were repeated.
- **The per-hour / per-day questions** (`eval/bugfix/before.json`): before the fix the agent could not answer the per-day average
  (n11, it said the tool does not give it); after the fix all three are correct. The stronger model did not report the hourly mean as a daily value,
  but it could not answer the daily question at all. (The earlier model did report the hourly mean as the daily one.)
- **Filters:** same accuracy (0.97), 26 % fewer tokens (10,573 to 7,777). The latency of that run (10.5 s) was measured while a large
  model was downloading in parallel, so it says nothing.
- **Saturation:** with the stronger model, 32 of 33 answers are fully correct. A difference of one question is 3 points and within the
  noise of the judge. In the final run (35 questions) m08 and m09 fail: the right paper is retrieved, but the passage with the detail is not.
- **Search alone** (`eval/retrieval_eval.py`, no LLM): hit@1 0.45 to 0.55, hit@3 0.70 to 0.70, hit@6 0.85 to 0.80, MRR 0.570 to 0.635
  with the multilingual `mmarco-mMiniLMv2` reranker. A gain at the top, one question lost at rank 6, not significant with 20 questions.
  Kept optional. The larger `bge-reranker-v2-m3` was not tested (the 2 GB download kept stalling).

## Next

A harder question set (the current one no longer separates configurations), evaluating the query the agent sends to the search
(it rewrites the question), and a reranker test with more questions.
