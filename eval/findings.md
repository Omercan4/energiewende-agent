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
