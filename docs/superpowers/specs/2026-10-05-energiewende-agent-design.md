# Energiewende Agent: Design

Date: 2026-10-05
Status: approved

## 1. Purpose

A small, public portfolio project that shows one thing clearly: an LLM agent that
chooses between **live data tools** and **document retrieval (RAG)**, is
**evaluated** with tracked experiments, and ships as a **containerized service
with CI**. It is built on open German data only. It is not a product.

Questions the agent answers:

- Number questions, answered by API tools: "What was the day-ahead electricity
  price last Tuesday?", "How much wind power was generated yesterday?"
- Text questions, answered by RAG: "What did the parties argue about the EEG in
  the Bundestag?"
- Mixed questions that need both.

## 2. Non-goals

- No price forecasting model, no Marktstammdatenregister (possible v2).
- No custom frontend. FastAPI's `/docs` page is the demo surface.
- No GraphRAG and no graph database (dropped on purpose: cost and complexity).
- No user accounts, no stored user data, no production hardening beyond what is
  listed in section 8.

## 3. Architecture

```
client -> FastAPI (/ask) -> LangGraph agent (OpenAI-compatible LLM, tool calling)
            tools: price_series, generation_load, weather   -> SMARD / Open-Meteo (cached)
                   bundestag_search                          -> LlamaIndex retriever (vector search)
offline:    ingest scripts build the index -> ./data/chroma (persisted)
tracking:   MLflow logs evaluation runs
```

Directory layout (each directory has one responsibility):

| Path | Responsibility |
|---|---|
| `src/energiewende/ingest/` | Typed HTTP clients for SMARD, Open-Meteo, Bundestag DIP, with on-disk caching |
| `src/energiewende/index/` | Chunking, embedding, building and loading the Chroma index |
| `src/energiewende/agent/` | LangGraph graph, tool definitions, prompts, guardrails |
| `src/energiewende/api/` | FastAPI app and request/response schemas |
| `src/energiewende/config.py` | All settings from environment variables |
| `eval/` | Question set (YAML), evaluation runner, MLflow logging |
| `tests/` | Unit and integration tests |
| `docker/`, `.github/workflows/` | Container and CI definitions |

## 4. Data and retrieval

**Structured data (agent tools).**

- SMARD (Bundesnetzagentur): wholesale electricity price, generation by source,
  grid load. License CC BY 4.0, attribution "Bundesnetzagentur | SMARD.de". No API key.
- Open-Meteo (DWD data): wind and solar weather, forecast and history for a few
  fixed regions. License CC BY 4.0, free for non-commercial use, no API key.
- Each tool validates its inputs (date range, region) with pydantic, caches
  responses on disk with a TTL, and returns a small structured result including
  the source name and the timestamp range.

**Text corpus (RAG).**

- Source: Bundestag DIP API, endpoint `drucksache-text`. Printed papers
  (Drucksachen) of the current electoral period (21) whose title contains an
  energy keyword (Energie, Strom, Erneuerbare, Wasserstoff, Waerme), deduplicated
  and capped at 200 documents. Verified on 2026-10-05: about 190 papers match,
  the API returns full text, and embedding them costs a few cents.
- Plenary minutes are left out on purpose: they cannot be filtered by topic and
  one session is about 440,000 characters, mostly on unrelated subjects.
- Chunking: about 800 tokens with 100 overlap as the starting point; chunk size
  is an evaluated parameter.
- Each chunk keeps metadata: document id, document type, date, and where
  available speaker and parliamentary group. Metadata is used for citations and
  optional filters. No graph is built.
- Embeddings: an OpenAI-compatible embeddings endpoint. Vector store: Chroma,
  persisted under `./data/chroma`. Retrieval: top-k similarity search through
  LlamaIndex.

## 5. Agent

LangGraph `StateGraph` with three steps: decide (the LLM picks zero or more
tools via tool calling), run tools, answer. The loop is bounded.

Guardrails:

- Every number in an answer must come from a tool result. No guessing.
- Every answer lists its sources: data source plus timestamp range, or document id.
- Maximum tool steps per question, per-call timeouts, clear error messages when
  an upstream API is down.
- Out-of-scope questions are declined.
- No personal data beyond public parliamentary records. Logs contain no user
  identifiers.

API:

- `POST /ask` with `{"question": "..."}` returns
  `{"answer", "sources", "tool_calls", "latency_ms", "tokens"}`.
- `GET /health` for the container health check.

## 6. Evaluation and MLflow

- About 30 hand-written questions in `eval/questions.yaml`: roughly 10 number,
  10 text, 10 mixed. Each has the expected tool(s) and reference facts.
- Metrics per run: tool-selection accuracy, retrieval hit@k (is the gold
  document among the top k), answer correctness, latency, token count and cost.
- Answer correctness: numbers are checked programmatically against the tool
  output; text answers use an LLM judge. The judge model is configured
  separately from the agent model and documented in the README.
- Each run logs its parameters to MLflow (model, chunk size, k, RAG on/off).
- Planned comparison (small, to keep cost low): RAG off vs on, k in {3, 6},
  chunk size in {500, 1000}. The README shows the resulting table. Evaluation
  runs are manual and are not part of CI.

## 7. Packaging, CI and deployment

- `Dockerfile` for the app and `docker-compose.yml` running the app plus an
  MLflow server. Keys come from `.env` (never committed; `.env.example` is
  committed).
- GitHub Actions on every push: ruff, pytest, docker build. HTTP and LLM calls
  are mocked in CI.
- Final milestone: deploy to Azure Container Apps, with Azure OpenAI used
  through the same OpenAI-compatible client, so it is a configuration change.
  The Azure account is created by the repo owner. If this milestone is blocked,
  the project is still complete without it.

## 8. Configuration, security, licensing

- All endpoints, keys and model names are environment variables read in
  `config.py`: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `EMBEDDING_MODEL`,
  `DIP_API_KEY`. The code does not care which provider sits behind the URL.
- Which LLM key is used (a personal key or an employer-provided one) is a
  decision of the repo owner and not part of the design. An employer's key
  should only be used with explicit permission, and never appear in the repo.
- The README states each data source, its license, and the required attribution.
  Reuse terms for Bundestag open data are confirmed and quoted in the README
  before publishing.
- Model names are chosen at implementation time (small, cheap, tool-calling
  capable). Library versions are pinned after checking PyPI.

## 9. Testing

- Unit tests for each ingest client using recorded sample responses, tool input
  validation, chunking and metadata handling.
- Integration test of the agent with a stubbed LLM and stubbed HTTP.
- No test hits a real LLM or a real upstream API.

## 10. Milestones and acceptance

| Milestone | Done when |
|---|---|
| M1 Data clients | SMARD, Open-Meteo and DIP clients return typed results; unit tests pass |
| M2 Index | The index builds from the capped corpus; a sample query returns relevant chunks with metadata |
| M3 Agent and API | `/ask` answers one number, one text and one mixed example with sources |
| M4 Evaluation | `eval/run_eval.py` runs the question set and logs a run to MLflow; comparison table produced |
| M5 Docker and CI | `docker compose up` starts the service and MLflow; CI is green on GitHub |
| M6 Azure and README | Deployed on Azure (if the account exists); README has architecture, eval table, licenses |

Estimated effort: about three weekends. Model cost: a few euros for embeddings
and evaluation runs.

## 11. Risks and open items

- **Framework churn.** LangGraph and LlamaIndex change APIs often. Mitigation:
  pin versions, read current docs at implementation time, keep framework code
  in thin adapters.
- **Bundestag API key.** Resolved: the API returns full text, and a public key
  (valid until end of May 2027) is listed at
  https://dip.bundestag.de/über-dip/hilfe/api. The key lives only in `.env`.
- **Local prerequisites.** Docker is not installed on the development machine
  and must be installed before M5. The Azure CLI is needed only for M6.
- **Open-Meteo terms.** The free tier is for non-commercial use. This is a
  portfolio project, so it fits; the README says so.
