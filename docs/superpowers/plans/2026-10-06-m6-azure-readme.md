# M6 Azure and README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The agent runs on a public Azure URL with its own LLM (Azure OpenAI), `/ask` is protected by a key, and the README explains the project, the results and the data licenses.

**Architecture:** A second small Dockerfile (`Dockerfile.azure`) puts the search index on top of the app image, because Azure cannot mount our local `./data`. Both images are built for `linux/amd64` on the Mac and pushed to an Azure Container Registry. Azure Container Apps runs the image, scales to zero when nobody uses it, and gets the LLM settings and keys as environment variables and secrets. The code does not change for Azure, except a small optional API key check in `/ask`.

**Tech Stack:** Azure Container Apps, Azure Container Registry (Basic), Azure OpenAI (`gpt-5-mini`), Docker buildx, Azure CLI 2.90 with the `containerapp` extension.

**Spec:** `docs/superpowers/specs/2026-10-05-energiewende-agent-design.md` (sections 7 and 8, milestone M6)

This is plan 6 of 6.

## Global Constraints

- Code style: as simple and plain as possible, short comments in simple English.
- No key, internal URL or Azure resource name with a secret in a committed file. Azure names and keys live in `.env` (git-ignored) or are read from Azure at deploy time.
- Locally nothing changes: `.env` keeps the gateway LLM; Azure gets its own settings.
- `APP_API_KEY` empty (local default) means `/ask` is open; when set, `/ask` needs the header `X-API-Key`.
- Costs stay inside the free trial credit: Container Apps with `--min-replicas 0`, ACR Basic, pay-per-token LLM.
- Every commit message ends with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Verified facts these tasks rely on (checked 2026-10-06)

- The subscription is a **Free Trial** (`quotaId FreeTrial_2014-09-01`, spending limit on). It cannot charge the card while the limit is on, and it ends after 30 days unless upgraded. No location policy.
- Providers `Microsoft.App`, `Microsoft.ContainerRegistry`, `Microsoft.CognitiveServices`, `Microsoft.OperationalInsights` are registered.
- Azure OpenAI quota on this subscription: `gpt-5-mini` GlobalStandard 500 in all checked EU regions; `gpt-4.1-mini` only batch (0 for Standard/GlobalStandard); `gpt-4o-mini` only Standard in Sweden Central.
- Created already: resource group `rg-energiewende` (Germany West Central), Azure OpenAI resource (name in `.env` as `AZURE_OPENAI_RESOURCE`), deployment `gpt-5-mini` (version 2025-08-07, GlobalStandard, capacity 50).
- **The unchanged agent works with Azure OpenAI** through `LLM_BASE_URL=https://<resource>.openai.azure.com/openai/v1/`, `LLM_MODEL=gpt-5-mini`: price question correct (156.62 EUR/MWh, same as with the gateway), Bundestag question cites 21/4461 and 21/4873. LangChain handles the reasoning model's temperature itself. It is slower: 10-17 s per question instead of 3-5 s.
- The app container uses about 1.5 GiB memory after a Bundestag search, so Azure gets 1.5 vCPU and 3 GiB.
- Docker buildx can build `linux/amd64` on the Mac (emulated). ACR name `energiewendeacr54a551` is available.
- The DIP terms of use are a PDF (`https://dip.bundestag.de/documents/nutzungsbedingungen_dip.pdf`) that the browser pane cannot render; reading it needs a download, which needs the user's OK (Task 4).

## File Structure

```
src/energiewende/config.py      + APP_API_KEY
src/energiewende/api/main.py    /ask checks X-API-Key when APP_API_KEY is set
.env.example                    + APP_API_KEY, Azure names
Dockerfile.azure                app image + search index
Dockerfile.azure.dockerignore   lets only data/chroma into that build
scripts/deploy_azure.sh         registry, images, Container App (create or update)
README.md                       the project page
tests/test_api.py               + key tests
```

---

### Task 1: Protect /ask with an optional key

**Files:**
- Modify: `src/energiewende/config.py`, `src/energiewende/api/main.py`, `.env.example`
- Test: `tests/test_api.py`

**Interfaces:**
- Produces: `config.APP_API_KEY: str` (default `""`). `POST /ask` returns 401 when `APP_API_KEY` is set and the header `X-API-Key` is missing or wrong.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_api.py`:

```python
from energiewende import config


def test_ask_needs_the_key_when_one_is_set(monkeypatch):
    monkeypatch.setattr(config, "APP_API_KEY", "secret")
    monkeypatch.setattr(graph, "ask", lambda question: {"answer": "x", "sources": [], "tool_calls": [], "latency_ms": 1, "tokens": 1})

    assert client.post("/ask", json={"question": "Hallo"}).status_code == 401
    assert client.post("/ask", json={"question": "Hallo"}, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/ask", json={"question": "Hallo"}, headers={"X-API-Key": "secret"}).status_code == 200


def test_health_needs_no_key(monkeypatch):
    monkeypatch.setattr(config, "APP_API_KEY", "secret")

    assert client.get("/health").status_code == 200
```

Move the line `from energiewende import config` to the import block at the top of the file (ruff E402).

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/pytest tests/test_api.py -v`
Expected: `test_ask_needs_the_key_when_one_is_set` FAILS (`AttributeError: ... no attribute 'APP_API_KEY'`).

- [ ] **Step 3: Add the setting and the check**

Append to `src/energiewende/config.py`:

```python

# Optional password for POST /ask (header X-API-Key). Empty means no password, e.g. locally.
APP_API_KEY = os.environ.get("APP_API_KEY", "")
```

In `src/energiewende/api/main.py`, replace the imports with:

```python
import secrets

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from energiewende import config
from energiewende.agent import graph
```

and replace the `ask` function with:

```python
@app.post("/ask", response_model=Answer)
def ask(body: Question, x_api_key: str | None = Header(default=None)):
    """Ask the agent one question. Needs the header X-API-Key if the server has APP_API_KEY set."""
    if config.APP_API_KEY and not secrets.compare_digest(x_api_key or "", config.APP_API_KEY):
        raise HTTPException(status_code=401, detail="Missing or wrong X-API-Key header")
    return graph.ask(body.question)
```

Append to `.env.example`:

```

# Optional password for POST /ask (header X-API-Key). Leave empty locally.
APP_API_KEY=

# Azure deployment (scripts/deploy_azure.sh)
AZURE_RESOURCE_GROUP=rg-energiewende
AZURE_LOCATION=germanywestcentral
AZURE_ACR=
AZURE_OPENAI_RESOURCE=
AZURE_OPENAI_DEPLOYMENT=gpt-5-mini
```

- [ ] **Step 4: Run all checks**

Run: `.venv/bin/ruff check . && .venv/bin/pytest -q`
Expected: `All checks passed!` and 52 passed.

- [ ] **Step 5: Commit**

```bash
git add src/energiewende/config.py src/energiewende/api/main.py .env.example tests/test_api.py
git commit -m "feat: optional X-API-Key protection for /ask" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: The Azure image (app + search index)

**Files:**
- Create: `Dockerfile.azure`, `Dockerfile.azure.dockerignore`

- [ ] **Step 1: Write the two files**

`Dockerfile.azure`:

```dockerfile
# Image for Azure: the app image plus the search index.
# Azure cannot mount our local ./data folder, so the index goes into the image.
ARG BASE=energiewende-agent
FROM ${BASE}
COPY data/chroma/ /app/data/chroma/
```

`Dockerfile.azure.dockerignore` (Docker uses this file instead of `.dockerignore` for `Dockerfile.azure`):

```
# Send only the search index to this build.
*
!data/chroma
```

- [ ] **Step 2: Try it locally (arm64)**

```bash
docker build -t energiewende-agent .
docker build -f Dockerfile.azure -t energiewende-agent-azure .
docker run --rm energiewende-agent-azure python -c "from energiewende.agent import tools; print(tools.bundestag_search.invoke({'question': 'Kernenergie AfD'})['sources'][:1])"
```

Expected: a source line with `Drucksache 21/4461`, without mounting `./data`.

- [ ] **Step 3: Commit**

```bash
git add Dockerfile.azure Dockerfile.azure.dockerignore
git commit -m "build: Azure image with the search index inside" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Deploy to Azure Container Apps

**Files:**
- Create: `scripts/deploy_azure.sh`
- Modify: `.env` (not committed): add `AZURE_ACR=energiewendeacr54a551` and a new random `APP_API_KEY`

- [ ] **Step 1: Write the script**

`scripts/deploy_azure.sh`:

```bash
#!/usr/bin/env bash
# Deploy the agent to Azure Container Apps. Run it again to deploy a new version.
#
# Needs: az login, Docker, the search index in data/chroma (scripts/build_index.py),
# and in .env: AZURE_RESOURCE_GROUP, AZURE_LOCATION, AZURE_ACR, AZURE_OPENAI_RESOURCE,
# AZURE_OPENAI_DEPLOYMENT, APP_API_KEY.
set -euo pipefail
set -a; source .env; set +a

APP=energiewende-agent
ENVIRONMENT=energiewende-env
IMAGE="$AZURE_ACR.azurecr.io/energiewende-agent"
TAG=$(git rev-parse --short HEAD)

# 1. A container registry for our images (Basic is the cheapest).
if ! az acr show -n "$AZURE_ACR" -o none 2>/dev/null; then
  az acr create -n "$AZURE_ACR" -g "$AZURE_RESOURCE_GROUP" -l "$AZURE_LOCATION" --sku Basic --admin-enabled true -o none
fi
az acr login -n "$AZURE_ACR"

# 2. Build for Azure's CPUs (linux/amd64) and push: the app image, then the app image plus the search index.
docker buildx build --platform linux/amd64 -t "$IMAGE:base-$TAG" --push .
docker buildx build --platform linux/amd64 -f Dockerfile.azure --build-arg BASE="$IMAGE:base-$TAG" -t "$IMAGE:$TAG" --push .

# 3. Read the LLM key and the registry password from Azure (they are not stored in .env).
LLM_URL="https://$AZURE_OPENAI_RESOURCE.openai.azure.com/openai/v1/"
LLM_KEY=$(az cognitiveservices account keys list -n "$AZURE_OPENAI_RESOURCE" -g "$AZURE_RESOURCE_GROUP" --query key1 -o tsv)
ACR_PASSWORD=$(az acr credential show -n "$AZURE_ACR" --query "passwords[0].value" -o tsv)

# 4. The Container Apps environment (once), then create or update the app.
if ! az containerapp env show -n "$ENVIRONMENT" -g "$AZURE_RESOURCE_GROUP" -o none 2>/dev/null; then
  az containerapp env create -n "$ENVIRONMENT" -g "$AZURE_RESOURCE_GROUP" -l "$AZURE_LOCATION" -o none
fi

if az containerapp show -n "$APP" -g "$AZURE_RESOURCE_GROUP" -o none 2>/dev/null; then
  az containerapp update -n "$APP" -g "$AZURE_RESOURCE_GROUP" --image "$IMAGE:$TAG" -o none
else
  # min-replicas 0: no costs while nobody uses it (the first request then takes longer).
  az containerapp create -n "$APP" -g "$AZURE_RESOURCE_GROUP" --environment "$ENVIRONMENT" \
    --image "$IMAGE:$TAG" \
    --registry-server "$AZURE_ACR.azurecr.io" --registry-username "$AZURE_ACR" --registry-password "$ACR_PASSWORD" \
    --target-port 8000 --ingress external \
    --cpu 1.5 --memory 3Gi --min-replicas 0 --max-replicas 1 \
    --secrets llm-api-key="$LLM_KEY" app-api-key="$APP_API_KEY" \
    --env-vars LLM_BASE_URL="$LLM_URL" LLM_MODEL="$AZURE_OPENAI_DEPLOYMENT" \
               LLM_API_KEY=secretref:llm-api-key APP_API_KEY=secretref:app-api-key \
    -o none
fi

echo "Deployed: https://$(az containerapp show -n "$APP" -g "$AZURE_RESOURCE_GROUP" --query properties.configuration.ingress.fqdn -o tsv)"
```

Run: `chmod +x scripts/deploy_azure.sh`

- [ ] **Step 2: Add the two values to `.env`**

```bash
echo "AZURE_ACR=energiewendeacr54a551" >> .env
echo "APP_API_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')" >> .env
```

- [ ] **Step 3: Deploy**

Run: `./scripts/deploy_azure.sh`
Expected: ends with `Deployed: https://energiewende-agent.<...>.germanywestcentral.azurecontainerapps.io`. The first run takes long (amd64 build under emulation, ~3.6 GB push).

- [ ] **Step 4: Check the public URL**

```bash
URL=https://<fqdn from step 3>
curl -s $URL/health                                   # {"status":"ok"} (the first call can take 1-2 minutes: scale from zero)
curl -s -o /dev/null -w "%{http_code}\n" -X POST $URL/ask -H "Content-Type: application/json" -d '{"question": "Strompreis gestern?"}'   # 401
curl -s -X POST $URL/ask -H "Content-Type: application/json" -H "X-API-Key: $APP_API_KEY" -d '{"question": "Was fordert die AfD zur Kernenergie?"}'
```

Expected: health ok, 401 without key, an answer citing 21/4461 with the key.

- [ ] **Step 5: Commit**

```bash
git add scripts/deploy_azure.sh
git commit -m "build: deploy script for Azure Container Apps" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Read the DIP terms of use**

Ask the user for permission to download `nutzungsbedingungen_dip.pdf` (3 pages) from `dip.bundestag.de`, read it, and quote the reuse rule in the README in at most one short sentence with the link. Do not guess the terms.

- [ ] **Step 2: Write `README.md`**

Sections, in this order, short and in plain English:

1. Title and one-paragraph summary (what a user can ask; data tools + RAG + evaluation + Docker/CI + Azure).
2. Example questions and answers (the three from M3, real outputs).
3. How it works: the architecture sketch from the spec (client → FastAPI → LangGraph agent → tools / retriever), and the design rule "Python computes, the LLM explains".
4. Evaluation: the table from `eval/results.md` (round 1 and round 2), and the main findings from `eval/findings.md` in 4-5 bullets, including the honest limits (retrieval mix-ups, judge noise).
5. Run it locally: build the index, `.env` from `.env.example`, `docker compose up -d`, URLs (8000/docs, 5001 for MLflow, Model training mode), notebooks.
6. Deploy to Azure: what the script does, costs (scale to zero), the X-API-Key header.
7. Data sources and licenses: SMARD (Bundesnetzagentur | SMARD.de, CC BY 4.0), Open-Meteo (CC BY 4.0, data from DWD; free API for non-commercial use), Bundestag DIP (quote from Step 1).
8. Project layout (directory table from the spec).

- [ ] **Step 3: Check every number in the README**

Every number must match `eval/results.md`, `eval/findings.md` or a real run. Run `.venv/bin/ruff check . && .venv/bin/pytest -q` (52 passed).

- [ ] **Step 4: Commit and push**

```bash
git add README.md
git commit -m "docs: README" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

- [ ] **Step 5: Ask about going public**

The repo is private. Ask the user whether to make it public now (`gh repo edit --visibility public --accept-visibility-change-consequences`). Before that, scan the full history once more for internal names and secrets.
