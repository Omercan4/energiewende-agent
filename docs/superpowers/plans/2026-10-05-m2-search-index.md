# M2 Search Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the 99 energy papers from M1 into a search index on disk, so a question returns the most relevant text passages together with their source.

**Architecture:** `store.py` splits each paper into chunks, turns each chunk into a vector with a local embedding model, and saves everything in a Chroma database under `data/chroma`. `search.py` has one function that asks the index for the chunks closest to a question and returns plain dicts. Tests use a fake embedding model, so they need no model download and no internet.

**Tech Stack:** LlamaIndex (core, HuggingFace embeddings, Chroma vector store), Chroma, sentence-transformers / PyTorch, model `intfloat/multilingual-e5-small`.

**Spec:** `docs/superpowers/specs/2026-10-05-energiewende-agent-design.md` (section 4, milestone M2)

This is plan 2 of 6. It builds on M1 (`dip.get_energy_papers()`).

## Global Constraints

- Code style: as simple and plain as possible. Plain functions, plain dicts, short comments in simple English.
- Pinned versions (verified installed and working on 2026-10-05): `llama-index-core==0.14.25`, `llama-index-embeddings-huggingface==0.8.0`, `llama-index-vector-stores-chroma==0.6.0`, `chromadb==1.5.9`, `sentence-transformers==6.1.0`, `torch==2.14.1`.
- Embedding model: `intfloat/multilingual-e5-small`, with the prefixes `query: ` for questions and `passage: ` for chunks (the e5 models need them).
- Default chunk size 400, overlap 50. One Chroma collection per chunk size, named `papers_<chunk_size>`.
- Metadata is stored with each chunk but excluded from the embedded text and from the chunk-size budget.
- Tests never download a model and never touch the internet (use `MockEmbedding`).
- `data/` stays git-ignored. No keys or internal URLs in committed files.
- Every commit message ends with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Verified facts these tasks rely on (checked 2026-10-05)

- With chunk size 400 the real corpus gives about 6,200 chunks; the longest is about 350 e5 tokens, below the model's 512 limit. Building takes about 85 seconds on the Mac GPU (`mps`).
- If metadata is not excluded, LlamaIndex counts it in the chunk size. With real titles (~900 characters) this produced 59,000 tiny chunks, and with a very long title the build fails with `ValueError: Metadata length (729) is longer than chunk size (400)`.
- `client.list_collections()` returns collection objects with a `.name`; `collection.count()` returns the number of saved chunks.

## File Structure

```
requirements.txt                       + the six packages above
.env.example                           + INDEX_DIR, EMBEDDING_MODEL
src/energiewende/config.py             + INDEX_DIR, EMBEDDING_MODEL
src/energiewende/index/__init__.py     empty
src/energiewende/index/store.py        papers -> documents -> chunks -> vectors in Chroma; load; count
src/energiewende/index/search.py       search(index, question, k) -> list of dicts
scripts/build_index.py                 downloads papers, builds the real index
scripts/search_index.py                asks the real index a question from the command line
tests/test_index.py
```

---

### Task 1: Dependencies, settings, and papers to documents

**Files:**
- Modify: `requirements.txt`, `.env.example`, `src/energiewende/config.py`
- Create: `src/energiewende/index/__init__.py`, `src/energiewende/index/store.py`
- Test: `tests/test_index.py`

**Interfaces:**
- Consumes: paper dicts from M1 `dip.to_paper`: `{"id", "number", "date", "type", "title", "authors": list[str], "url", "text"}`.
- Produces:
  - `config.INDEX_DIR: str` (default `"data/chroma"`), `config.EMBEDDING_MODEL: str` (default `"intfloat/multilingual-e5-small"`)
  - `store.papers_to_documents(papers: list[dict]) -> list[llama_index.core.Document]`. Metadata keys: `number, date, type, title, authors` (one string, comma separated), `url`.

- [ ] **Step 1: Add the packages and settings**

Append to `requirements.txt`:

```
llama-index-core==0.14.25
llama-index-embeddings-huggingface==0.8.0
llama-index-vector-stores-chroma==0.6.0
chromadb==1.5.9
sentence-transformers==6.1.0
torch==2.14.1
```

Append to `.env.example`:

```

# Folder for the search index (Chroma database)
INDEX_DIR=data/chroma

# Local HuggingFace embedding model
EMBEDDING_MODEL=intfloat/multilingual-e5-small
```

Append to `src/energiewende/config.py`:

```python

# Folder where the search index (Chroma database) is stored.
INDEX_DIR = os.environ.get("INDEX_DIR", "data/chroma")

# Local HuggingFace model that turns text into vectors.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
```

Create the empty file `src/energiewende/index/__init__.py`.

Run: `.venv/bin/pip install -r requirements.txt`
Expected: finishes without errors (the packages may already be installed).

- [ ] **Step 2: Write the failing test**

`tests/test_index.py`:

```python
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import MetadataMode

from energiewende.index import store


def fake_embed_model():
    """Gives every text the same small vector. Fast and needs no download.
    Good for testing that saving and loading work, not for testing relevance."""
    return MockEmbedding(embed_dim=8)


def make_paper(n, title="Kurzer Titel", text="Der Strompreis ist gestiegen. " * 10):
    """One paper, shaped like the dicts from dip.py."""
    return {
        "id": str(n),
        "number": f"21/{n}",
        "date": "2026-09-30",
        "type": "Antrag",
        "title": title,
        "authors": ["Bundesregierung", "Bundesrat"],
        "url": f"https://example.com/{n}.pdf",
        "text": text,
    }


def test_documents_keep_metadata_but_do_not_embed_it():
    doc = store.papers_to_documents([make_paper(1)])[0]

    assert doc.metadata["number"] == "21/1"
    assert doc.metadata["authors"] == "Bundesregierung, Bundesrat"
    # Only the paper text is embedded, not the title or other metadata.
    assert doc.get_content(metadata_mode=MetadataMode.EMBED) == doc.text
```

- [ ] **Step 3: Run the test to see it fail**

Run: `.venv/bin/pytest tests/test_index.py -v`
Expected: FAIL with `ImportError: cannot import name 'store'`.

- [ ] **Step 4: Write `papers_to_documents`**

`src/energiewende/index/store.py`:

```python
"""Build and load the search index for the Bundestag papers.

Each paper is split into chunks of about 400 tokens. A local embedding model
turns every chunk into a vector (a list of 384 numbers). Texts with similar
meaning get similar vectors. The vectors are saved in a Chroma database.
"""

from llama_index.core import Document


def papers_to_documents(papers):
    """Turn paper dicts (from dip.py) into LlamaIndex documents."""
    documents = []
    for paper in papers:
        doc = Document(
            text=paper["text"],
            metadata={
                "number": paper["number"],
                "date": paper["date"],
                "type": paper["type"],
                "title": paper["title"],
                "authors": ", ".join(paper["authors"]),  # Chroma cannot store lists
                "url": paper["url"],
            },
        )
        # Keep the metadata for citations, but do not embed it and do not
        # count it in the chunk size. Titles are often 900+ characters long.
        doc.excluded_embed_metadata_keys = list(doc.metadata)
        doc.excluded_llm_metadata_keys = list(doc.metadata)
        documents.append(doc)
    return documents
```

- [ ] **Step 5: Run the test to see it pass**

Run: `.venv/bin/pytest tests/test_index.py -v`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .env.example src/energiewende/config.py src/energiewende/index tests/test_index.py
git commit -m "feat: index settings and papers-to-documents conversion" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Build, load and count the index

**Files:**
- Modify: `src/energiewende/index/store.py`
- Test: `tests/test_index.py`

**Interfaces:**
- Consumes: `store.papers_to_documents` (Task 1), `config.INDEX_DIR`, `config.EMBEDDING_MODEL`.
- Produces:
  - `store.get_embed_model() -> HuggingFaceEmbedding`
  - `store.build_index(papers: list[dict], chunk_size: int = 400, chunk_overlap: int = 50, index_dir: str | None = None, embed_model=None) -> VectorStoreIndex`
  - `store.load_index(chunk_size: int = 400, index_dir: str | None = None, embed_model=None) -> VectorStoreIndex`
  - `store.count_chunks(chunk_size: int = 400, index_dir: str | None = None) -> int`
  - `index_dir=None` means `config.INDEX_DIR`; `embed_model=None` means `get_embed_model()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_index.py`:

```python
def test_long_titles_do_not_shrink_the_chunks(tmp_path):
    # Real Bundestag titles are often 900+ characters long.
    paper = make_paper(1, title="Sehr langer Titel " * 120, text="Der Strompreis ist gestiegen. " * 300)

    store.build_index([paper], index_dir=str(tmp_path), embed_model=fake_embed_model())

    # About 9 chunks. If the title were counted, the build would fail
    # or produce many tiny chunks.
    assert store.count_chunks(index_dir=str(tmp_path)) < 20


def test_saved_index_can_be_loaded_again(tmp_path):
    store.build_index([make_paper(1), make_paper(2)], index_dir=str(tmp_path), embed_model=fake_embed_model())

    index = store.load_index(index_dir=str(tmp_path), embed_model=fake_embed_model())

    hits = index.as_retriever(similarity_top_k=2).retrieve("Strompreis")
    assert len(hits) == 2


def test_building_again_replaces_the_old_index(tmp_path):
    papers = [make_paper(1), make_paper(2)]
    store.build_index(papers, index_dir=str(tmp_path), embed_model=fake_embed_model())
    first = store.count_chunks(index_dir=str(tmp_path))

    store.build_index(papers, index_dir=str(tmp_path), embed_model=fake_embed_model())

    assert store.count_chunks(index_dir=str(tmp_path)) == first  # replaced, not doubled


def test_each_chunk_size_has_its_own_index(tmp_path):
    papers = [make_paper(1, text="Der Strompreis ist gestiegen. " * 300)]
    store.build_index(papers, chunk_size=300, index_dir=str(tmp_path), embed_model=fake_embed_model())
    store.build_index(papers, chunk_size=500, index_dir=str(tmp_path), embed_model=fake_embed_model())

    small = store.count_chunks(chunk_size=300, index_dir=str(tmp_path))
    large = store.count_chunks(chunk_size=500, index_dir=str(tmp_path))
    assert small > large  # smaller chunks, so more of them
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/pytest tests/test_index.py -v`
Expected: 1 passed, 4 failed with `AttributeError: module 'energiewende.index.store' has no attribute 'build_index'`.

- [ ] **Step 3: Write build, load and count**

Replace the import block at the top of `src/energiewende/index/store.py` (the line `from llama_index.core import Document`) with:

```python
import chromadb
from chromadb.config import Settings
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from energiewende import config
```

Append to `src/energiewende/index/store.py`:

```python


def get_embed_model():
    """The local embedding model. The e5 models expect these two prefixes."""
    return HuggingFaceEmbedding(
        model_name=config.EMBEDDING_MODEL,
        query_instruction="query: ",
        text_instruction="passage: ",
    )


def get_client(index_dir):
    """The Chroma database in index_dir (created if it does not exist)."""
    return chromadb.PersistentClient(path=index_dir, settings=Settings(anonymized_telemetry=False))


def collection_name(chunk_size):
    """One collection per chunk size, so different sizes can be compared later."""
    return f"papers_{chunk_size}"


def build_index(papers, chunk_size=400, chunk_overlap=50, index_dir=None, embed_model=None):
    """Split, embed and save all papers. Replaces an old index with the same chunk size."""
    client = get_client(index_dir or config.INDEX_DIR)
    name = collection_name(chunk_size)
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)

    vector_store = ChromaVectorStore(chroma_collection=client.create_collection(name))
    return VectorStoreIndex.from_documents(
        papers_to_documents(papers),
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        embed_model=embed_model or get_embed_model(),
        transformations=[SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)],
    )


def load_index(chunk_size=400, index_dir=None, embed_model=None):
    """Open an index that build_index saved earlier."""
    client = get_client(index_dir or config.INDEX_DIR)
    collection = client.get_collection(collection_name(chunk_size))
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model or get_embed_model())


def count_chunks(chunk_size=400, index_dir=None):
    """How many chunks are saved for this chunk size."""
    client = get_client(index_dir or config.INDEX_DIR)
    return client.get_collection(collection_name(chunk_size)).count()
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/pytest tests/test_index.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/energiewende/index/store.py tests/test_index.py
git commit -m "feat: build, load and count the Chroma search index" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Search function

**Files:**
- Create: `src/energiewende/index/search.py`
- Test: `tests/test_index.py`

**Interfaces:**
- Consumes: an index from `store.build_index` / `store.load_index` (Task 2).
- Produces: `search.search(index, question: str, k: int = 4) -> list[dict]`, each dict `{"number": str, "date": str, "title": str, "url": str, "score": float, "text": str}`, best match first. M3 wraps this as the agent's `bundestag_search` tool.

- [ ] **Step 1: Write the failing test**

Change the import line at the top of `tests/test_index.py` from `from energiewende.index import store` to:

```python
from energiewende.index import search, store
```

Append to `tests/test_index.py`:

```python
def test_search_returns_text_and_source(tmp_path):
    store.build_index([make_paper(7)], index_dir=str(tmp_path), embed_model=fake_embed_model())
    index = store.load_index(index_dir=str(tmp_path), embed_model=fake_embed_model())

    hit = search.search(index, "Strompreis", k=1)[0]

    assert hit["number"] == "21/7"
    assert hit["date"] == "2026-09-30"
    assert hit["title"] == "Kurzer Titel"
    assert hit["url"] == "https://example.com/7.pdf"
    assert "Strompreis" in hit["text"]
    assert isinstance(hit["score"], float)
```

- [ ] **Step 2: Run the test to see it fail**

Run: `.venv/bin/pytest tests/test_index.py -v`
Expected: FAIL with `ImportError: cannot import name 'search'`.

- [ ] **Step 3: Write the search function**

`src/energiewende/index/search.py`:

```python
"""Find the text passages that best match a question."""


def search(index, question, k=4):
    """The k chunks most similar to the question, best first, with their source."""
    results = []
    for hit in index.as_retriever(similarity_top_k=k).retrieve(question):
        meta = hit.metadata
        results.append(
            {
                "number": meta["number"],
                "date": meta["date"],
                "title": meta["title"],
                "url": meta["url"],
                "score": round(hit.score, 3),
                "text": hit.get_content(),
            }
        )
    return results
```

- [ ] **Step 4: Run the test to see it pass**

Run: `.venv/bin/pytest tests/test_index.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/energiewende/index/search.py tests/test_index.py
git commit -m "feat: search function over the index" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Build the real index and try it

This downloads the embedding model once (~470 MB) and builds the real index (about 1.5 minutes).

**Files:**
- Create: `scripts/build_index.py`, `scripts/search_index.py`

**Interfaces:**
- Consumes: `dip.get_energy_papers()` (M1), `store.build_index`, `store.count_chunks`, `store.load_index`, `search.search`.
- Produces: the real index in `data/chroma` (collection `papers_400`), used by M3.

- [ ] **Step 1: Write the two scripts**

`scripts/build_index.py`:

```python
"""Download the energy papers and build the search index.

Run from the project folder:  PYTHONPATH=src .venv/bin/python scripts/build_index.py [chunk_size]
"""

import sys
import time

from energiewende.index import store
from energiewende.ingest import dip

chunk_size = int(sys.argv[1]) if len(sys.argv) > 1 else 400

papers = dip.get_energy_papers()
print(f"{len(papers)} papers downloaded")

start = time.time()
store.build_index(papers, chunk_size=chunk_size)
seconds = time.time() - start
print(f"{store.count_chunks(chunk_size)} chunks of size {chunk_size} saved in {seconds:.0f} seconds")
```

`scripts/search_index.py`:

```python
"""Ask the search index a question from the command line.

Run:  PYTHONPATH=src .venv/bin/python scripts/search_index.py "Wie wird Wasserstoff gefördert?"
"""

import sys

from energiewende.index import search, store

question = sys.argv[1]
index = store.load_index()
for hit in search.search(index, question, k=4):
    print(f"{hit['score']}  {hit['number']}  {hit['date']}  {hit['title'][:70]}")
    print("      " + hit["text"][:200].replace("\n", " "))
```

- [ ] **Step 2: Build the real index**

Run: `PYTHONPATH=src .venv/bin/python scripts/build_index.py`
Expected: `99 papers downloaded`, then about `6200 chunks of size 400 saved in` 60-120 `seconds`.

- [ ] **Step 3: Ask it two questions**

Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/search_index.py "Wie soll der Ausbau von Wasserstoff gefördert werden?"
PYTHONPATH=src .venv/bin/python scripts/search_index.py "Kritik an Kürzungen beim EEG"
```

Expected: four results each, scores around 0.7-0.8, with paper numbers like `21/2506` and passages that are about the question.

- [ ] **Step 4: Run the whole test suite**

Run: `.venv/bin/pytest -q`
Expected: 24 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_index.py scripts/search_index.py
git commit -m "feat: scripts to build and query the real index" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
