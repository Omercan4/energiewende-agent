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
