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
