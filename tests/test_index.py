from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import MetadataMode

from energiewende.index import search, store


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


class FakeNode:
    """Looks like a LlamaIndex search hit."""

    def __init__(self, number, authors, paper_type, text, score):
        self.metadata = {"number": number, "date": "2026-09-30", "title": "Titel", "url": f"https://x/{number}.pdf",
                         "authors": authors, "type": paper_type}
        self.score = score
        self._text = text

    def get_content(self):
        return self._text


class FakeIndex:
    """Remembers how many hits were asked for and returns the nodes it was given."""

    def __init__(self, nodes):
        self.nodes = nodes
        self.asked_for = None

    def as_retriever(self, similarity_top_k):
        self.asked_for = similarity_top_k
        index = self

        class Retriever:
            def retrieve(self, question):
                return index.nodes[:similarity_top_k]

        return Retriever()


NODES = [
    FakeNode("21/1", "Fraktion der AfD", "Antrag", "AfD Antrag zur Kernenergie", 0.9),
    FakeNode("21/2", "Bundesregierung", "Gesetzentwurf", "Gesetzentwurf zum EEG", 0.8),
    FakeNode("21/3", "Fraktion BÜNDNIS 90/DIE GRÜNEN", "Antrag", "Grüner Antrag zum Heizen", 0.7),
    FakeNode("21/4", "Fraktion der AfD", "Kleine Anfrage", "AfD Anfrage zum EEG", 0.6),
]


def test_search_without_filters_asks_for_exactly_k():
    index = FakeIndex(NODES)

    hits = search.search(index, "EEG", k=2)

    assert index.asked_for == 2
    assert [h["number"] for h in hits] == ["21/1", "21/2"]


def test_party_filter_keeps_only_papers_of_that_party():
    index = FakeIndex(NODES)

    hits = search.search(index, "EEG", k=4, party="afd")

    assert [h["number"] for h in hits] == ["21/1", "21/4"]
    assert index.asked_for == search.FETCH_K  # looks at more candidates, because filtering removes some


def test_paper_type_filter():
    hits = search.search(FakeIndex(NODES), "EEG", k=4, paper_type="Antrag")

    assert [h["number"] for h in hits] == ["21/1", "21/3"]


def test_filters_can_be_combined():
    hits = search.search(FakeIndex(NODES), "EEG", k=4, party="afd", paper_type="Kleine Anfrage")

    assert [h["number"] for h in hits] == ["21/4"]


def test_unknown_party_is_rejected_with_the_allowed_values():
    try:
        search.search(FakeIndex(NODES), "EEG", party="cdu")
        assert False, "should have raised"
    except ValueError as error:
        assert "afd" in str(error) and "gruene" in str(error)


def test_reranker_changes_the_order_and_keeps_the_best_k():
    class FakeReranker:
        def predict(self, pairs):
            # prefers texts that mention "EEG"
            return [1.0 if "EEG" in text else 0.0 for _, text in pairs]

    index = FakeIndex(NODES)

    hits = search.search(index, "EEG", k=2, reranker=FakeReranker())

    assert [h["number"] for h in hits] == ["21/2", "21/4"]
    assert index.asked_for == search.FETCH_K
    assert hits[0]["score"] == 1.0
