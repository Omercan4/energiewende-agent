import pytest

from energiewende import config
from energiewende.ingest import dip


def make_doc(n):
    """One document, shaped like the real DIP response."""
    return {
        "id": str(n),
        "dokumentnummer": f"21/{n}",
        "datum": "2026-09-30",
        "drucksachetyp": "Antrag",
        "titel": f"Paper {n}",
        "urheber": [{"titel": "Bundesregierung"}],
        "fundstelle": {"pdf_url": f"https://example.com/{n}.pdf"},
        "text": f"Text of paper {n}",
    }


# Three pages, found by cursor. The last page is empty and repeats its cursor.
PAGES = {
    None: {"numFound": 3, "cursor": "c1", "documents": [make_doc(1), make_doc(2)]},
    "c1": {"numFound": 3, "cursor": "c2", "documents": [make_doc(3)]},
    "c2": {"numFound": 3, "cursor": "c2", "documents": []},
}


def use_fake_dip(monkeypatch):
    """Replace downloads with the pages above. Returns the list of params sent."""
    monkeypatch.setattr(config, "DIP_API_KEY", "test-key")
    sent = []

    def fake_get_json(url, params=None, headers=None, max_age_hours=24):
        assert url == "https://search.dip.bundestag.de/api/v1/drucksache-text"
        assert headers == {"Authorization": "ApiKey test-key"}
        sent.append(params)
        return PAGES[params.get("cursor")]

    monkeypatch.setattr(dip, "get_json", fake_get_json)
    return sent


def test_follows_the_cursor_until_the_end(monkeypatch):
    sent = use_fake_dip(monkeypatch)

    papers = dip.search_papers("Strom")

    assert [p["number"] for p in papers] == ["21/1", "21/2", "21/3"]
    assert sent[0] == {"f.wahlperiode": 21, "f.titel": "Strom"}
    assert sent[1]["cursor"] == "c1"


def test_keeps_only_the_fields_we_need(monkeypatch):
    use_fake_dip(monkeypatch)

    paper = dip.search_papers("Strom")[0]

    assert paper == {
        "id": "1",
        "number": "21/1",
        "date": "2026-09-30",
        "type": "Antrag",
        "title": "Paper 1",
        "authors": ["Bundesregierung"],
        "url": "https://example.com/1.pdf",
        "text": "Text of paper 1",
    }


def test_max_docs_limits_the_result(monkeypatch):
    use_fake_dip(monkeypatch)

    papers = dip.search_papers("Strom", max_docs=2)

    assert len(papers) == 2


def test_missing_key_gives_a_clear_error(monkeypatch):
    monkeypatch.setattr(config, "DIP_API_KEY", "")

    with pytest.raises(RuntimeError, match="DIP_API_KEY"):
        dip.search_papers("Strom")


def test_energy_papers_have_no_duplicates(monkeypatch):
    use_fake_dip(monkeypatch)

    # The fake returns the same three papers for every keyword.
    papers = dip.get_energy_papers()

    assert sorted(p["id"] for p in papers) == ["1", "2", "3"]
