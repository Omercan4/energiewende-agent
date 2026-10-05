"""Printed papers (Drucksachen) from the Bundestag DIP API.

The API needs a key, sent in the Authorization header. We only download
papers whose title contains an energy keyword, because full texts are long
and we want a small corpus.
"""

from energiewende import config
from energiewende.ingest.download import get_json

URL = "https://search.dip.bundestag.de/api/v1/drucksache-text"
SOURCE = "Deutscher Bundestag, DIP"
ENERGY_KEYWORDS = ["Energie", "Strom", "Erneuerbare", "Wasserstoff", "Wärme"]


def to_paper(doc):
    """Keep only the fields we need from one DIP document."""
    return {
        "id": doc["id"],
        "number": doc["dokumentnummer"],
        "date": doc["datum"],
        "type": doc.get("drucksachetyp", ""),
        "title": doc["titel"],
        "authors": [u["titel"] for u in doc.get("urheber", [])],
        "url": doc.get("fundstelle", {}).get("pdf_url", ""),
        "text": doc.get("text") or "",
    }


def search_papers(keyword, wahlperiode=21, max_docs=50):
    """Papers of one electoral period whose title contains the keyword."""
    if not config.DIP_API_KEY:
        raise RuntimeError("DIP_API_KEY is not set. See .env.example.")

    headers = {"Authorization": "ApiKey " + config.DIP_API_KEY}
    params = {"f.wahlperiode": wahlperiode, "f.titel": keyword}
    papers = []

    while len(papers) < max_docs:
        page = get_json(URL, params=params, headers=headers)
        if not page["documents"]:
            break  # no more results
        papers += [to_paper(doc) for doc in page["documents"]]
        # Each page gives the cursor for the next page. When there are
        # no more pages, the API returns the same cursor again.
        if page["cursor"] == params.get("cursor"):
            break
        params = {**params, "cursor": page["cursor"]}

    return papers[:max_docs]


def get_energy_papers(wahlperiode=21, max_docs=200):
    """Papers for all energy keywords, without duplicates."""
    papers = {}  # id -> paper, so the same paper is stored only once
    for keyword in ENERGY_KEYWORDS:
        for paper in search_papers(keyword, wahlperiode, max_docs):
            papers[paper["id"]] = paper
    return list(papers.values())[:max_docs]
