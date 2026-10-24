"""Find the text passages that best match a question.

Three steps, each optional:
1. vector search: the passages whose meaning is closest to the question
2. filters: keep only papers of one party / one paper type (the embedding model matches topics well,
   but not "who said it", so this is done on the metadata)
3. reranker: a second model reads question and passage together and sorts the candidates again
"""

FETCH_K = 30  # candidates to look at when filters or a reranker are used

# What the tool offers -> the text in the paper's "authors" field (the DIP "Urheber").
PARTIES = {
    "afd": "AfD",
    "gruene": "GRÜNEN",
    "linke": "LINKE",
    "bundesregierung": "Bundesregierung",
    "ausschuss": "Ausschuss",
}
PAPER_TYPES = ["Gesetzentwurf", "Antrag", "Kleine Anfrage", "Antwort", "Beschlussempfehlung und Bericht", "Unterrichtung"]


def check_party(party):
    if party is not None and party not in PARTIES:
        raise ValueError(f"Unknown party '{party}'. Allowed: {', '.join(PARTIES)}.")


def search(index, question, k=4, party=None, paper_type=None, reranker=None):
    """The k best passages for the question, best first, with their source.

    party: one of PARTIES (who wrote the paper), paper_type: for example "Antrag".
    reranker: an object with predict([(question, text), ...]) -> scores, like a sentence-transformers CrossEncoder.
    """
    check_party(party)
    needs_candidates = party or paper_type or reranker
    nodes = index.as_retriever(similarity_top_k=FETCH_K if needs_candidates else k).retrieve(question)

    if party:
        nodes = [n for n in nodes if PARTIES[party].lower() in n.metadata["authors"].lower()]
    if paper_type:
        nodes = [n for n in nodes if n.metadata["type"] == paper_type]

    scores = [n.score for n in nodes]
    if reranker and nodes:
        scores = [float(s) for s in reranker.predict([(question, n.get_content()) for n in nodes])]
        ranked = sorted(zip(scores, range(len(nodes))), key=lambda pair: (-pair[0], pair[1]))  # stable for equal scores
        nodes, scores = [nodes[i] for _, i in ranked], [s for s, _ in ranked]

    results = []
    for node, score in list(zip(nodes, scores))[:k]:
        meta = node.metadata
        results.append(
            {
                "number": meta["number"],
                "date": meta["date"],
                "title": meta["title"],
                "url": meta["url"],
                "score": round(score, 3),
                "text": node.get_content(),
            }
        )
    return results
