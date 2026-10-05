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
