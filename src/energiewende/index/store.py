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
