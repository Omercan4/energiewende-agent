"""Build and load the search index for the Bundestag papers.

Each paper is split into chunks of about 400 tokens. A local embedding model
turns every chunk into a vector (a list of 384 numbers). Texts with similar
meaning get similar vectors. The vectors are saved in a Chroma database.
"""

import chromadb
from chromadb.config import Settings
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from energiewende import config


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
