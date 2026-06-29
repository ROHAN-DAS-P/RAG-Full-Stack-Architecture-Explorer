"""
ChromaDB wrapper.

Owns the one place in the app that talks to Chroma directly. Everything
else (ingestion endpoint, retriever) goes through `get_vector_store()` so
that swapping persistence details later only touches this file.

Persistence: Chroma is configured with a local on-disk directory
(settings.chroma_persist_dir), so the index survives process restarts —
no server, no Docker, no paid service.
"""
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.config import get_settings
from app.rag.embeddings import get_embedding_function


@lru_cache
def get_vector_store() -> Chroma:
    """
    Cached singleton for the Chroma client + collection.

    Cached for the same reason as the embedding function: re-opening the
    persistent store on every call would be wasteful and risks file-lock
    contention against itself within one process.
    """
    settings = get_settings()
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embedding_function(),
        persist_directory=str(settings.chroma_persist_dir),
    )


def add_documents(chunks: list[Document]) -> list[str]:
    """
    Embed and upsert a list of chunked Documents into the persistent store.

    Returns the list of generated Chroma IDs (useful for logging / tests).
    Empty input is a no-op that returns an empty list rather than raising,
    so callers don't need to special-case "ingested a file with 0 chunks".
    """
    if not chunks:
        return []

    store = get_vector_store()
    return store.add_documents(chunks)


def similarity_search(query: str, k: int | None = None) -> list[Document]:
    """
    Top-k similarity search against the persistent store.

    `k` defaults to settings.retrieval_top_k so callers don't need to know
    the configured default; Phase 3's retriever module will be the main
    caller of this function.
    """
    settings = get_settings()
    store = get_vector_store()
    return store.similarity_search(query, k=k or settings.retrieval_top_k)


def collection_count() -> int:
    """Number of chunks currently stored — used by the ingest endpoint's
    response and by health/diagnostics."""
    store = get_vector_store()
    return store._collection.count()  # noqa: SLF001 — Chroma doesn't expose this publicly yet
