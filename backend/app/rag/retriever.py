"""
Retrieval.

Thin layer on top of app.db.chroma_store.similarity_search that:
  1. Runs the top-k search for a user query
  2. Shapes the results into a consistent `RetrievedChunk` the rest of the
     chat pipeline (prompt builder, WebSocket route) can rely on, instead
     of passing raw LangChain Documents around everywhere.

Kept separate from chroma_store.py because retrieval logic (e.g. future
re-ranking, score thresholds, hybrid search) belongs here, while
chroma_store.py should stay a dumb persistence wrapper.
"""
from dataclasses import dataclass

from app.db.chroma_store import similarity_search


@dataclass
class RetrievedChunk:
    text: str
    source: str
    file_type: str
    page: int | None = None
    line_start: int | None = None
    line_end: int | None = None

    def citation_label(self) -> str:
        """Human-readable citation, e.g. 'architecture.pdf, p.4' or
        'readme.md, lines 12-18'. Used in the prompt sent to the LLM and
        can also be reused as-is by the frontend if it doesn't want to
        build its own formatting."""
        if self.file_type == "pdf" and self.page is not None:
            return f"{self.source}, p.{self.page}"
        if self.line_start is not None and self.line_end is not None:
            return f"{self.source}, lines {self.line_start}-{self.line_end}"
        return self.source


def retrieve(query: str, k: int | None = None) -> list[RetrievedChunk]:
    """
    Run similarity search and return shaped, citation-ready chunks.

    Returns an empty list (not an error) when nothing is found — callers
    decide how to handle "no context" (e.g. the WebSocket route sends an
    explicit warning to the client rather than silently hallucinating).
    """
    docs = similarity_search(query, k=k)

    chunks: list[RetrievedChunk] = []
    for doc in docs:
        meta = doc.metadata
        chunks.append(
            RetrievedChunk(
                text=doc.page_content,
                source=meta.get("source", "unknown"),
                file_type=meta.get("file_type", "unknown"),
                page=meta.get("page"),
                line_start=meta.get("line_start"),
                line_end=meta.get("line_end"),
            )
        )
    return chunks
