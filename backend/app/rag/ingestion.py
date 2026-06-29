"""
Ingestion orchestration.

This is the one function both the `/ingest` endpoint and the standalone
CLI script call. Keeping the orchestration here (rather than duplicating
it in both the route and the script) means there's exactly one place that
defines "what does it mean to ingest a file."
"""
import logging
from dataclasses import dataclass
from pathlib import Path

from app.db.chroma_store import add_documents
from app.rag.loaders import UnsupportedFileTypeError, load_document
from app.rag.splitter import split_documents

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    filename: str
    pages_or_sections_loaded: int
    chunks_created: int
    chunk_ids: list[str]


class IngestionError(Exception):
    """Wraps any failure during the load -> split -> embed -> store pipeline
    with the filename that caused it, so callers (route or CLI) can report
    a useful error instead of a bare traceback."""

    def __init__(self, filename: str, original: Exception):
        self.filename = filename
        self.original = original
        super().__init__(f"Failed to ingest '{filename}': {original}")


def ingest_file(file_path: Path) -> IngestResult:
    """
    Run the full ingestion pipeline for a single file already saved to disk.

    Raises IngestionError (wrapping UnsupportedFileTypeError or any other
    failure) so callers get a consistent error type regardless of which
    stage failed.
    """
    try:
        docs = load_document(file_path)
        if not docs:
            logger.warning("No content extracted from %s", file_path.name)

        chunks = split_documents(docs)
        chunk_ids = add_documents(chunks)

        logger.info(
            "Ingested %s: %d source section(s) -> %d chunk(s)",
            file_path.name,
            len(docs),
            len(chunks),
        )

        return IngestResult(
            filename=file_path.name,
            pages_or_sections_loaded=len(docs),
            chunks_created=len(chunks),
            chunk_ids=chunk_ids,
        )
    except UnsupportedFileTypeError:
        raise  # let callers distinguish "bad file type" from other failures
    except Exception as exc:  # noqa: BLE001 — intentionally broad: this is a pipeline boundary
        raise IngestionError(file_path.name, exc) from exc
