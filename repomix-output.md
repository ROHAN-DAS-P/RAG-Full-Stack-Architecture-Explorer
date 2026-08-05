This file is a merged representation of the entire codebase, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
````
backend/
  app/
    core/
      __init__.py
      config.py
    db/
      __init__.py
      chroma_store.py
    rag/
      __init__.py
      embeddings.py
      ingestion.py
      loaders.py
      ollama_client.py
      prompt.py
      retriever.py
      splitter.py
      ws_protocol.py
    routes/
      __init__.py
      chat.py
      health.py
      ingest.py
    scripts/
      __init__.py
      ingest_cli.py
    __init__.py
    main.py
  .gitignore
  .python-version
  README.md
  requirements.txt
  test_chat.py
frontend/
  public/
    favicon.svg
    icons.svg
  src/
    assets/
      hero.png
      react.svg
      vite.svg
    hooks/
      useChatWebSocket.js
    store/
      chatSlice.js
      store.js
    App.css
    App.jsx
    index.css
    main.jsx
  .gitignore
  eslint.config.js
  index.html
  package.json
  README.md
  vite.config.js
repomix-output.xml
````

# Files

## File: backend/app/core/__init__.py
````python

````

## File: backend/app/core/config.py
````python
"""
Centralized application configuration.

All tunables (model names, chunk sizes, paths, timeouts) live here so that
ingestion, retrieval, and chat logic never hardcode values. Override any
of these via a `.env` file in /backend or real environment variables.
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent  # /backend


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- App ---
    app_name: str = "Full-Stack Architecture Explorer"
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # --- Storage paths ---
    data_dir: Path = BACKEND_ROOT / "data"
    chroma_persist_dir: Path = BACKEND_ROOT / "data" / "chroma_db"
    upload_dir: Path = BACKEND_ROOT / "data" / "uploads"

    # --- Embeddings ---
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Chunking ---
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # --- Vector store / retrieval ---
    chroma_collection_name: str = "docs"
    retrieval_top_k: int = 4

    # --- Ollama / LLM ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    # ollama_model: str = "llama3.2:1b"

    ollama_timeout_seconds: int = 60
    llm_temperature: float = 0.0

    # --- WebSocket ---
    ws_heartbeat_seconds: int = 30


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton so we parse .env / env vars only once."""
    settings = Settings()
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return settings
````

## File: backend/app/db/__init__.py
````python

````

## File: backend/app/db/chroma_store.py
````python
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
````

## File: backend/app/rag/__init__.py
````python

````

## File: backend/app/rag/embeddings.py
````python
"""
Embeddings.

Wraps a local HuggingFace sentence-transformers model so the rest of the
app never imports `langchain_huggingface` directly. The model is loaded
once and cached, since loading it from disk/HF cache on every request
would be slow and pointless (the weights don't change).

No API keys, no network calls at inference time (the model is downloaded
once to the local HF cache on first run, then used fully offline).
"""
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import get_settings


@lru_cache
def get_embedding_function() -> HuggingFaceEmbeddings:
    """
    Cached singleton so the embedding model is loaded into memory exactly
    once per process, regardless of how many times this is called.
    """
    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        # CPU is the safe default for a "runs anywhere locally" project.
        # If the user has a CUDA GPU available, they can override this by
        # editing model_kwargs below — left explicit here rather than
        # auto-detected so behavior is predictable across machines.
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
````

## File: backend/app/rag/ingestion.py
````python
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
````

## File: backend/app/rag/loaders.py
````python
"""
Document loading.

Responsible for turning a raw file on disk (PDF or Markdown) into a list of
LangChain `Document` objects, each carrying metadata we'll need later for
citations: source filename and a page/line locator.

This module does NOT chunk or embed — that's the job of `splitter.py` and
`embeddings.py`. Keeping load/split/embed separate makes each piece testable
and replaceable on its own.
"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown"}


class UnsupportedFileTypeError(ValueError):
    """Raised when a file extension isn't one we know how to ingest."""


def load_document(file_path: Path) -> list[Document]:
    """
    Load a single file into LangChain Documents.

    - PDF: one Document per page, with `page` metadata (1-indexed) set by
      PyPDFLoader (it's 0-indexed natively; we normalize to 1-indexed here
      since that's what a human looking at the PDF would expect).
    - Markdown: one Document per file, with `line_start`/`line_end` set to
      cover the whole file. (Per-chunk line numbers are refined later in
      the splitter, since chunk boundaries don't exist yet at load time.)

    Raises UnsupportedFileTypeError for anything else.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(file_path)
    if suffix in (".md", ".markdown"):
        return _load_markdown(file_path)

    raise UnsupportedFileTypeError(
        f"Unsupported file type '{suffix}' for {file_path.name}. "
        f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
    )


def _load_pdf(file_path: Path) -> list[Document]:
    loader = PyPDFLoader(str(file_path))
    raw_docs = loader.load()  # one Document per page, 0-indexed "page" metadata

    docs: list[Document] = []
    for doc in raw_docs:
        page_zero_indexed = doc.metadata.get("page", 0)
        docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": file_path.name,
                    "page": page_zero_indexed + 1,  # human-friendly, 1-indexed
                    "file_type": "pdf",
                },
            )
        )
    return docs


def _load_markdown(file_path: Path) -> list[Document]:
    text = file_path.read_text(encoding="utf-8")
    line_count = text.count("\n") + 1

    return [
        Document(
            page_content=text,
            metadata={
                "source": file_path.name,
                "line_start": 1,
                "line_end": line_count,
                "file_type": "markdown",
            },
        )
    ]
````

## File: backend/app/rag/ollama_client.py
````python
"""
Ollama client wrapper.

Isolates all direct interaction with the `ollama` Python package so that
the WebSocket route doesn't need to know about Ollama's client API,
timeouts, or connection errors directly.

Streaming is exposed as a generator yielding plain text chunks. Cancellation
is cooperative: the caller (the WebSocket route) is responsible for
breaking out of the `for` loop early when it receives a stop signal from
the client — Python generators stop producing work the moment you stop
iterating them, so no extra cancellation plumbing is needed here.
"""
import logging
from collections.abc import Iterator

import httpx
import ollama

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class OllamaConnectionError(Exception):
    """Raised when Ollama can't be reached at all (not running, wrong port)."""


class OllamaTimeoutError(Exception):
    """Raised when Ollama is reachable but a generation call exceeds the
    configured timeout."""


def stream_completion(prompt: str) -> Iterator[str]:
    """
    Stream a completion from Ollama token-chunk by token-chunk.

    Yields plain text pieces as they arrive. Raises OllamaConnectionError
    or OllamaTimeoutError on failure so the caller can send a clean error
    message to the client instead of an unhandled exception killing the
    WebSocket connection.
    """
    settings = get_settings()
    client = ollama.Client(host=settings.ollama_base_url, timeout=settings.ollama_timeout_seconds)

    try:
        stream = client.generate(
            model=settings.ollama_model,
            prompt=prompt,
            stream=True,
            options={"temperature": settings.llm_temperature},
        )
        for chunk in stream:
            text_piece = chunk.get("response", "")
            if text_piece:
                yield text_piece
            if chunk.get("done"):
                break
    except ollama.ResponseError as exc:
        # Model not pulled, bad request, etc. — Ollama returned an error
        # response rather than failing the connection outright.
        logger.error("Ollama response error: %s", exc)
        raise OllamaConnectionError(
            f"Ollama returned an error (is '{settings.ollama_model}' pulled? "
            f"Run: ollama pull {settings.ollama_model}): {exc}"
        ) from exc
    except (ConnectionError, TimeoutError, httpx.ConnectError, httpx.TimeoutException) as exc:
        logger.error("Ollama unreachable at %s: %s", settings.ollama_base_url, exc)
        raise OllamaConnectionError(
            f"Could not reach Ollama at {settings.ollama_base_url}. "
            f"Is it running? Try: ollama serve"
        ) from exc
    except Exception as exc:  # noqa: BLE001 — last-resort boundary around a 3rd-party client
        logger.exception("Unexpected error streaming from Ollama")
        raise OllamaConnectionError(f"Unexpected error from Ollama: {exc}") from exc
````

## File: backend/app/rag/prompt.py
````python
"""
Prompt construction.

Builds the exact string sent to Ollama. Kept as plain string templating
(no LangChain PromptTemplate/Runnable chain) so it's trivial to read,
test, and tweak the wording without learning LangChain's chain API for
something this simple — LangChain earns its keep in the loaders/splitters,
not necessarily here.

The instruction to cite using [n] markers is what lets the frontend
(Phase 5) regex-match citation markers in the streamed text and map them
back to the `citations` list sent alongside the stream.
"""
from app.rag.retriever import RetrievedChunk

SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant answering questions about software "
    "architecture documentation. Answer ONLY using the provided context. "
    "If the context does not contain the answer, say you don't have "
    "enough information in the ingested documents — do not make things up.\n\n"
    "When you state a fact from the context, cite it inline using the "
    "format [n], where n is the chunk number from the Context section "
    "below. Example: \"The backend uses FastAPI [1].\""
    # "You are a strict, factual assistant answering questions about software "
    # "architecture documentation. Answer ONLY using the explicitly provided context. "
    # "If the answer is not explicitly written in the context, do not make it up. "
    # "State that you do not know. Stick strictly to the facts provided.\n\n"
    # "When you state a fact from the context, cite it inline using the "
    # "format [n], where n is the chunk number from the Context section "
    # "below. Example: \"The backend uses FastAPI [1].\""
)


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    """
    Assemble the full prompt string: system instructions, numbered
    context chunks (each labeled with its citation), then the user query.

    If `chunks` is empty, the context section explicitly says so rather
    than being silently omitted — this nudges the model toward the
    "I don't have enough information" answer instead of guessing.
    """
    if not chunks:
        context_block = "(No relevant context was found in the ingested documents.)"
    else:
        context_block = "\n\n".join(
            f"[{i+1}] (Source: {chunk.citation_label()})\n{chunk.text}"
            for i, chunk in enumerate(chunks)
        )

    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"--- Context ---\n{context_block}\n\n"
        f"--- Question ---\n{query}\n\n"
        f"--- Answer ---\n"
    )
````

## File: backend/app/rag/retriever.py
````python
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
````

## File: backend/app/rag/splitter.py
````python
"""
Chunking.

Splits loaded Documents into overlapping chunks suitable for embedding.
Chunk size/overlap come from app.core.config so they're tunable without
touching this code.

Citation note: for PDFs, every chunk inherits the `page` metadata from its
source page Document (set in loaders.py), since PyPDFLoader already gives
us one Document per page. For Markdown, the source Document spans the
whole file, so after splitting we recompute approximate line numbers per
chunk by locating each chunk's text within the original content. This is
what lets the frontend later cite "source.md, lines 40-58" instead of just
"source.md".
"""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings


def split_documents(docs: list[Document]) -> list[Document]:
    """
    Split a list of loaded Documents into overlapping chunks.

    Each output Document's metadata always contains at least:
      - source: original filename
      - file_type: "pdf" | "markdown"
    plus either:
      - page: int (PDF)
      - line_start, line_end: int (Markdown)
    """
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # Prefer splitting on paragraph/sentence boundaries before falling
        # back to raw character cuts, so chunks stay semantically coherent.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for doc in docs:
        if doc.metadata.get("file_type") == "markdown":
            chunks.extend(_split_markdown_with_line_numbers(doc, splitter))
        else:
            # PDF (or any future page-based format): every chunk just
            # inherits the parent page's metadata as-is.
            for chunk in splitter.split_documents([doc]):
                chunks.append(chunk)

    return chunks


def _split_markdown_with_line_numbers(
    doc: Document, splitter: RecursiveCharacterTextSplitter
) -> list[Document]:
    """
    Split a whole-file Markdown Document into chunks, then back-compute each
    chunk's line range by finding its text's offset in the original content.

    This is a best-effort locator: if a chunk's exact text can't be found
    (which shouldn't normally happen since the splitter only cuts the
    original text), we fall back to the parent document's full line range
    rather than failing the whole ingestion.
    """
    full_text = doc.page_content
    sub_chunks = splitter.split_text(full_text)

    result: list[Document] = []
    search_start = 0
    for chunk_text in sub_chunks:
        offset = full_text.find(chunk_text, search_start)
        if offset == -1:
            # Overlap can occasionally make forward-only search miss a
            # match; retry from the beginning once before giving up.
            offset = full_text.find(chunk_text)

        if offset == -1:
            line_start = doc.metadata["line_start"]
            line_end = doc.metadata["line_end"]
        else:
            line_start = full_text.count("\n", 0, offset) + 1
            line_end = line_start + chunk_text.count("\n")
            search_start = offset + 1  # allow overlapping matches to advance

        result.append(
            Document(
                page_content=chunk_text,
                metadata={
                    "source": doc.metadata["source"],
                    "file_type": "markdown",
                    "line_start": line_start,
                    "line_end": line_end,
                },
            )
        )

    return result
````

## File: backend/app/rag/ws_protocol.py
````python
"""
WebSocket message protocol for /ws/chat.

Defining this as Pydantic models (rather than ad-hoc dicts scattered
through the route) gives us validation on incoming client messages and a
single source of truth for what the frontend should expect to receive.
This file is the "API contract" between Phase 3 (backend) and Phase 5
(frontend WebSocket hook) — keep it in sync with the frontend types when
either side changes.

--- Client -> Server ---
  {"type": "query", "payload": {"text": "..."}}
  {"type": "stop"}

--- Server -> Client ---
  {"type": "citations", "payload": {"citations": [...]}}   # sent once, before tokens
  {"type": "token", "payload": {"text": "..."}}            # sent many times
  {"type": "done"}                                          # sent once, stream finished
  {"type": "error", "payload": {"message": "..."}}         # sent on failure, stream ends
  {"type": "stopped"}                                       # sent once, if client cancelled
"""
from typing import Literal

from pydantic import BaseModel, ValidationError


class ClientQueryPayload(BaseModel):
    text: str


class ClientQueryMessage(BaseModel):
    type: Literal["query"]
    payload: ClientQueryPayload


class ClientStopMessage(BaseModel):
    type: Literal["stop"]


class CitationPayload(BaseModel):
    index: int  # matches the [n] marker used in the prompt/response text
    source: str
    file_type: str
    page: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    label: str  # pre-formatted, e.g. "architecture.pdf, p.4"
    snippet: str  # the chunk text itself, for the frontend's citation panel


class ServerCitationsMessage(BaseModel):
    type: Literal["citations"] = "citations"
    payload: dict[str, list[CitationPayload]]


class ServerTokenMessage(BaseModel):
    type: Literal["token"] = "token"
    payload: dict[str, str]


class ServerDoneMessage(BaseModel):
    type: Literal["done"] = "done"


class ServerStoppedMessage(BaseModel):
    type: Literal["stopped"] = "stopped"


class ServerErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    payload: dict[str, str]


class ProtocolError(Exception):
    """Raised when an incoming client message doesn't match any known
    message type, so the route can send a clean ServerErrorMessage instead
    of crashing the connection on a raw ValidationError/KeyError."""


def parse_client_message(raw: dict) -> ClientQueryMessage | ClientStopMessage:
    """Validate and parse a raw client WebSocket message dict into one of
    the known typed messages."""
    msg_type = raw.get("type")

    try:
        if msg_type == "query":
            return ClientQueryMessage.model_validate(raw)
        if msg_type == "stop":
            return ClientStopMessage.model_validate(raw)
    except ValidationError as exc:
        raise ProtocolError(f"Malformed '{msg_type}' message: {exc}") from exc

    raise ProtocolError(f"Unknown message type: {msg_type!r}")
````

## File: backend/app/routes/__init__.py
````python

````

## File: backend/app/routes/chat.py
````python
"""
WebSocket streaming chat endpoint.

Protocol is defined in app/rag/ws_protocol.py. Summary of the message flow
for a single query:

  client --query--> server
  server --citations--> client      (sent once, before any tokens, so the
                                      frontend can render citation badges
                                      as soon as the answer starts streaming)
  server --token--> client           (repeated, as Ollama streams)
  server --done--> client            (stream finished normally)

  OR, if the client sends `stop` mid-stream:
  server --stopped--> client         (generation cancelled, no further tokens)

  OR, on failure at any point:
  server --error--> client           (stream ends; connection stays open
                                       for the next query)

Concurrency design: a single background task (`_reader`) owns
`websocket.receive_json()` for the lifetime of the connection. Every
incoming message is pushed onto an asyncio.Queue. The main loop pulls
"query" messages off that queue and processes them one at a time; "stop"
messages instead set an asyncio.Event directly from the reader task, so a
stop sent while a query is mid-stream is observed immediately rather than
waiting for the next `receive_json()` call (which is what a naive
sequential receive-then-process loop would do — see inline comment below
for why that matters).

A "stop" with nothing in flight is harmless: it just sets an Event that
gets cleared again at the start of the next query.
"""
import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.rag.ollama_client import OllamaConnectionError, stream_completion
from app.rag.prompt import build_prompt
from app.rag.retriever import retrieve
from app.rag.ws_protocol import (
    ClientQueryMessage,
    ClientStopMessage,
    CitationPayload,
    ProtocolError,
    parse_client_message,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@router.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket) -> None:
    await websocket.accept()

    stop_event = asyncio.Event()
    query_queue: asyncio.Queue[str] = asyncio.Queue()
    disconnect_event = asyncio.Event()

    reader_task = asyncio.create_task(
        _reader(websocket, query_queue, stop_event, disconnect_event)
    )

    try:
        while not disconnect_event.is_set():
            get_query = asyncio.create_task(query_queue.get())
            wait_disconnect = asyncio.create_task(disconnect_event.wait())
            done, pending = await asyncio.wait(
                {get_query, wait_disconnect}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()

            if wait_disconnect in done:
                break

            query_text = get_query.result()
            stop_event.clear()
            await _handle_query(websocket, query_text, stop_event)
    finally:
        reader_task.cancel()


async def _reader(
    websocket: WebSocket,
    query_queue: asyncio.Queue[str],
    stop_event: asyncio.Event,
    disconnect_event: asyncio.Event,
) -> None:
    """
    Continuously reads incoming client messages for the life of the
    connection. This runs concurrently with `_handle_query` in the main
    loop, which is what lets a "stop" message take effect immediately
    instead of waiting behind whatever `_handle_query` is currently doing.

    Without this split, a single sequential loop (receive -> process ->
    receive -> process) would only ever check for a new message *after*
    the current query's stream finished — by which point "stop" arrived
    too late to do anything.
    """
    try:
        while True:
            raw = await websocket.receive_json()
            try:
                message = parse_client_message(raw)
            except ProtocolError as exc:
                await websocket.send_json(
                    {"type": "error", "payload": {"message": str(exc)}}
                )
                continue

            if isinstance(message, ClientStopMessage):
                stop_event.set()
            elif isinstance(message, ClientQueryMessage):
                await query_queue.put(message.payload.text)
    except WebSocketDisconnect:
        logger.info("Client disconnected from /ws/chat")
    finally:
        disconnect_event.set()


async def _handle_query(websocket: WebSocket, query_text: str, stop_event: asyncio.Event) -> None:
    query_text = query_text.strip()
    if not query_text:
        await websocket.send_json(
            {"type": "error", "payload": {"message": "Query text cannot be empty."}}
        )
        return

    # --- Retrieval ---
    try:
        chunks = await asyncio.to_thread(retrieve, query_text)
    except Exception as exc:  # noqa: BLE001 — boundary around the vector store
        logger.exception("Retrieval failed for query: %r", query_text)
        await websocket.send_json(
            {"type": "error", "payload": {"message": f"Retrieval failed: {exc}"}}
        )
        return

    citations = [
        CitationPayload(
            index=i + 1,
            source=chunk.source,
            file_type=chunk.file_type,
            page=chunk.page,
            line_start=chunk.line_start,
            line_end=chunk.line_end,
            label=chunk.citation_label(),
            snippet=chunk.text,
        )
        for i, chunk in enumerate(chunks)
    ]
    await websocket.send_json(
        {"type": "citations", "payload": {"citations": [c.model_dump() for c in citations]}}
    )

    # --- Generation ---
    prompt = build_prompt(query_text, chunks)

    try:
        async for text_piece in _astream(prompt):
            if stop_event.is_set():
                await websocket.send_json({"type": "stopped"})
                return
            await websocket.send_json({"type": "token", "payload": {"text": text_piece}})
    except OllamaConnectionError as exc:
        await websocket.send_json({"type": "error", "payload": {"message": str(exc)}})
        return
    except Exception as exc:  # noqa: BLE001 — boundary around the LLM call
        logger.exception("Unexpected error during generation")
        await websocket.send_json(
            {"type": "error", "payload": {"message": f"Unexpected error: {exc}"}}
        )
        return

    await websocket.send_json({"type": "done"})


async def _astream(prompt: str):
    """
    Bridge Ollama's synchronous generator (stream_completion) into an async
    generator, since FastAPI's WebSocket send calls are async but the
    `ollama` client's streaming API is blocking/sync.

    Each `next()` call runs in the default executor so the event loop isn't
    blocked while waiting on Ollama — necessary so the `_reader` task (and
    therefore stop-message handling) keeps running concurrently.
    """
    iterator = stream_completion(prompt)
    loop = asyncio.get_event_loop()

    def _next_or_sentinel():
        try:
            return next(iterator)
        except StopIteration:
            return None

    while True:
        piece = await loop.run_in_executor(None, _next_or_sentinel)
        if piece is None:
            break
        yield piece
````

## File: backend/app/routes/health.py
````python
"""
Simple liveness/readiness endpoint.

Kept dumb on purpose: it should never import heavy RAG/embedding/DB code,
so that `/health` stays fast and reliable even if the vector store or
Ollama is down. Phase 3 will likely add a `/health/ready` variant that
checks Ollama + Chroma connectivity.
"""
from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "app": settings.app_name,
    }
````

## File: backend/app/routes/ingest.py
````python
"""
Ingestion endpoint.

Accepts a file upload, saves it to disk under settings.upload_dir, then
runs it through the ingestion pipeline (load -> split -> embed -> store).

Error handling per the engineering constraints:
  - Unsupported file type -> 400, not 500
  - Any other pipeline failure (corrupt PDF, encoding issue, etc.) -> 500
    with a message, not a bare stack trace leaked to the client
"""
import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.core.config import get_settings
from app.rag.ingestion import IngestionError, ingest_file
from app.rag.loaders import SUPPORTED_EXTENSIONS, UnsupportedFileTypeError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ingestion"])


class IngestResponse(BaseModel):
    filename: str
    pages_or_sections_loaded: int
    chunks_created: int


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest(file: UploadFile) -> IngestResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
            ),
        )

    settings = get_settings()
    destination = settings.upload_dir / file.filename

    try:
        with destination.open("wb") as out_file:
            shutil.copyfileobj(file.file, out_file)
    except OSError as exc:
        logger.exception("Failed to save uploaded file %s", file.filename)
        raise HTTPException(
            status_code=500, detail=f"Could not save uploaded file: {exc}"
        ) from exc

    try:
        result = ingest_file(destination)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except IngestionError as exc:
        logger.exception("Ingestion pipeline failed for %s", file.filename)
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed for '{exc.filename}': {exc.original}",
        ) from exc

    return IngestResponse(
        filename=result.filename,
        pages_or_sections_loaded=result.pages_or_sections_loaded,
        chunks_created=result.chunks_created,
    )
````

## File: backend/app/scripts/__init__.py
````python

````

## File: backend/app/scripts/ingest_cli.py
````python
#!/usr/bin/env python3
"""
Bulk ingestion CLI.

Usage:
    python -m app.scripts.ingest_cli /path/to/docs_folder
    python -m app.scripts.ingest_cli /path/to/single_file.pdf

Walks a folder (or takes a single file), runs every supported file through
the same ingestion pipeline the `/ingest` endpoint uses, and prints a
summary. This exists because re-uploading 50 PDFs through curl/Postman one
at a time is not anyone's idea of a good afternoon.
"""
import logging
import sys
from pathlib import Path

from app.rag.ingestion import IngestionError, ingest_file
from app.rag.loaders import SUPPORTED_EXTENSIONS, UnsupportedFileTypeError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def collect_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(
            p for p in target.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    raise FileNotFoundError(f"Path does not exist: {target}")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"Usage: python -m app.scripts.ingest_cli <file_or_folder>", file=sys.stderr)
        return 1

    target = Path(argv[1]).expanduser().resolve()
    try:
        files = collect_files(target)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1

    if not files:
        logger.warning(
            "No supported files (%s) found under %s", sorted(SUPPORTED_EXTENSIONS), target
        )
        return 0

    logger.info("Found %d file(s) to ingest", len(files))

    succeeded, failed = 0, 0
    for file_path in files:
        try:
            result = ingest_file(file_path)
            logger.info(
                "  ✓ %s — %d section(s) -> %d chunk(s)",
                result.filename,
                result.pages_or_sections_loaded,
                result.chunks_created,
            )
            succeeded += 1
        except UnsupportedFileTypeError as exc:
            logger.error("  ✗ %s — %s", file_path.name, exc)
            failed += 1
        except IngestionError as exc:
            logger.error("  ✗ %s — %s", exc.filename, exc.original)
            failed += 1

    logger.info("Done. %d succeeded, %d failed.", succeeded, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
````

## File: backend/app/__init__.py
````python

````

## File: backend/app/main.py
````python
"""
FastAPI application entrypoint.

This file should stay thin: wire up middleware and routers only.
Business logic (ingestion, retrieval, streaming chat) lives in
app/rag and app/db, exposed through app/routes/*.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.routes import chat, health, ingest

settings = get_settings()

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix=settings.api_v1_prefix)
app.include_router(ingest.router, prefix=settings.api_v1_prefix)
app.include_router(chat.router)  # WebSocket route: no REST prefix, lives at /ws/chat
````

## File: backend/.gitignore
````
venv/
__pycache__/
*.pyc
.env
data/chroma_db/
data/uploads/
.env.example
````

## File: backend/.python-version
````
3.11
````

## File: backend/README.md
````markdown
# Backend — Full-Stack Architecture Explorer

Phase 1 scaffold: FastAPI app skeleton, config, health check. No RAG logic yet
(that's Phase 2/3).

## Setup

```bash
cd backend

# 1. Create and activate a Python 3.11 virtual environment
python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy env template (optional — defaults work out of the box)
cp .env.example .env

# 4. Run the dev server
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000/api/v1/health` — you should see:

```json
{"status": "ok", "app": "Full-Stack Architecture Explorer"}
```

## Ingesting documents (Phase 2)

Two ways to get PDF/Markdown docs into the vector store:

**1. Via the API** (good for one-off uploads, e.g. from the future frontend):

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@/path/to/document.pdf"
```

Response:
```json
{"filename": "document.pdf", "pages_or_sections_loaded": 12, "chunks_created": 47}
```

**2. Via the CLI script** (good for bulk-loading a whole docs folder):

```bash
python -m app.scripts.ingest_cli /path/to/docs_folder
# or a single file:
python -m app.scripts.ingest_cli /path/to/document.pdf
```

Both paths run the same pipeline: load → chunk (with overlap) → embed
(local HuggingFace model) → store in ChromaDB (persisted to
`data/chroma_db/`). Supported file types: `.pdf`, `.md`, `.markdown`.

Every chunk is stored with citation metadata:
- PDF chunks: `source` (filename) + `page` (1-indexed)
- Markdown chunks: `source` (filename) + `line_start` / `line_end`

**Note:** the first ingestion will download the embedding model
(`sentence-transformers/all-MiniLM-L6-v2`, ~90MB) from Hugging Face. After
that it's cached locally and runs fully offline.

## Streaming chat (Phase 3)

WebSocket endpoint: `ws://localhost:8000/ws/chat`

**Message protocol** (full schema in `app/rag/ws_protocol.py`):

Client → Server:
```json
{"type": "query", "payload": {"text": "What does the backend use?"}}
{"type": "stop"}
```

Server → Client (in order, for one query):
```json
{"type": "citations", "payload": {"citations": [{"index": 1, "source": "architecture.md", "file_type": "markdown", "line_start": 10, "line_end": 12, "label": "architecture.md, lines 10-12", "snippet": "..."}]}}
{"type": "token", "payload": {"text": "The"}}
{"type": "token", "payload": {"text": " backend"}}
...
{"type": "done"}
```

If the client sends `stop` while a response is streaming, the server
sends `{"type": "stopped"}` instead of `done` and stops generating
immediately — it does **not** wait for the current query to finish before
checking for `stop`, since the reader and the generator run concurrently.

On any failure (empty retrieval is *not* a failure — it just means zero
citations — but a down Ollama instance, a model that isn't pulled, or an
empty query string are), the server sends:
```json
{"type": "error", "payload": {"message": "..."}}
```
and the connection stays open for the next query.

**Quick manual test** (using `websocat`, or any WS client):
```bash
websocat ws://localhost:8000/ws/chat
# paste: {"type": "query", "payload": {"text": "What does the backend use?"}}
```

## Ollama prerequisite

This project calls a **local** Ollama instance — no API keys, no cloud calls.

```bash
# Install Ollama: https://ollama.com/download
ollama pull llama3
ollama serve        # usually auto-starts; runs on http://localhost:11434
```

Verify it's reachable:

```bash
curl http://localhost:11434/api/tags
```

## Project layout (Phase 1-3)

```
backend/
├── app/
│   ├── main.py             # FastAPI app, CORS, router wiring
│   ├── core/
│   │   └── config.py       # All settings (paths, model names, chunk sizes, Ollama config)
│   ├── routes/
│   │   ├── health.py       # Liveness check
│   │   ├── ingest.py       # POST /ingest — file upload endpoint
│   │   └── chat.py         # WebSocket /ws/chat — streaming RAG chat
│   ├── rag/
│   │   ├── loaders.py       # PDF/Markdown -> LangChain Documents (+ metadata)
│   │   ├── splitter.py      # Chunking with overlap, line-number tracking for MD
│   │   ├── embeddings.py    # Local HF embedding model (cached singleton)
│   │   ├── ingestion.py     # Orchestrates load -> split -> embed -> store
│   │   ├── retriever.py     # Top-k similarity search, shaped for citations
│   │   ├── prompt.py        # Builds the context+query prompt sent to Ollama
│   │   ├── ollama_client.py # Streaming Ollama client, connection/timeout errors
│   │   └── ws_protocol.py   # Typed client<->server WebSocket message schemas
│   ├── db/
│   │   └── chroma_store.py  # ChromaDB client/collection wrapper
│   └── scripts/
│       └── ingest_cli.py    # Standalone bulk-ingestion CLI
├── data/
│   ├── chroma_db/           # Chroma persistent store (gitignored)
│   └── uploads/             # Raw ingested files (gitignored)
├── requirements.txt
├── .env.example
└── .python-version
```

## Status

- [x] Phase 1: Project skeleton & backend environment
- [x] Phase 2: Ingestion pipeline
- [x] Phase 3: Retrieval + WebSocket streaming chat
- [ ] Phase 4: Frontend foundation
- [ ] Phase 5: Streaming UI + citations
- [ ] Phase 6: Polish
````

## File: backend/requirements.txt
````
# --- Web framework ---
fastapi==0.115.0
uvicorn[standard]==0.30.6      # [standard] pulls in websockets + httptools, needed for WS support
python-multipart==0.0.9        # required for FastAPI file upload (ingestion endpoint)
websockets==13.0.1

# --- RAG orchestration ---
langchain==0.3.1
langchain-community==0.3.1
langchain-text-splitters==0.3.0

# --- Embeddings (local, no API calls) ---
sentence-transformers==3.1.1
langchain-huggingface==0.1.0

# --- Vector store (local persistence) ---
chromadb==0.5.3
langchain-chroma==0.1.4

# --- Document loaders ---
pypdf==4.3.1
unstructured==0.15.13          # markdown parsing support

# --- Ollama client ---
ollama==0.3.3
httpx==0.27.2

# --- Utilities ---
python-dotenv==1.0.1
pydantic==2.9.2
pydantic-settings==2.5.2
````

## File: backend/test_chat.py
````python
import asyncio
import json
import websockets

async def test_rag():
    uri = "ws://localhost:8000/ws/chat"
    
    # Connect to the FastAPI WebSocket
    async with websockets.connect(uri) as websocket:
        question = input("Ask a question about your documents: ")
        
        # 1. Send the query matching your ws_protocol.py schema
        request = {"type": "query", "payload": {"text": question}}
        await websocket.send(json.dumps(request))
        
        print("\nAI Thinking...\n")
        
        # 2. Listen for the streaming response
        while True:
            response_raw = await websocket.recv()
            response = json.loads(response_raw)
            
            msg_type = response.get("type")
            
            if msg_type == "citations":
                citations = response["payload"]["citations"]
                if citations:
                    print(f"[Found {len(citations)} sources in your database]")
            
            elif msg_type == "token":
                # Print tokens as they arrive without adding new lines
                print(response["payload"]["text"], end="", flush=True)
                
            elif msg_type == "done":
                print("\n\n[Stream finished]")
                break
                
            elif msg_type == "error":
                print(f"\n[ERROR]: {response['payload']['message']}")
                break

if __name__ == "__main__":
    asyncio.run(test_rag())
````

## File: frontend/public/favicon.svg
````xml
<svg xmlns="http://www.w3.org/2000/svg" width="48" height="46" fill="none" viewBox="0 0 48 46"><path fill="#863bff" d="M25.946 44.938c-.664.845-2.021.375-2.021-.698V33.937a2.26 2.26 0 0 0-2.262-2.262H10.287c-.92 0-1.456-1.04-.92-1.788l7.48-10.471c1.07-1.497 0-3.578-1.842-3.578H1.237c-.92 0-1.456-1.04-.92-1.788L10.013.474c.214-.297.556-.474.92-.474h28.894c.92 0 1.456 1.04.92 1.788l-7.48 10.471c-1.07 1.498 0 3.579 1.842 3.579h11.377c.943 0 1.473 1.088.89 1.83L25.947 44.94z" style="fill:#863bff;fill:color(display-p3 .5252 .23 1);fill-opacity:1"/><mask id="a" width="48" height="46" x="0" y="0" maskUnits="userSpaceOnUse" style="mask-type:alpha"><path fill="#000" d="M25.842 44.938c-.664.844-2.021.375-2.021-.698V33.937a2.26 2.26 0 0 0-2.262-2.262H10.183c-.92 0-1.456-1.04-.92-1.788l7.48-10.471c1.07-1.498 0-3.579-1.842-3.579H1.133c-.92 0-1.456-1.04-.92-1.787L9.91.473c.214-.297.556-.474.92-.474h28.894c.92 0 1.456 1.04.92 1.788l-7.48 10.471c-1.07 1.498 0 3.578 1.842 3.578h11.377c.943 0 1.473 1.088.89 1.832L25.843 44.94z" style="fill:#000;fill-opacity:1"/></mask><g mask="url(#a)"><g filter="url(#b)"><ellipse cx="5.508" cy="14.704" fill="#ede6ff" rx="5.508" ry="14.704" style="fill:#ede6ff;fill:color(display-p3 .9275 .9033 1);fill-opacity:1" transform="matrix(.00324 1 1 -.00324 -4.47 31.516)"/></g><g filter="url(#c)"><ellipse cx="10.399" cy="29.851" fill="#ede6ff" rx="10.399" ry="29.851" style="fill:#ede6ff;fill:color(display-p3 .9275 .9033 1);fill-opacity:1" transform="matrix(.00324 1 1 -.00324 -39.328 7.883)"/></g><g filter="url(#d)"><ellipse cx="5.508" cy="30.487" fill="#7e14ff" rx="5.508" ry="30.487" style="fill:#7e14ff;fill:color(display-p3 .4922 .0767 1);fill-opacity:1" transform="rotate(89.814 -25.913 -14.639)scale(1 -1)"/></g><g filter="url(#e)"><ellipse cx="5.508" cy="30.599" fill="#7e14ff" rx="5.508" ry="30.599" style="fill:#7e14ff;fill:color(display-p3 .4922 .0767 1);fill-opacity:1" transform="rotate(89.814 -32.644 -3.334)scale(1 -1)"/></g><g filter="url(#f)"><ellipse cx="5.508" cy="30.599" fill="#7e14ff" rx="5.508" ry="30.599" style="fill:#7e14ff;fill:color(display-p3 .4922 .0767 1);fill-opacity:1" transform="matrix(.00324 1 1 -.00324 -34.34 30.47)"/></g><g filter="url(#g)"><ellipse cx="14.072" cy="22.078" fill="#ede6ff" rx="14.072" ry="22.078" style="fill:#ede6ff;fill:color(display-p3 .9275 .9033 1);fill-opacity:1" transform="rotate(93.35 24.506 48.493)scale(-1 1)"/></g><g filter="url(#h)"><ellipse cx="3.47" cy="21.501" fill="#7e14ff" rx="3.47" ry="21.501" style="fill:#7e14ff;fill:color(display-p3 .4922 .0767 1);fill-opacity:1" transform="rotate(89.009 28.708 47.59)scale(-1 1)"/></g><g filter="url(#i)"><ellipse cx="3.47" cy="21.501" fill="#7e14ff" rx="3.47" ry="21.501" style="fill:#7e14ff;fill:color(display-p3 .4922 .0767 1);fill-opacity:1" transform="rotate(89.009 28.708 47.59)scale(-1 1)"/></g><g filter="url(#j)"><ellipse cx=".387" cy="8.972" fill="#7e14ff" rx="4.407" ry="29.108" style="fill:#7e14ff;fill:color(display-p3 .4922 .0767 1);fill-opacity:1" transform="rotate(39.51 .387 8.972)"/></g><g filter="url(#k)"><ellipse cx="47.523" cy="-6.092" fill="#7e14ff" rx="4.407" ry="29.108" style="fill:#7e14ff;fill:color(display-p3 .4922 .0767 1);fill-opacity:1" transform="rotate(37.892 47.523 -6.092)"/></g><g filter="url(#l)"><ellipse cx="41.412" cy="6.333" fill="#47bfff" rx="5.971" ry="9.665" style="fill:#47bfff;fill:color(display-p3 .2799 .748 1);fill-opacity:1" transform="rotate(37.892 41.412 6.333)"/></g><g filter="url(#m)"><ellipse cx="-1.879" cy="38.332" fill="#7e14ff" rx="4.407" ry="29.108" style="fill:#7e14ff;fill:color(display-p3 .4922 .0767 1);fill-opacity:1" transform="rotate(37.892 -1.88 38.332)"/></g><g filter="url(#n)"><ellipse cx="-1.879" cy="38.332" fill="#7e14ff" rx="4.407" ry="29.108" style="fill:#7e14ff;fill:color(display-p3 .4922 .0767 1);fill-opacity:1" transform="rotate(37.892 -1.88 38.332)"/></g><g filter="url(#o)"><ellipse cx="35.651" cy="29.907" fill="#7e14ff" rx="4.407" ry="29.108" style="fill:#7e14ff;fill:color(display-p3 .4922 .0767 1);fill-opacity:1" transform="rotate(37.892 35.651 29.907)"/></g><g filter="url(#p)"><ellipse cx="38.418" cy="32.4" fill="#47bfff" rx="5.971" ry="15.297" style="fill:#47bfff;fill:color(display-p3 .2799 .748 1);fill-opacity:1" transform="rotate(37.892 38.418 32.4)"/></g></g><defs><filter id="b" width="60.045" height="41.654" x="-19.77" y="16.149" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17158" stdDeviation="7.659"/></filter><filter id="c" width="90.34" height="51.437" x="-54.613" y="-7.533" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17158" stdDeviation="7.659"/></filter><filter id="d" width="79.355" height="29.4" x="-49.64" y="2.03" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17158" stdDeviation="4.596"/></filter><filter id="e" width="79.579" height="29.4" x="-45.045" y="20.029" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17158" stdDeviation="4.596"/></filter><filter id="f" width="79.579" height="29.4" x="-43.513" y="21.178" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17158" stdDeviation="4.596"/></filter><filter id="g" width="74.749" height="58.852" x="15.756" y="-17.901" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17158" stdDeviation="7.659"/></filter><filter id="h" width="61.377" height="25.362" x="23.548" y="2.284" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17158" stdDeviation="4.596"/></filter><filter id="i" width="61.377" height="25.362" x="23.548" y="2.284" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17158" stdDeviation="4.596"/></filter><filter id="j" width="56.045" height="63.649" x="-27.636" y="-22.853" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17158" stdDeviation="4.596"/></filter><filter id="k" width="54.814" height="64.646" x="20.116" y="-38.415" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17158" stdDeviation="4.596"/></filter><filter id="l" width="33.541" height="35.313" x="24.641" y="-11.323" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17158" stdDeviation="4.596"/></filter><filter id="m" width="54.814" height="64.646" x="-29.286" y="6.009" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17158" stdDeviation="4.596"/></filter><filter id="n" width="54.814" height="64.646" x="-29.286" y="6.009" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17158" stdDeviation="4.596"/></filter><filter id="o" width="54.814" height="64.646" x="8.244" y="-2.416" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17158" stdDeviation="4.596"/></filter><filter id="p" width="39.409" height="43.623" x="18.713" y="10.588" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17158" stdDeviation="4.596"/></filter></defs></svg>
````

## File: frontend/public/icons.svg
````xml
<svg xmlns="http://www.w3.org/2000/svg">
  <symbol id="bluesky-icon" viewBox="0 0 16 17">
    <g clip-path="url(#bluesky-clip)"><path fill="#08060d" d="M7.75 7.735c-.693-1.348-2.58-3.86-4.334-5.097-1.68-1.187-2.32-.981-2.74-.79C.188 2.065.1 2.812.1 3.251s.241 3.602.398 4.13c.52 1.744 2.367 2.333 4.07 2.145-2.495.37-4.71 1.278-1.805 4.512 3.196 3.309 4.38-.71 4.987-2.746.608 2.036 1.307 5.91 4.93 2.746 2.72-2.746.747-4.143-1.747-4.512 1.702.189 3.55-.4 4.07-2.145.156-.528.397-3.691.397-4.13s-.088-1.186-.575-1.406c-.42-.19-1.06-.395-2.741.79-1.755 1.24-3.64 3.752-4.334 5.099"/></g>
    <defs><clipPath id="bluesky-clip"><path fill="#fff" d="M.1.85h15.3v15.3H.1z"/></clipPath></defs>
  </symbol>
  <symbol id="discord-icon" viewBox="0 0 20 19">
    <path fill="#08060d" d="M16.224 3.768a14.5 14.5 0 0 0-3.67-1.153c-.158.286-.343.67-.47.976a13.5 13.5 0 0 0-4.067 0c-.128-.306-.317-.69-.476-.976A14.4 14.4 0 0 0 3.868 3.77C1.546 7.28.916 10.703 1.231 14.077a14.7 14.7 0 0 0 4.5 2.306q.545-.748.965-1.587a9.5 9.5 0 0 1-1.518-.74q.191-.14.372-.293c2.927 1.369 6.107 1.369 8.999 0q.183.152.372.294-.723.437-1.52.74.418.838.963 1.588a14.6 14.6 0 0 0 4.504-2.308c.37-3.911-.63-7.302-2.644-10.309m-9.13 8.234c-.878 0-1.599-.82-1.599-1.82 0-.998.705-1.82 1.6-1.82.894 0 1.614.82 1.599 1.82.001 1-.705 1.82-1.6 1.82m5.91 0c-.878 0-1.599-.82-1.599-1.82 0-.998.705-1.82 1.6-1.82.893 0 1.614.82 1.599 1.82 0 1-.706 1.82-1.6 1.82"/>
  </symbol>
  <symbol id="documentation-icon" viewBox="0 0 21 20">
    <path fill="none" stroke="#aa3bff" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.35" d="m15.5 13.333 1.533 1.322c.645.555.967.833.967 1.178s-.322.623-.967 1.179L15.5 18.333m-3.333-5-1.534 1.322c-.644.555-.966.833-.966 1.178s.322.623.966 1.179l1.534 1.321"/>
    <path fill="none" stroke="#aa3bff" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.35" d="M17.167 10.836v-4.32c0-1.41 0-2.117-.224-2.68-.359-.906-1.118-1.621-2.08-1.96-.599-.21-1.349-.21-2.848-.21-2.623 0-3.935 0-4.983.369-1.684.591-3.013 1.842-3.641 3.428C3 6.449 3 7.684 3 10.154v2.122c0 2.558 0 3.838.706 4.726q.306.383.713.671c.76.536 1.79.64 3.581.66"/>
    <path fill="none" stroke="#aa3bff" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.35" d="M3 10a2.78 2.78 0 0 1 2.778-2.778c.555 0 1.209.097 1.748-.047.48-.129.854-.503.982-.982.145-.54.048-1.194.048-1.749a2.78 2.78 0 0 1 2.777-2.777"/>
  </symbol>
  <symbol id="github-icon" viewBox="0 0 19 19">
    <path fill="#08060d" fill-rule="evenodd" d="M9.356 1.85C5.05 1.85 1.57 5.356 1.57 9.694a7.84 7.84 0 0 0 5.324 7.44c.387.079.528-.168.528-.376 0-.182-.013-.805-.013-1.454-2.165.467-2.616-.935-2.616-.935-.349-.91-.864-1.143-.864-1.143-.71-.48.051-.48.051-.48.787.051 1.2.805 1.2.805.695 1.194 1.817.857 2.268.649.064-.507.27-.857.49-1.052-1.728-.182-3.545-.857-3.545-3.87 0-.857.31-1.558.8-2.104-.078-.195-.349-1 .077-2.078 0 0 .657-.208 2.14.805a7.5 7.5 0 0 1 1.946-.26c.657 0 1.328.092 1.946.26 1.483-1.013 2.14-.805 2.14-.805.426 1.078.155 1.883.078 2.078.502.546.799 1.247.799 2.104 0 3.013-1.818 3.675-3.558 3.87.284.247.528.714.528 1.454 0 1.052-.012 1.896-.012 2.156 0 .208.142.455.528.377a7.84 7.84 0 0 0 5.324-7.441c.013-4.338-3.48-7.844-7.773-7.844" clip-rule="evenodd"/>
  </symbol>
  <symbol id="social-icon" viewBox="0 0 20 20">
    <path fill="none" stroke="#aa3bff" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.35" d="M12.5 6.667a4.167 4.167 0 1 0-8.334 0 4.167 4.167 0 0 0 8.334 0"/>
    <path fill="none" stroke="#aa3bff" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.35" d="M2.5 16.667a5.833 5.833 0 0 1 8.75-5.053m3.837.474.513 1.035c.07.144.257.282.414.309l.93.155c.596.1.736.536.307.965l-.723.73a.64.64 0 0 0-.152.531l.207.903c.164.715-.213.991-.84.618l-.872-.52a.63.63 0 0 0-.577 0l-.872.52c-.624.373-1.003.094-.84-.618l.207-.903a.64.64 0 0 0-.152-.532l-.723-.729c-.426-.43-.289-.864.306-.964l.93-.156a.64.64 0 0 0 .412-.31l.513-1.034c.28-.562.735-.562 1.012 0"/>
  </symbol>
  <symbol id="x-icon" viewBox="0 0 19 19">
    <path fill="#08060d" fill-rule="evenodd" d="M1.893 1.98c.052.072 1.245 1.769 2.653 3.77l2.892 4.114c.183.261.333.48.333.486s-.068.089-.152.183l-.522.593-.765.867-3.597 4.087c-.375.426-.734.834-.798.905a1 1 0 0 0-.118.148c0 .01.236.017.664.017h.663l.729-.83c.4-.457.796-.906.879-.999a692 692 0 0 0 1.794-2.038c.034-.037.301-.34.594-.675l.551-.624.345-.392a7 7 0 0 1 .34-.374c.006 0 .93 1.306 2.052 2.903l2.084 2.965.045.063h2.275c1.87 0 2.273-.003 2.266-.021-.008-.02-1.098-1.572-3.894-5.547-2.013-2.862-2.28-3.246-2.273-3.266.008-.019.282-.332 2.085-2.38l2-2.274 1.567-1.782c.022-.028-.016-.03-.65-.03h-.674l-.3.342a871 871 0 0 1-1.782 2.025c-.067.075-.405.458-.75.852a100 100 0 0 1-.803.91c-.148.172-.299.344-.99 1.127-.304.343-.32.358-.345.327-.015-.019-.904-1.282-1.976-2.808L6.365 1.85H1.8zm1.782.91 8.078 11.294c.772 1.08 1.413 1.973 1.425 1.984.016.017.241.02 1.05.017l1.03-.004-2.694-3.766L7.796 5.75 5.722 2.852l-1.039-.004-1.039-.004z" clip-rule="evenodd"/>
  </symbol>
</svg>
````

## File: frontend/src/assets/react.svg
````xml
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" class="iconify iconify--logos" width="35.93" height="32" preserveAspectRatio="xMidYMid meet" viewBox="0 0 256 228"><path fill="#00D8FF" d="M210.483 73.824a171.49 171.49 0 0 0-8.24-2.597c.465-1.9.893-3.777 1.273-5.621c6.238-30.281 2.16-54.676-11.769-62.708c-13.355-7.7-35.196.329-57.254 19.526a171.23 171.23 0 0 0-6.375 5.848a155.866 155.866 0 0 0-4.241-3.917C100.759 3.829 77.587-4.822 63.673 3.233C50.33 10.957 46.379 33.89 51.995 62.588a170.974 170.974 0 0 0 1.892 8.48c-3.28.932-6.445 1.924-9.474 2.98C17.309 83.498 0 98.307 0 113.668c0 15.865 18.582 31.778 46.812 41.427a145.52 145.52 0 0 0 6.921 2.165a167.467 167.467 0 0 0-2.01 9.138c-5.354 28.2-1.173 50.591 12.134 58.266c13.744 7.926 36.812-.22 59.273-19.855a145.567 145.567 0 0 0 5.342-4.923a168.064 168.064 0 0 0 6.92 6.314c21.758 18.722 43.246 26.282 56.54 18.586c13.731-7.949 18.194-32.003 12.4-61.268a145.016 145.016 0 0 0-1.535-6.842c1.62-.48 3.21-.974 4.76-1.488c29.348-9.723 48.443-25.443 48.443-41.52c0-15.417-17.868-30.326-45.517-39.844Zm-6.365 70.984c-1.4.463-2.836.91-4.3 1.345c-3.24-10.257-7.612-21.163-12.963-32.432c5.106-11 9.31-21.767 12.459-31.957c2.619.758 5.16 1.557 7.61 2.4c23.69 8.156 38.14 20.213 38.14 29.504c0 9.896-15.606 22.743-40.946 31.14Zm-10.514 20.834c2.562 12.94 2.927 24.64 1.23 33.787c-1.524 8.219-4.59 13.698-8.382 15.893c-8.067 4.67-25.32-1.4-43.927-17.412a156.726 156.726 0 0 1-6.437-5.87c7.214-7.889 14.423-17.06 21.459-27.246c12.376-1.098 24.068-2.894 34.671-5.345a134.17 134.17 0 0 1 1.386 6.193ZM87.276 214.515c-7.882 2.783-14.16 2.863-17.955.675c-8.075-4.657-11.432-22.636-6.853-46.752a156.923 156.923 0 0 1 1.869-8.499c10.486 2.32 22.093 3.988 34.498 4.994c7.084 9.967 14.501 19.128 21.976 27.15a134.668 134.668 0 0 1-4.877 4.492c-9.933 8.682-19.886 14.842-28.658 17.94ZM50.35 144.747c-12.483-4.267-22.792-9.812-29.858-15.863c-6.35-5.437-9.555-10.836-9.555-15.216c0-9.322 13.897-21.212 37.076-29.293c2.813-.98 5.757-1.905 8.812-2.773c3.204 10.42 7.406 21.315 12.477 32.332c-5.137 11.18-9.399 22.249-12.634 32.792a134.718 134.718 0 0 1-6.318-1.979Zm12.378-84.26c-4.811-24.587-1.616-43.134 6.425-47.789c8.564-4.958 27.502 2.111 47.463 19.835a144.318 144.318 0 0 1 3.841 3.545c-7.438 7.987-14.787 17.08-21.808 26.988c-12.04 1.116-23.565 2.908-34.161 5.309a160.342 160.342 0 0 1-1.76-7.887Zm110.427 27.268a347.8 347.8 0 0 0-7.785-12.803c8.168 1.033 15.994 2.404 23.343 4.08c-2.206 7.072-4.956 14.465-8.193 22.045a381.151 381.151 0 0 0-7.365-13.322Zm-45.032-43.861c5.044 5.465 10.096 11.566 15.065 18.186a322.04 322.04 0 0 0-30.257-.006c4.974-6.559 10.069-12.652 15.192-18.18ZM82.802 87.83a323.167 323.167 0 0 0-7.227 13.238c-3.184-7.553-5.909-14.98-8.134-22.152c7.304-1.634 15.093-2.97 23.209-3.984a321.524 321.524 0 0 0-7.848 12.897Zm8.081 65.352c-8.385-.936-16.291-2.203-23.593-3.793c2.26-7.3 5.045-14.885 8.298-22.6a321.187 321.187 0 0 0 7.257 13.246c2.594 4.48 5.28 8.868 8.038 13.147Zm37.542 31.03c-5.184-5.592-10.354-11.779-15.403-18.433c4.902.192 9.899.29 14.978.29c5.218 0 10.376-.117 15.453-.343c-4.985 6.774-10.018 12.97-15.028 18.486Zm52.198-57.817c3.422 7.8 6.306 15.345 8.596 22.52c-7.422 1.694-15.436 3.058-23.88 4.071a382.417 382.417 0 0 0 7.859-13.026a347.403 347.403 0 0 0 7.425-13.565Zm-16.898 8.101a358.557 358.557 0 0 1-12.281 19.815a329.4 329.4 0 0 1-23.444.823c-7.967 0-15.716-.248-23.178-.732a310.202 310.202 0 0 1-12.513-19.846h.001a307.41 307.41 0 0 1-10.923-20.627a310.278 310.278 0 0 1 10.89-20.637l-.001.001a307.318 307.318 0 0 1 12.413-19.761c7.613-.576 15.42-.876 23.31-.876H128c7.926 0 15.743.303 23.354.883a329.357 329.357 0 0 1 12.335 19.695a358.489 358.489 0 0 1 11.036 20.54a329.472 329.472 0 0 1-11 20.722Zm22.56-122.124c8.572 4.944 11.906 24.881 6.52 51.026c-.344 1.668-.73 3.367-1.15 5.09c-10.622-2.452-22.155-4.275-34.23-5.408c-7.034-10.017-14.323-19.124-21.64-27.008a160.789 160.789 0 0 1 5.888-5.4c18.9-16.447 36.564-22.941 44.612-18.3ZM128 90.808c12.625 0 22.86 10.235 22.86 22.86s-10.235 22.86-22.86 22.86s-22.86-10.235-22.86-22.86s10.235-22.86 22.86-22.86Z"></path></svg>
````

## File: frontend/src/assets/vite.svg
````xml
<svg xmlns="http://www.w3.org/2000/svg" width="77" height="47" fill="none" aria-labelledby="vite-logo-title" viewBox="0 0 77 47"><title id="vite-logo-title">Vite</title><style>.parenthesis{fill:#000}@media (prefers-color-scheme:dark){.parenthesis{fill:#fff}}</style><path fill="#9135ff" d="M40.151 45.71c-.663.844-2.02.374-2.02-.699V34.708a2.26 2.26 0 0 0-2.262-2.262H24.493c-.92 0-1.457-1.04-.92-1.788l7.479-10.471c1.07-1.498 0-3.578-1.842-3.578H15.443c-.92 0-1.456-1.04-.92-1.788l9.696-13.576c.213-.297.556-.474.92-.474h28.894c.92 0 1.456 1.04.92 1.788l-7.48 10.472c-1.07 1.497 0 3.578 1.842 3.578h11.376c.944 0 1.474 1.087.89 1.83L40.153 45.712z"/><mask id="a" width="48" height="47" x="14" y="0" maskUnits="userSpaceOnUse" style="mask-type:alpha"><path fill="#000" d="M40.047 45.71c-.663.843-2.02.374-2.02-.699V34.708a2.26 2.26 0 0 0-2.262-2.262H24.389c-.92 0-1.457-1.04-.92-1.788l7.479-10.472c1.07-1.497 0-3.578-1.842-3.578H15.34c-.92 0-1.456-1.04-.92-1.788l9.696-13.575c.213-.297.556-.474.92-.474H53.93c.92 0 1.456 1.04.92 1.788L47.37 13.03c-1.07 1.498 0 3.578 1.842 3.578h11.376c.944 0 1.474 1.088.89 1.831L40.049 45.712z"/></mask><g mask="url(#a)"><g filter="url(#b)"><ellipse cx="5.508" cy="14.704" fill="#eee6ff" rx="5.508" ry="14.704" transform="rotate(269.814 20.96 11.29)scale(-1 1)"/></g><g filter="url(#c)"><ellipse cx="10.399" cy="29.851" fill="#eee6ff" rx="10.399" ry="29.851" transform="rotate(89.814 -16.902 -8.275)scale(1 -1)"/></g><g filter="url(#d)"><ellipse cx="5.508" cy="30.487" fill="#8900ff" rx="5.508" ry="30.487" transform="rotate(89.814 -19.197 -7.127)scale(1 -1)"/></g><g filter="url(#e)"><ellipse cx="5.508" cy="30.599" fill="#8900ff" rx="5.508" ry="30.599" transform="rotate(89.814 -25.928 4.177)scale(1 -1)"/></g><g filter="url(#f)"><ellipse cx="5.508" cy="30.599" fill="#8900ff" rx="5.508" ry="30.599" transform="rotate(89.814 -25.738 5.52)scale(1 -1)"/></g><g filter="url(#g)"><ellipse cx="14.072" cy="22.078" fill="#eee6ff" rx="14.072" ry="22.078" transform="rotate(93.35 31.245 55.578)scale(-1 1)"/></g><g filter="url(#h)"><ellipse cx="3.47" cy="21.501" fill="#8900ff" rx="3.47" ry="21.501" transform="rotate(89.009 35.419 55.202)scale(-1 1)"/></g><g filter="url(#i)"><ellipse cx="3.47" cy="21.501" fill="#8900ff" rx="3.47" ry="21.501" transform="rotate(89.009 35.419 55.202)scale(-1 1)"/></g><g filter="url(#j)"><ellipse cx="14.592" cy="9.743" fill="#8900ff" rx="4.407" ry="29.108" transform="rotate(39.51 14.592 9.743)"/></g><g filter="url(#k)"><ellipse cx="61.728" cy="-5.321" fill="#8900ff" rx="4.407" ry="29.108" transform="rotate(37.892 61.728 -5.32)"/></g><g filter="url(#l)"><ellipse cx="55.618" cy="7.104" fill="#00c2ff" rx="5.971" ry="9.665" transform="rotate(37.892 55.618 7.104)"/></g><g filter="url(#m)"><ellipse cx="12.326" cy="39.103" fill="#8900ff" rx="4.407" ry="29.108" transform="rotate(37.892 12.326 39.103)"/></g><g filter="url(#n)"><ellipse cx="12.326" cy="39.103" fill="#8900ff" rx="4.407" ry="29.108" transform="rotate(37.892 12.326 39.103)"/></g><g filter="url(#o)"><ellipse cx="49.857" cy="30.678" fill="#8900ff" rx="4.407" ry="29.108" transform="rotate(37.892 49.857 30.678)"/></g><g filter="url(#p)"><ellipse cx="52.623" cy="33.171" fill="#00c2ff" rx="5.971" ry="15.297" transform="rotate(37.892 52.623 33.17)"/></g></g><path d="M6.919 0c-9.198 13.166-9.252 33.575 0 46.789h6.215c-9.25-13.214-9.196-33.623 0-46.789zm62.424 0h-6.215c9.198 13.166 9.252 33.575 0 46.789h6.215c9.25-13.214 9.196-33.623 0-46.789" class="parenthesis"/><defs><filter id="b" width="60.045" height="41.654" x="-5.564" y="16.92" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17286" stdDeviation="7.659"/></filter><filter id="c" width="90.34" height="51.437" x="-40.407" y="-6.762" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17286" stdDeviation="7.659"/></filter><filter id="d" width="79.355" height="29.4" x="-35.435" y="2.801" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17286" stdDeviation="4.596"/></filter><filter id="e" width="79.579" height="29.4" x="-30.84" y="20.8" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17286" stdDeviation="4.596"/></filter><filter id="f" width="79.579" height="29.4" x="-29.307" y="21.949" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17286" stdDeviation="4.596"/></filter><filter id="g" width="74.749" height="58.852" x="29.961" y="-17.13" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17286" stdDeviation="7.659"/></filter><filter id="h" width="61.377" height="25.362" x="37.754" y="3.055" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17286" stdDeviation="4.596"/></filter><filter id="i" width="61.377" height="25.362" x="37.754" y="3.055" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17286" stdDeviation="4.596"/></filter><filter id="j" width="56.045" height="63.649" x="-13.43" y="-22.082" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17286" stdDeviation="4.596"/></filter><filter id="k" width="54.814" height="64.646" x="34.321" y="-37.644" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17286" stdDeviation="4.596"/></filter><filter id="l" width="33.541" height="35.313" x="38.847" y="-10.552" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17286" stdDeviation="4.596"/></filter><filter id="m" width="54.814" height="64.646" x="-15.081" y="6.78" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17286" stdDeviation="4.596"/></filter><filter id="n" width="54.814" height="64.646" x="-15.081" y="6.78" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17286" stdDeviation="4.596"/></filter><filter id="o" width="54.814" height="64.646" x="22.45" y="-1.645" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17286" stdDeviation="4.596"/></filter><filter id="p" width="39.409" height="43.623" x="32.919" y="11.36" color-interpolation-filters="sRGB" filterUnits="userSpaceOnUse"><feFlood flood-opacity="0" result="BackgroundImageFix"/><feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape"/><feGaussianBlur result="effect1_foregroundBlur_2002_17286" stdDeviation="4.596"/></filter></defs></svg>
````

## File: frontend/src/hooks/useChatWebSocket.js
````javascript
import { useRef, useCallback } from "react";
import { useDispatch } from "react-redux";
import {
  addUserMessage,
  addCitations,
  appendToken,
  setStreamDone,
  setError,
} from "../store/chatSlice";

export function useChatWebSocket() {
  const dispatch = useDispatch();
  const wsRef = useRef(null);

  const sendMessage = useCallback(
    (text) => {
      // 1. Update UI immediately
      dispatch(addUserMessage(text));

      // 2. Connect to FastAPI
      const ws = new WebSocket("ws://localhost:8000/ws/chat");
      wsRef.current = ws;

      ws.onopen = () => {
        // 3. Send query matching your backend protocol
        ws.send(JSON.stringify({ type: "query", payload: { text } }));
      };

      // 4. Listen for streaming responses
      ws.onmessage = (event) => {
        const response = JSON.parse(event.data);

        switch (response.type) {
          case "citations":
            dispatch(addCitations(response.payload.citations));
            break;
          case "token":
            dispatch(appendToken(response.payload.text));
            break;
          case "done":
          case "stopped":
            dispatch(setStreamDone());
            ws.close();
            break;
          case "error":
            dispatch(setError(response.payload.message));
            ws.close();
            break;
          default:
            console.warn("Unknown message type:", response.type);
        }
      };

      ws.onerror = () => {
        dispatch(
          setError("WebSocket connection failed. Is the backend running?"),
        );
        dispatch(setStreamDone());
      };
    },
    [dispatch],
  );

  const stopStream = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "stop" }));
    }
  }, []);

  return { sendMessage, stopStream };
}
````

## File: frontend/src/store/chatSlice.js
````javascript
import { createSlice } from '@reduxjs/toolkit';

const initialState = {
  messages: [], // Array of { role: 'user' | 'ai', text: '', citations: [] }
  isGenerating: false,
  error: null,
};

const chatSlice = createSlice({
  name: 'chat',
  initialState,
  reducers: {
    // 1. User sends a message
    addUserMessage: (state, action) => {
      state.messages.push({ role: 'user', text: action.payload });
      // Immediately add an empty AI message placeholder to stream into
      state.messages.push({ role: 'ai', text: '', citations: [] });
      state.isGenerating = true;
      state.error = null;
    },
    // 2. Received citations from backend (happens before text streams)
    addCitations: (state, action) => {
      const lastMessage = state.messages[state.messages.length - 1];
      if (lastMessage && lastMessage.role === 'ai') {
        lastMessage.citations = action.payload;
      }
    },
    // 3. Received a text token from backend
    appendToken: (state, action) => {
      const lastMessage = state.messages[state.messages.length - 1];
      if (lastMessage && lastMessage.role === 'ai') {
        lastMessage.text += action.payload;
      }
    },
    // 4. Stream finished or stopped
    setStreamDone: (state) => {
      state.isGenerating = false;
    },
    // 5. Handle Errors
    setError: (state, action) => {
      state.error = action.payload;
      state.isGenerating = false;
    },
  },
});

export const { addUserMessage, addCitations, appendToken, setStreamDone, setError } = chatSlice.actions;
export default chatSlice.reducer;
````

## File: frontend/src/store/store.js
````javascript
import { configureStore } from '@reduxjs/toolkit';
import chatReducer from './chatSlice';

export const store = configureStore({
  reducer: {
    chat: chatReducer,
  },
});
````

## File: frontend/src/App.css
````css
.counter {
  font-size: 16px;
  padding: 5px 10px;
  border-radius: 5px;
  color: var(--accent);
  background: var(--accent-bg);
  border: 2px solid transparent;
  transition: border-color 0.3s;
  margin-bottom: 24px;

  &:hover {
    border-color: var(--accent-border);
  }
  &:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }
}

.hero {
  position: relative;

  .base,
  .framework,
  .vite {
    inset-inline: 0;
    margin: 0 auto;
  }

  .base {
    width: 170px;
    position: relative;
    z-index: 0;
  }

  .framework,
  .vite {
    position: absolute;
  }

  .framework {
    z-index: 1;
    top: 34px;
    height: 28px;
    transform: perspective(2000px) rotateZ(300deg) rotateX(44deg) rotateY(39deg)
      scale(1.4);
  }

  .vite {
    z-index: 0;
    top: 107px;
    height: 26px;
    width: auto;
    transform: perspective(2000px) rotateZ(300deg) rotateX(40deg) rotateY(39deg)
      scale(0.8);
  }
}

#center {
  display: flex;
  flex-direction: column;
  gap: 25px;
  place-content: center;
  place-items: center;
  flex-grow: 1;

  @media (max-width: 1024px) {
    padding: 32px 20px 24px;
    gap: 18px;
  }
}

#next-steps {
  display: flex;
  border-top: 1px solid var(--border);
  text-align: left;

  & > div {
    flex: 1 1 0;
    padding: 32px;
    @media (max-width: 1024px) {
      padding: 24px 20px;
    }
  }

  .icon {
    margin-bottom: 16px;
    width: 22px;
    height: 22px;
  }

  @media (max-width: 1024px) {
    flex-direction: column;
    text-align: center;
  }
}

#docs {
  border-right: 1px solid var(--border);

  @media (max-width: 1024px) {
    border-right: none;
    border-bottom: 1px solid var(--border);
  }
}

#next-steps ul {
  list-style: none;
  padding: 0;
  display: flex;
  gap: 8px;
  margin: 32px 0 0;

  .logo {
    height: 18px;
  }

  a {
    color: var(--text-h);
    font-size: 16px;
    border-radius: 6px;
    background: var(--social-bg);
    display: flex;
    padding: 6px 12px;
    align-items: center;
    gap: 8px;
    text-decoration: none;
    transition: box-shadow 0.3s;

    &:hover {
      box-shadow: var(--shadow);
    }
    .button-icon {
      height: 18px;
      width: 18px;
    }
  }

  @media (max-width: 1024px) {
    margin-top: 20px;
    flex-wrap: wrap;
    justify-content: center;

    li {
      flex: 1 1 calc(50% - 8px);
    }

    a {
      width: 100%;
      justify-content: center;
      box-sizing: border-box;
    }
  }
}

#spacer {
  height: 88px;
  border-top: 1px solid var(--border);
  @media (max-width: 1024px) {
    height: 48px;
  }
}

.ticks {
  position: relative;
  width: 100%;

  &::before,
  &::after {
    content: '';
    position: absolute;
    top: -4.5px;
    border: 5px solid transparent;
  }

  &::before {
    left: 0;
    border-left-color: var(--border);
  }
  &::after {
    right: 0;
    border-right-color: var(--border);
  }
}
````

## File: frontend/src/index.css
````css
@import "tailwindcss";
````

## File: frontend/.gitignore
````
# Logs
logs
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*
pnpm-debug.log*
lerna-debug.log*

node_modules
dist
dist-ssr
*.local

# Editor directories and files
.vscode/*
!.vscode/extensions.json
.idea
.DS_Store
*.suo
*.ntvs*
*.njsproj
*.sln
*.sw?
````

## File: frontend/eslint.config.js
````javascript
import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{js,jsx}'],
    extends: [
      js.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      globals: globals.browser,
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
  },
])
````

## File: frontend/index.html
````html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>frontend</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
````

## File: frontend/package.json
````json
{
  "name": "frontend",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "lint": "eslint .",
    "preview": "vite preview"
  },
  "dependencies": {
    "@reduxjs/toolkit": "^2.12.0",
    "@tailwindcss/vite": "^4.3.1",
    "lucide-react": "^1.22.0",
    "react": "^19.2.7",
    "react-dom": "^19.2.7",
    "react-redux": "^9.3.0"
  },
  "devDependencies": {
    "@eslint/js": "^10.0.1",
    "@types/react": "^19.2.17",
    "@types/react-dom": "^19.2.3",
    "@vitejs/plugin-react": "^6.0.2",
    "autoprefixer": "^10.5.2",
    "eslint": "^10.5.0",
    "eslint-plugin-react-hooks": "^7.1.1",
    "eslint-plugin-react-refresh": "^0.5.3",
    "globals": "^17.6.0",
    "postcss": "^8.5.16",
    "tailwindcss": "^4.3.1",
    "vite": "^8.1.0"
  }
}
````

## File: frontend/README.md
````markdown
# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.
````

## File: frontend/vite.config.js
````javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
})
````

## File: repomix-output.xml
````xml
This file is a merged representation of the entire codebase, combined into a single document by Repomix.

<file_summary>
This section contains a summary of this file.

<purpose>
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.
</purpose>

<file_format>
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  - File path as an attribute
  - Full contents of the file
</file_format>

<usage_guidelines>
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.
</usage_guidelines>

<notes>
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)
</notes>

</file_summary>

<directory_structure>
.env.example
.gitignore
.python-version
app/__init__.py
app/core/__init__.py
app/core/config.py
app/db/__init__.py
app/db/chroma_store.py
app/main.py
app/rag/__init__.py
app/rag/embeddings.py
app/rag/ingestion.py
app/rag/loaders.py
app/rag/ollama_client.py
app/rag/prompt.py
app/rag/retriever.py
app/rag/splitter.py
app/rag/ws_protocol.py
app/routes/__init__.py
app/routes/chat.py
app/routes/health.py
app/routes/ingest.py
app/scripts/__init__.py
app/scripts/ingest_cli.py
README.md
requirements.txt
</directory_structure>

<files>
This section contains the contents of the repository's files.

<file path=".env.example">
# Copy this file to `.env` and adjust as needed. All values here have
# sane defaults in app/core/config.py if you omit them.

# --- Ollama ---
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
OLLAMA_TIMEOUT_SECONDS=60

# --- Embeddings ---
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# --- Chunking ---
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# --- Retrieval ---
RETRIEVAL_TOP_K=4

# --- CORS (comma-separated list parsed by pydantic-settings) ---
CORS_ORIGINS=["http://localhost:5173","http://127.0.0.1:5173"]
</file>

<file path=".gitignore">
venv/
__pycache__/
*.pyc
.env
data/chroma_db/
data/uploads/
</file>

<file path=".python-version">
3.11
</file>

<file path="app/__init__.py">

</file>

<file path="app/core/__init__.py">

</file>

<file path="app/core/config.py">
"""
Centralized application configuration.

All tunables (model names, chunk sizes, paths, timeouts) live here so that
ingestion, retrieval, and chat logic never hardcode values. Override any
of these via a `.env` file in /backend or real environment variables.
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent  # /backend


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- App ---
    app_name: str = "Full-Stack Architecture Explorer"
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # --- Storage paths ---
    data_dir: Path = BACKEND_ROOT / "data"
    chroma_persist_dir: Path = BACKEND_ROOT / "data" / "chroma_db"
    upload_dir: Path = BACKEND_ROOT / "data" / "uploads"

    # --- Embeddings ---
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Chunking ---
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # --- Vector store / retrieval ---
    chroma_collection_name: str = "docs"
    retrieval_top_k: int = 4

    # --- Ollama / LLM ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ollama_timeout_seconds: int = 60
    llm_temperature: float = 0.2

    # --- WebSocket ---
    ws_heartbeat_seconds: int = 30


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton so we parse .env / env vars only once."""
    settings = Settings()
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return settings
</file>

<file path="app/db/__init__.py">

</file>

<file path="app/db/chroma_store.py">
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
</file>

<file path="app/main.py">
"""
FastAPI application entrypoint.

This file should stay thin: wire up middleware and routers only.
Business logic (ingestion, retrieval, streaming chat) lives in
app/rag and app/db, exposed through app/routes/*.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.routes import chat, health, ingest

settings = get_settings()

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix=settings.api_v1_prefix)
app.include_router(ingest.router, prefix=settings.api_v1_prefix)
app.include_router(chat.router)  # WebSocket route: no REST prefix, lives at /ws/chat
</file>

<file path="app/rag/__init__.py">

</file>

<file path="app/rag/embeddings.py">
"""
Embeddings.

Wraps a local HuggingFace sentence-transformers model so the rest of the
app never imports `langchain_huggingface` directly. The model is loaded
once and cached, since loading it from disk/HF cache on every request
would be slow and pointless (the weights don't change).

No API keys, no network calls at inference time (the model is downloaded
once to the local HF cache on first run, then used fully offline).
"""
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import get_settings


@lru_cache
def get_embedding_function() -> HuggingFaceEmbeddings:
    """
    Cached singleton so the embedding model is loaded into memory exactly
    once per process, regardless of how many times this is called.
    """
    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        # CPU is the safe default for a "runs anywhere locally" project.
        # If the user has a CUDA GPU available, they can override this by
        # editing model_kwargs below — left explicit here rather than
        # auto-detected so behavior is predictable across machines.
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
</file>

<file path="app/rag/ingestion.py">
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
</file>

<file path="app/rag/loaders.py">
"""
Document loading.

Responsible for turning a raw file on disk (PDF or Markdown) into a list of
LangChain `Document` objects, each carrying metadata we'll need later for
citations: source filename and a page/line locator.

This module does NOT chunk or embed — that's the job of `splitter.py` and
`embeddings.py`. Keeping load/split/embed separate makes each piece testable
and replaceable on its own.
"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown"}


class UnsupportedFileTypeError(ValueError):
    """Raised when a file extension isn't one we know how to ingest."""


def load_document(file_path: Path) -> list[Document]:
    """
    Load a single file into LangChain Documents.

    - PDF: one Document per page, with `page` metadata (1-indexed) set by
      PyPDFLoader (it's 0-indexed natively; we normalize to 1-indexed here
      since that's what a human looking at the PDF would expect).
    - Markdown: one Document per file, with `line_start`/`line_end` set to
      cover the whole file. (Per-chunk line numbers are refined later in
      the splitter, since chunk boundaries don't exist yet at load time.)

    Raises UnsupportedFileTypeError for anything else.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(file_path)
    if suffix in (".md", ".markdown"):
        return _load_markdown(file_path)

    raise UnsupportedFileTypeError(
        f"Unsupported file type '{suffix}' for {file_path.name}. "
        f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
    )


def _load_pdf(file_path: Path) -> list[Document]:
    loader = PyPDFLoader(str(file_path))
    raw_docs = loader.load()  # one Document per page, 0-indexed "page" metadata

    docs: list[Document] = []
    for doc in raw_docs:
        page_zero_indexed = doc.metadata.get("page", 0)
        docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": file_path.name,
                    "page": page_zero_indexed + 1,  # human-friendly, 1-indexed
                    "file_type": "pdf",
                },
            )
        )
    return docs


def _load_markdown(file_path: Path) -> list[Document]:
    text = file_path.read_text(encoding="utf-8")
    line_count = text.count("\n") + 1

    return [
        Document(
            page_content=text,
            metadata={
                "source": file_path.name,
                "line_start": 1,
                "line_end": line_count,
                "file_type": "markdown",
            },
        )
    ]
</file>

<file path="app/rag/ollama_client.py">
"""
Ollama client wrapper.

Isolates all direct interaction with the `ollama` Python package so that
the WebSocket route doesn't need to know about Ollama's client API,
timeouts, or connection errors directly.

Streaming is exposed as a generator yielding plain text chunks. Cancellation
is cooperative: the caller (the WebSocket route) is responsible for
breaking out of the `for` loop early when it receives a stop signal from
the client — Python generators stop producing work the moment you stop
iterating them, so no extra cancellation plumbing is needed here.
"""
import logging
from collections.abc import Iterator

import httpx
import ollama

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class OllamaConnectionError(Exception):
    """Raised when Ollama can't be reached at all (not running, wrong port)."""


class OllamaTimeoutError(Exception):
    """Raised when Ollama is reachable but a generation call exceeds the
    configured timeout."""


def stream_completion(prompt: str) -> Iterator[str]:
    """
    Stream a completion from Ollama token-chunk by token-chunk.

    Yields plain text pieces as they arrive. Raises OllamaConnectionError
    or OllamaTimeoutError on failure so the caller can send a clean error
    message to the client instead of an unhandled exception killing the
    WebSocket connection.
    """
    settings = get_settings()
    client = ollama.Client(host=settings.ollama_base_url, timeout=settings.ollama_timeout_seconds)

    try:
        stream = client.generate(
            model=settings.ollama_model,
            prompt=prompt,
            stream=True,
            options={"temperature": settings.llm_temperature},
        )
        for chunk in stream:
            text_piece = chunk.get("response", "")
            if text_piece:
                yield text_piece
            if chunk.get("done"):
                break
    except ollama.ResponseError as exc:
        # Model not pulled, bad request, etc. — Ollama returned an error
        # response rather than failing the connection outright.
        logger.error("Ollama response error: %s", exc)
        raise OllamaConnectionError(
            f"Ollama returned an error (is '{settings.ollama_model}' pulled? "
            f"Run: ollama pull {settings.ollama_model}): {exc}"
        ) from exc
    except (ConnectionError, TimeoutError, httpx.ConnectError, httpx.TimeoutException) as exc:
        logger.error("Ollama unreachable at %s: %s", settings.ollama_base_url, exc)
        raise OllamaConnectionError(
            f"Could not reach Ollama at {settings.ollama_base_url}. "
            f"Is it running? Try: ollama serve"
        ) from exc
    except Exception as exc:  # noqa: BLE001 — last-resort boundary around a 3rd-party client
        logger.exception("Unexpected error streaming from Ollama")
        raise OllamaConnectionError(f"Unexpected error from Ollama: {exc}") from exc
</file>

<file path="app/rag/prompt.py">
"""
Prompt construction.

Builds the exact string sent to Ollama. Kept as plain string templating
(no LangChain PromptTemplate/Runnable chain) so it's trivial to read,
test, and tweak the wording without learning LangChain's chain API for
something this simple — LangChain earns its keep in the loaders/splitters,
not necessarily here.

The instruction to cite using [n] markers is what lets the frontend
(Phase 5) regex-match citation markers in the streamed text and map them
back to the `citations` list sent alongside the stream.
"""
from app.rag.retriever import RetrievedChunk

SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant answering questions about software "
    "architecture documentation. Answer ONLY using the provided context. "
    "If the context does not contain the answer, say you don't have "
    "enough information in the ingested documents — do not make things up.\n\n"
    "When you state a fact from the context, cite it inline using the "
    "format [n], where n is the chunk number from the Context section "
    "below. Example: \"The backend uses FastAPI [1].\""
)


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    """
    Assemble the full prompt string: system instructions, numbered
    context chunks (each labeled with its citation), then the user query.

    If `chunks` is empty, the context section explicitly says so rather
    than being silently omitted — this nudges the model toward the
    "I don't have enough information" answer instead of guessing.
    """
    if not chunks:
        context_block = "(No relevant context was found in the ingested documents.)"
    else:
        context_block = "\n\n".join(
            f"[{i+1}] (Source: {chunk.citation_label()})\n{chunk.text}"
            for i, chunk in enumerate(chunks)
        )

    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"--- Context ---\n{context_block}\n\n"
        f"--- Question ---\n{query}\n\n"
        f"--- Answer ---\n"
    )
</file>

<file path="app/rag/retriever.py">
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
</file>

<file path="app/rag/splitter.py">
"""
Chunking.

Splits loaded Documents into overlapping chunks suitable for embedding.
Chunk size/overlap come from app.core.config so they're tunable without
touching this code.

Citation note: for PDFs, every chunk inherits the `page` metadata from its
source page Document (set in loaders.py), since PyPDFLoader already gives
us one Document per page. For Markdown, the source Document spans the
whole file, so after splitting we recompute approximate line numbers per
chunk by locating each chunk's text within the original content. This is
what lets the frontend later cite "source.md, lines 40-58" instead of just
"source.md".
"""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings


def split_documents(docs: list[Document]) -> list[Document]:
    """
    Split a list of loaded Documents into overlapping chunks.

    Each output Document's metadata always contains at least:
      - source: original filename
      - file_type: "pdf" | "markdown"
    plus either:
      - page: int (PDF)
      - line_start, line_end: int (Markdown)
    """
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # Prefer splitting on paragraph/sentence boundaries before falling
        # back to raw character cuts, so chunks stay semantically coherent.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for doc in docs:
        if doc.metadata.get("file_type") == "markdown":
            chunks.extend(_split_markdown_with_line_numbers(doc, splitter))
        else:
            # PDF (or any future page-based format): every chunk just
            # inherits the parent page's metadata as-is.
            for chunk in splitter.split_documents([doc]):
                chunks.append(chunk)

    return chunks


def _split_markdown_with_line_numbers(
    doc: Document, splitter: RecursiveCharacterTextSplitter
) -> list[Document]:
    """
    Split a whole-file Markdown Document into chunks, then back-compute each
    chunk's line range by finding its text's offset in the original content.

    This is a best-effort locator: if a chunk's exact text can't be found
    (which shouldn't normally happen since the splitter only cuts the
    original text), we fall back to the parent document's full line range
    rather than failing the whole ingestion.
    """
    full_text = doc.page_content
    sub_chunks = splitter.split_text(full_text)

    result: list[Document] = []
    search_start = 0
    for chunk_text in sub_chunks:
        offset = full_text.find(chunk_text, search_start)
        if offset == -1:
            # Overlap can occasionally make forward-only search miss a
            # match; retry from the beginning once before giving up.
            offset = full_text.find(chunk_text)

        if offset == -1:
            line_start = doc.metadata["line_start"]
            line_end = doc.metadata["line_end"]
        else:
            line_start = full_text.count("\n", 0, offset) + 1
            line_end = line_start + chunk_text.count("\n")
            search_start = offset + 1  # allow overlapping matches to advance

        result.append(
            Document(
                page_content=chunk_text,
                metadata={
                    "source": doc.metadata["source"],
                    "file_type": "markdown",
                    "line_start": line_start,
                    "line_end": line_end,
                },
            )
        )

    return result
</file>

<file path="app/rag/ws_protocol.py">
"""
WebSocket message protocol for /ws/chat.

Defining this as Pydantic models (rather than ad-hoc dicts scattered
through the route) gives us validation on incoming client messages and a
single source of truth for what the frontend should expect to receive.
This file is the "API contract" between Phase 3 (backend) and Phase 5
(frontend WebSocket hook) — keep it in sync with the frontend types when
either side changes.

--- Client -> Server ---
  {"type": "query", "payload": {"text": "..."}}
  {"type": "stop"}

--- Server -> Client ---
  {"type": "citations", "payload": {"citations": [...]}}   # sent once, before tokens
  {"type": "token", "payload": {"text": "..."}}            # sent many times
  {"type": "done"}                                          # sent once, stream finished
  {"type": "error", "payload": {"message": "..."}}         # sent on failure, stream ends
  {"type": "stopped"}                                       # sent once, if client cancelled
"""
from typing import Literal

from pydantic import BaseModel, ValidationError


class ClientQueryPayload(BaseModel):
    text: str


class ClientQueryMessage(BaseModel):
    type: Literal["query"]
    payload: ClientQueryPayload


class ClientStopMessage(BaseModel):
    type: Literal["stop"]


class CitationPayload(BaseModel):
    index: int  # matches the [n] marker used in the prompt/response text
    source: str
    file_type: str
    page: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    label: str  # pre-formatted, e.g. "architecture.pdf, p.4"
    snippet: str  # the chunk text itself, for the frontend's citation panel


class ServerCitationsMessage(BaseModel):
    type: Literal["citations"] = "citations"
    payload: dict[str, list[CitationPayload]]


class ServerTokenMessage(BaseModel):
    type: Literal["token"] = "token"
    payload: dict[str, str]


class ServerDoneMessage(BaseModel):
    type: Literal["done"] = "done"


class ServerStoppedMessage(BaseModel):
    type: Literal["stopped"] = "stopped"


class ServerErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    payload: dict[str, str]


class ProtocolError(Exception):
    """Raised when an incoming client message doesn't match any known
    message type, so the route can send a clean ServerErrorMessage instead
    of crashing the connection on a raw ValidationError/KeyError."""


def parse_client_message(raw: dict) -> ClientQueryMessage | ClientStopMessage:
    """Validate and parse a raw client WebSocket message dict into one of
    the known typed messages."""
    msg_type = raw.get("type")

    try:
        if msg_type == "query":
            return ClientQueryMessage.model_validate(raw)
        if msg_type == "stop":
            return ClientStopMessage.model_validate(raw)
    except ValidationError as exc:
        raise ProtocolError(f"Malformed '{msg_type}' message: {exc}") from exc

    raise ProtocolError(f"Unknown message type: {msg_type!r}")
</file>

<file path="app/routes/__init__.py">

</file>

<file path="app/routes/chat.py">
"""
WebSocket streaming chat endpoint.

Protocol is defined in app/rag/ws_protocol.py. Summary of the message flow
for a single query:

  client --query--> server
  server --citations--> client      (sent once, before any tokens, so the
                                      frontend can render citation badges
                                      as soon as the answer starts streaming)
  server --token--> client           (repeated, as Ollama streams)
  server --done--> client            (stream finished normally)

  OR, if the client sends `stop` mid-stream:
  server --stopped--> client         (generation cancelled, no further tokens)

  OR, on failure at any point:
  server --error--> client           (stream ends; connection stays open
                                       for the next query)

Concurrency design: a single background task (`_reader`) owns
`websocket.receive_json()` for the lifetime of the connection. Every
incoming message is pushed onto an asyncio.Queue. The main loop pulls
"query" messages off that queue and processes them one at a time; "stop"
messages instead set an asyncio.Event directly from the reader task, so a
stop sent while a query is mid-stream is observed immediately rather than
waiting for the next `receive_json()` call (which is what a naive
sequential receive-then-process loop would do — see inline comment below
for why that matters).

A "stop" with nothing in flight is harmless: it just sets an Event that
gets cleared again at the start of the next query.
"""
import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.rag.ollama_client import OllamaConnectionError, stream_completion
from app.rag.prompt import build_prompt
from app.rag.retriever import retrieve
from app.rag.ws_protocol import (
    ClientQueryMessage,
    ClientStopMessage,
    CitationPayload,
    ProtocolError,
    parse_client_message,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@router.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket) -> None:
    await websocket.accept()

    stop_event = asyncio.Event()
    query_queue: asyncio.Queue[str] = asyncio.Queue()
    disconnect_event = asyncio.Event()

    reader_task = asyncio.create_task(
        _reader(websocket, query_queue, stop_event, disconnect_event)
    )

    try:
        while not disconnect_event.is_set():
            get_query = asyncio.create_task(query_queue.get())
            wait_disconnect = asyncio.create_task(disconnect_event.wait())
            done, pending = await asyncio.wait(
                {get_query, wait_disconnect}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()

            if wait_disconnect in done:
                break

            query_text = get_query.result()
            stop_event.clear()
            await _handle_query(websocket, query_text, stop_event)
    finally:
        reader_task.cancel()


async def _reader(
    websocket: WebSocket,
    query_queue: asyncio.Queue[str],
    stop_event: asyncio.Event,
    disconnect_event: asyncio.Event,
) -> None:
    """
    Continuously reads incoming client messages for the life of the
    connection. This runs concurrently with `_handle_query` in the main
    loop, which is what lets a "stop" message take effect immediately
    instead of waiting behind whatever `_handle_query` is currently doing.

    Without this split, a single sequential loop (receive -> process ->
    receive -> process) would only ever check for a new message *after*
    the current query's stream finished — by which point "stop" arrived
    too late to do anything.
    """
    try:
        while True:
            raw = await websocket.receive_json()
            try:
                message = parse_client_message(raw)
            except ProtocolError as exc:
                await websocket.send_json(
                    {"type": "error", "payload": {"message": str(exc)}}
                )
                continue

            if isinstance(message, ClientStopMessage):
                stop_event.set()
            elif isinstance(message, ClientQueryMessage):
                await query_queue.put(message.payload.text)
    except WebSocketDisconnect:
        logger.info("Client disconnected from /ws/chat")
    finally:
        disconnect_event.set()


async def _handle_query(websocket: WebSocket, query_text: str, stop_event: asyncio.Event) -> None:
    query_text = query_text.strip()
    if not query_text:
        await websocket.send_json(
            {"type": "error", "payload": {"message": "Query text cannot be empty."}}
        )
        return

    # --- Retrieval ---
    try:
        chunks = await asyncio.to_thread(retrieve, query_text)
    except Exception as exc:  # noqa: BLE001 — boundary around the vector store
        logger.exception("Retrieval failed for query: %r", query_text)
        await websocket.send_json(
            {"type": "error", "payload": {"message": f"Retrieval failed: {exc}"}}
        )
        return

    citations = [
        CitationPayload(
            index=i + 1,
            source=chunk.source,
            file_type=chunk.file_type,
            page=chunk.page,
            line_start=chunk.line_start,
            line_end=chunk.line_end,
            label=chunk.citation_label(),
            snippet=chunk.text,
        )
        for i, chunk in enumerate(chunks)
    ]
    await websocket.send_json(
        {"type": "citations", "payload": {"citations": [c.model_dump() for c in citations]}}
    )

    # --- Generation ---
    prompt = build_prompt(query_text, chunks)

    try:
        async for text_piece in _astream(prompt):
            if stop_event.is_set():
                await websocket.send_json({"type": "stopped"})
                return
            await websocket.send_json({"type": "token", "payload": {"text": text_piece}})
    except OllamaConnectionError as exc:
        await websocket.send_json({"type": "error", "payload": {"message": str(exc)}})
        return
    except Exception as exc:  # noqa: BLE001 — boundary around the LLM call
        logger.exception("Unexpected error during generation")
        await websocket.send_json(
            {"type": "error", "payload": {"message": f"Unexpected error: {exc}"}}
        )
        return

    await websocket.send_json({"type": "done"})


async def _astream(prompt: str):
    """
    Bridge Ollama's synchronous generator (stream_completion) into an async
    generator, since FastAPI's WebSocket send calls are async but the
    `ollama` client's streaming API is blocking/sync.

    Each `next()` call runs in the default executor so the event loop isn't
    blocked while waiting on Ollama — necessary so the `_reader` task (and
    therefore stop-message handling) keeps running concurrently.
    """
    iterator = stream_completion(prompt)
    loop = asyncio.get_event_loop()

    def _next_or_sentinel():
        try:
            return next(iterator)
        except StopIteration:
            return None

    while True:
        piece = await loop.run_in_executor(None, _next_or_sentinel)
        if piece is None:
            break
        yield piece
</file>

<file path="app/routes/health.py">
"""
Simple liveness/readiness endpoint.

Kept dumb on purpose: it should never import heavy RAG/embedding/DB code,
so that `/health` stays fast and reliable even if the vector store or
Ollama is down. Phase 3 will likely add a `/health/ready` variant that
checks Ollama + Chroma connectivity.
"""
from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "app": settings.app_name,
    }
</file>

<file path="app/routes/ingest.py">
"""
Ingestion endpoint.

Accepts a file upload, saves it to disk under settings.upload_dir, then
runs it through the ingestion pipeline (load -> split -> embed -> store).

Error handling per the engineering constraints:
  - Unsupported file type -> 400, not 500
  - Any other pipeline failure (corrupt PDF, encoding issue, etc.) -> 500
    with a message, not a bare stack trace leaked to the client
"""
import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.core.config import get_settings
from app.rag.ingestion import IngestionError, ingest_file
from app.rag.loaders import SUPPORTED_EXTENSIONS, UnsupportedFileTypeError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ingestion"])


class IngestResponse(BaseModel):
    filename: str
    pages_or_sections_loaded: int
    chunks_created: int


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest(file: UploadFile) -> IngestResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
            ),
        )

    settings = get_settings()
    destination = settings.upload_dir / file.filename

    try:
        with destination.open("wb") as out_file:
            shutil.copyfileobj(file.file, out_file)
    except OSError as exc:
        logger.exception("Failed to save uploaded file %s", file.filename)
        raise HTTPException(
            status_code=500, detail=f"Could not save uploaded file: {exc}"
        ) from exc

    try:
        result = ingest_file(destination)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except IngestionError as exc:
        logger.exception("Ingestion pipeline failed for %s", file.filename)
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed for '{exc.filename}': {exc.original}",
        ) from exc

    return IngestResponse(
        filename=result.filename,
        pages_or_sections_loaded=result.pages_or_sections_loaded,
        chunks_created=result.chunks_created,
    )
</file>

<file path="app/scripts/__init__.py">

</file>

<file path="app/scripts/ingest_cli.py">
#!/usr/bin/env python3
"""
Bulk ingestion CLI.

Usage:
    python -m app.scripts.ingest_cli /path/to/docs_folder
    python -m app.scripts.ingest_cli /path/to/single_file.pdf

Walks a folder (or takes a single file), runs every supported file through
the same ingestion pipeline the `/ingest` endpoint uses, and prints a
summary. This exists because re-uploading 50 PDFs through curl/Postman one
at a time is not anyone's idea of a good afternoon.
"""
import logging
import sys
from pathlib import Path

from app.rag.ingestion import IngestionError, ingest_file
from app.rag.loaders import SUPPORTED_EXTENSIONS, UnsupportedFileTypeError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def collect_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(
            p for p in target.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    raise FileNotFoundError(f"Path does not exist: {target}")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"Usage: python -m app.scripts.ingest_cli <file_or_folder>", file=sys.stderr)
        return 1

    target = Path(argv[1]).expanduser().resolve()
    try:
        files = collect_files(target)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1

    if not files:
        logger.warning(
            "No supported files (%s) found under %s", sorted(SUPPORTED_EXTENSIONS), target
        )
        return 0

    logger.info("Found %d file(s) to ingest", len(files))

    succeeded, failed = 0, 0
    for file_path in files:
        try:
            result = ingest_file(file_path)
            logger.info(
                "  ✓ %s — %d section(s) -> %d chunk(s)",
                result.filename,
                result.pages_or_sections_loaded,
                result.chunks_created,
            )
            succeeded += 1
        except UnsupportedFileTypeError as exc:
            logger.error("  ✗ %s — %s", file_path.name, exc)
            failed += 1
        except IngestionError as exc:
            logger.error("  ✗ %s — %s", exc.filename, exc.original)
            failed += 1

    logger.info("Done. %d succeeded, %d failed.", succeeded, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
</file>

<file path="README.md">
# Backend — Full-Stack Architecture Explorer

Phase 1 scaffold: FastAPI app skeleton, config, health check. No RAG logic yet
(that's Phase 2/3).

## Setup

```bash
cd backend

# 1. Create and activate a Python 3.11 virtual environment
python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy env template (optional — defaults work out of the box)
cp .env.example .env

# 4. Run the dev server
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000/api/v1/health` — you should see:

```json
{"status": "ok", "app": "Full-Stack Architecture Explorer"}
```

## Ingesting documents (Phase 2)

Two ways to get PDF/Markdown docs into the vector store:

**1. Via the API** (good for one-off uploads, e.g. from the future frontend):

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@/path/to/document.pdf"
```

Response:
```json
{"filename": "document.pdf", "pages_or_sections_loaded": 12, "chunks_created": 47}
```

**2. Via the CLI script** (good for bulk-loading a whole docs folder):

```bash
python -m app.scripts.ingest_cli /path/to/docs_folder
# or a single file:
python -m app.scripts.ingest_cli /path/to/document.pdf
```

Both paths run the same pipeline: load → chunk (with overlap) → embed
(local HuggingFace model) → store in ChromaDB (persisted to
`data/chroma_db/`). Supported file types: `.pdf`, `.md`, `.markdown`.

Every chunk is stored with citation metadata:
- PDF chunks: `source` (filename) + `page` (1-indexed)
- Markdown chunks: `source` (filename) + `line_start` / `line_end`

**Note:** the first ingestion will download the embedding model
(`sentence-transformers/all-MiniLM-L6-v2`, ~90MB) from Hugging Face. After
that it's cached locally and runs fully offline.

## Streaming chat (Phase 3)

WebSocket endpoint: `ws://localhost:8000/ws/chat`

**Message protocol** (full schema in `app/rag/ws_protocol.py`):

Client → Server:
```json
{"type": "query", "payload": {"text": "What does the backend use?"}}
{"type": "stop"}
```

Server → Client (in order, for one query):
```json
{"type": "citations", "payload": {"citations": [{"index": 1, "source": "architecture.md", "file_type": "markdown", "line_start": 10, "line_end": 12, "label": "architecture.md, lines 10-12", "snippet": "..."}]}}
{"type": "token", "payload": {"text": "The"}}
{"type": "token", "payload": {"text": " backend"}}
...
{"type": "done"}
```

If the client sends `stop` while a response is streaming, the server
sends `{"type": "stopped"}` instead of `done` and stops generating
immediately — it does **not** wait for the current query to finish before
checking for `stop`, since the reader and the generator run concurrently.

On any failure (empty retrieval is *not* a failure — it just means zero
citations — but a down Ollama instance, a model that isn't pulled, or an
empty query string are), the server sends:
```json
{"type": "error", "payload": {"message": "..."}}
```
and the connection stays open for the next query.

**Quick manual test** (using `websocat`, or any WS client):
```bash
websocat ws://localhost:8000/ws/chat
# paste: {"type": "query", "payload": {"text": "What does the backend use?"}}
```

## Ollama prerequisite

This project calls a **local** Ollama instance — no API keys, no cloud calls.

```bash
# Install Ollama: https://ollama.com/download
ollama pull llama3
ollama serve        # usually auto-starts; runs on http://localhost:11434
```

Verify it's reachable:

```bash
curl http://localhost:11434/api/tags
```

## Project layout (Phase 1-3)

```
backend/
├── app/
│   ├── main.py             # FastAPI app, CORS, router wiring
│   ├── core/
│   │   └── config.py       # All settings (paths, model names, chunk sizes, Ollama config)
│   ├── routes/
│   │   ├── health.py       # Liveness check
│   │   ├── ingest.py       # POST /ingest — file upload endpoint
│   │   └── chat.py         # WebSocket /ws/chat — streaming RAG chat
│   ├── rag/
│   │   ├── loaders.py       # PDF/Markdown -> LangChain Documents (+ metadata)
│   │   ├── splitter.py      # Chunking with overlap, line-number tracking for MD
│   │   ├── embeddings.py    # Local HF embedding model (cached singleton)
│   │   ├── ingestion.py     # Orchestrates load -> split -> embed -> store
│   │   ├── retriever.py     # Top-k similarity search, shaped for citations
│   │   ├── prompt.py        # Builds the context+query prompt sent to Ollama
│   │   ├── ollama_client.py # Streaming Ollama client, connection/timeout errors
│   │   └── ws_protocol.py   # Typed client<->server WebSocket message schemas
│   ├── db/
│   │   └── chroma_store.py  # ChromaDB client/collection wrapper
│   └── scripts/
│       └── ingest_cli.py    # Standalone bulk-ingestion CLI
├── data/
│   ├── chroma_db/           # Chroma persistent store (gitignored)
│   └── uploads/             # Raw ingested files (gitignored)
├── requirements.txt
├── .env.example
└── .python-version
```

## Status

- [x] Phase 1: Project skeleton & backend environment
- [x] Phase 2: Ingestion pipeline
- [x] Phase 3: Retrieval + WebSocket streaming chat
- [ ] Phase 4: Frontend foundation
- [ ] Phase 5: Streaming UI + citations
- [ ] Phase 6: Polish
</file>

<file path="requirements.txt">
# --- Web framework ---
fastapi==0.115.0
uvicorn[standard]==0.30.6      # [standard] pulls in websockets + httptools, needed for WS support
python-multipart==0.0.9        # required for FastAPI file upload (ingestion endpoint)
websockets==13.0.1

# --- RAG orchestration ---
langchain==0.3.1
langchain-community==0.3.1
langchain-text-splitters==0.3.0

# --- Embeddings (local, no API calls) ---
sentence-transformers==3.1.1
langchain-huggingface==0.1.0

# --- Vector store (local persistence) ---
chromadb==0.5.5
langchain-chroma==0.1.4

# --- Document loaders ---
pypdf==4.3.1
unstructured==0.15.13          # markdown parsing support

# --- Ollama client ---
ollama==0.3.3
httpx==0.27.2

# --- Utilities ---
python-dotenv==1.0.1
pydantic==2.9.2
pydantic-settings==2.5.2
</file>

</files>
````

## File: frontend/src/App.jsx
````javascript
import React, { useState, useRef, useEffect } from 'react';
import { useSelector } from 'react-redux';
import { useChatWebSocket } from './hooks/useChatWebSocket';
import { SquareTerminal, Send, StopCircle, FileText, AlertCircle } from 'lucide-react';

export default function App() {
  const [input, setInput] = useState('');
  const { messages, isGenerating, error } = useSelector((state) => state.chat);
  const { sendMessage, stopStream } = useChatWebSocket();
  const [selectedCitation, setSelectedCitation] = useState(null);

  const messagesEndRef = useRef(null);

  // Auto-scroll to the bottom when new tokens stream in
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || isGenerating) return;
    sendMessage(input.trim());
    setInput('');
    setSelectedCitation(null); // Clear previous active citation view
  };

  return (
    <div className="flex h-screen w-screen bg-slate-900 text-slate-100 font-sans overflow-hidden">
      
      {/* LEFT SIDE: Chat Interface */}
      <div className="flex flex-col flex-1 h-full border-r border-slate-800">
        
        {/* Header */}
        <header className="flex items-center gap-3 px-6 py-4 border-b border-slate-800 bg-slate-950/50">
          <SquareTerminal className="w-6 h-6 text-emerald-400" />
          <div>
            <h1 className="text-lg font-bold tracking-tight text-slate-200">Architecture Explorer</h1>
            <p className="text-xs text-slate-400">Local RAG Node running fully offline</p>
          </div>
        </header>

        {/* Messages Feed */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-thin">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center max-w-md mx-auto space-y-3">
              <div className="p-4 bg-slate-800/50 rounded-full border border-slate-700/50 text-slate-400">
                <FileText className="w-10 h-10" />
              </div>
              <h3 className="font-semibold text-slate-300">Ask your Documentation</h3>
              <p className="text-sm text-slate-400">
                Type a query below to retrieve context from your local vector database and generate streaming answers.
              </p>
            </div>
          ) : (
            messages.map((msg, index) => (
              <div 
                key={index} 
                className={`flex w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`max-w-2xl px-4 py-3 rounded-2xl border leading-relaxed text-sm ${
                  msg.role === 'user' 
                    ? 'bg-emerald-600 border-emerald-500 text-white rounded-br-none' 
                    : 'bg-slate-950 border-slate-800 text-slate-200 rounded-bl-none'
                }`}>
                  
                  {/* Message Content */}
                  <p className="whitespace-pre-wrap">{msg.text || (isGenerating && index === messages.length - 1 ? '▋' : '')}</p>

                  {/* Citation Pills below AI response */}
                  {msg.role === 'ai' && msg.citations && msg.citations.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-slate-800/60 flex flex-wrap gap-2">
                      <span className="text-xs text-slate-500 block w-full mb-1">Sources retrieved:</span>
                      {msg.citations.map((cite) => (
                        <button
                          key={cite.index}
                          onClick={() => setSelectedCitation(cite)}
                          className={`text-xs px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 border text-slate-300 flex items-center gap-1.5 transition-colors cursor-pointer ${
                            selectedCitation?.index === cite.index ? 'border-emerald-500/80 bg-slate-800' : 'border-slate-700/60'
                          }`}
                        >
                          <span className="font-bold text-emerald-400">[{cite.index}]</span>
                          <span className="max-w-[140px] truncate">{cite.source}</span>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Error Alert Box */}
        {error && (
          <div className="mx-6 my-2 p-3 bg-rose-950/40 border border-rose-900/50 text-rose-200 rounded-xl flex items-start gap-3 text-sm">
            <AlertCircle className="w-5 h-5 text-rose-400 flex-shrink-0 mt-0.5" />
            <p>{error}</p>
          </div>
        )}

        {/* Input Dock */}
        <footer className="p-6 bg-slate-950/30 border-t border-slate-800">
          <form onSubmit={handleSubmit} className="flex gap-2 max-w-4xl mx-auto relative">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={isGenerating ? "AI is typing..." : "Ask about software architecture patterns, APIs, configurations..."}
              disabled={isGenerating}
              className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-emerald-500/80 focus:ring-1 focus:ring-emerald-500/30 disabled:opacity-50 text-slate-200 placeholder-slate-500 transition-all pr-12"
            />
            
            {isGenerating ? (
              <button
                type="button"
                onClick={stopStream}
                className="absolute right-2 top-1/2 -translate-y-1/2 p-2 rounded-lg text-rose-400 hover:bg-slate-800/80 transition-colors cursor-pointer"
                title="Stop generation"
              >
                <StopCircle className="w-5 h-5" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={!input.trim()}
                className="absolute right-2 top-1/2 -translate-y-1/2 p-2 rounded-lg text-emerald-400 hover:bg-slate-800/80 disabled:text-slate-600 disabled:hover:bg-transparent transition-colors cursor-pointer"
              >
                <Send className="w-5 h-5" />
              </button>
            )}
          </form>
        </footer>
      </div>

      {/* RIGHT SIDE: Dedicated Citation & Metadata Panel */}
      <div className={`w-96 h-full bg-slate-950 flex flex-col transition-all duration-300 ${
        selectedCitation ? 'translate-x-0 opacity-100' : 'translate-x-full w-0 opacity-0'
      }`}>
        {selectedCitation && (
          <div className="flex flex-col h-full p-6 space-y-6 overflow-y-auto">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <h2 className="font-bold text-md text-slate-200 flex items-center gap-2">
                <span className="text-emerald-400 font-extrabold text-lg">[{selectedCitation.index}]</span> 
                Document Snippet
              </h2>
              <button 
                onClick={() => setSelectedCitation(null)}
                className="text-xs px-2 py-1 bg-slate-900 border border-slate-800 rounded text-slate-400 hover:text-slate-200 cursor-pointer"
              >
                Close
              </button>
            </div>

            {/* Source Label Meta */}
            <div className="space-y-3 bg-slate-900/60 border border-slate-800 p-4 rounded-xl text-sm">
              <div className="flex justify-between">
                <span className="text-slate-500">Source File:</span>
                <span className="font-medium text-slate-300 truncate max-w-[160px]">{selectedCitation.source}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Location Reference:</span>
                <span className="font-mono text-emerald-400/90 text-xs">{selectedCitation.label}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Format:</span>
                <span className="text-xs uppercase px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700/50">
                  {selectedCitation.file_type}
                </span>
              </div>
            </div>

            {/* Ingested Chunk Snippet Text */}
            <div className="flex-1 flex flex-col space-y-2">
              <span className="text-xs text-slate-500 font-semibold tracking-wider uppercase">Ground Truth Context</span>
              <div className="flex-1 bg-slate-900 border border-slate-800 rounded-xl p-4 text-xs font-mono leading-relaxed overflow-y-auto text-slate-300 select-all whitespace-pre-wrap">
                {selectedCitation.snippet}
              </div>
            </div>
          </div>
        )}
      </div>

    </div>
  );
}
````

## File: frontend/src/main.jsx
````javascript
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { Provider } from 'react-redux'
import { store } from './store/store.js'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Provider store={store}>
      <App />
    </Provider>
  </StrictMode>,
)
````
