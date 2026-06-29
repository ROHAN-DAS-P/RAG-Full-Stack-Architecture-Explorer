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
