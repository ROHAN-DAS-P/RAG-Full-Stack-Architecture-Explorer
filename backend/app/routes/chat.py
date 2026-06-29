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
