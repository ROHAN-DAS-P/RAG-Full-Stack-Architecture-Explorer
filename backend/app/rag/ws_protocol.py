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
