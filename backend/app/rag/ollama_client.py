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
