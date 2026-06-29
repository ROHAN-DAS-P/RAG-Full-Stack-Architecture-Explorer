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
