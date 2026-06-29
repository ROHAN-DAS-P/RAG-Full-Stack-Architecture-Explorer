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
