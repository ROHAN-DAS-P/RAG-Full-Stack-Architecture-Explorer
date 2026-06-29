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
