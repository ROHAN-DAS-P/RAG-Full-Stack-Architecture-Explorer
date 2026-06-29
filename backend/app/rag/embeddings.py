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
