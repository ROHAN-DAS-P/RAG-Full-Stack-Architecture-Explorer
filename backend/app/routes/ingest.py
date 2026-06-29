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
