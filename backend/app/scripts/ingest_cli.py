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
