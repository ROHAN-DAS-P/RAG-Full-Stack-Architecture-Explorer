"""
Document loading.

Responsible for turning a raw file on disk (PDF or Markdown) into a list of
LangChain `Document` objects, each carrying metadata we'll need later for
citations: source filename and a page/line locator.

This module does NOT chunk or embed — that's the job of `splitter.py` and
`embeddings.py`. Keeping load/split/embed separate makes each piece testable
and replaceable on its own.
"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown"}


class UnsupportedFileTypeError(ValueError):
    """Raised when a file extension isn't one we know how to ingest."""


def load_document(file_path: Path) -> list[Document]:
    """
    Load a single file into LangChain Documents.

    - PDF: one Document per page, with `page` metadata (1-indexed) set by
      PyPDFLoader (it's 0-indexed natively; we normalize to 1-indexed here
      since that's what a human looking at the PDF would expect).
    - Markdown: one Document per file, with `line_start`/`line_end` set to
      cover the whole file. (Per-chunk line numbers are refined later in
      the splitter, since chunk boundaries don't exist yet at load time.)

    Raises UnsupportedFileTypeError for anything else.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(file_path)
    if suffix in (".md", ".markdown"):
        return _load_markdown(file_path)

    raise UnsupportedFileTypeError(
        f"Unsupported file type '{suffix}' for {file_path.name}. "
        f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
    )


def _load_pdf(file_path: Path) -> list[Document]:
    loader = PyPDFLoader(str(file_path))
    raw_docs = loader.load()  # one Document per page, 0-indexed "page" metadata

    docs: list[Document] = []
    for doc in raw_docs:
        page_zero_indexed = doc.metadata.get("page", 0)
        docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": file_path.name,
                    "page": page_zero_indexed + 1,  # human-friendly, 1-indexed
                    "file_type": "pdf",
                },
            )
        )
    return docs


def _load_markdown(file_path: Path) -> list[Document]:
    text = file_path.read_text(encoding="utf-8")
    line_count = text.count("\n") + 1

    return [
        Document(
            page_content=text,
            metadata={
                "source": file_path.name,
                "line_start": 1,
                "line_end": line_count,
                "file_type": "markdown",
            },
        )
    ]
