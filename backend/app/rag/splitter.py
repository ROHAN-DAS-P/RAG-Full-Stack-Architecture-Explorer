"""
Chunking.

Splits loaded Documents into overlapping chunks suitable for embedding.
Chunk size/overlap come from app.core.config so they're tunable without
touching this code.

Citation note: for PDFs, every chunk inherits the `page` metadata from its
source page Document (set in loaders.py), since PyPDFLoader already gives
us one Document per page. For Markdown, the source Document spans the
whole file, so after splitting we recompute approximate line numbers per
chunk by locating each chunk's text within the original content. This is
what lets the frontend later cite "source.md, lines 40-58" instead of just
"source.md".
"""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings


def split_documents(docs: list[Document]) -> list[Document]:
    """
    Split a list of loaded Documents into overlapping chunks.

    Each output Document's metadata always contains at least:
      - source: original filename
      - file_type: "pdf" | "markdown"
    plus either:
      - page: int (PDF)
      - line_start, line_end: int (Markdown)
    """
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # Prefer splitting on paragraph/sentence boundaries before falling
        # back to raw character cuts, so chunks stay semantically coherent.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for doc in docs:
        if doc.metadata.get("file_type") == "markdown":
            chunks.extend(_split_markdown_with_line_numbers(doc, splitter))
        else:
            # PDF (or any future page-based format): every chunk just
            # inherits the parent page's metadata as-is.
            for chunk in splitter.split_documents([doc]):
                chunks.append(chunk)

    return chunks


def _split_markdown_with_line_numbers(
    doc: Document, splitter: RecursiveCharacterTextSplitter
) -> list[Document]:
    """
    Split a whole-file Markdown Document into chunks, then back-compute each
    chunk's line range by finding its text's offset in the original content.

    This is a best-effort locator: if a chunk's exact text can't be found
    (which shouldn't normally happen since the splitter only cuts the
    original text), we fall back to the parent document's full line range
    rather than failing the whole ingestion.
    """
    full_text = doc.page_content
    sub_chunks = splitter.split_text(full_text)

    result: list[Document] = []
    search_start = 0
    for chunk_text in sub_chunks:
        offset = full_text.find(chunk_text, search_start)
        if offset == -1:
            # Overlap can occasionally make forward-only search miss a
            # match; retry from the beginning once before giving up.
            offset = full_text.find(chunk_text)

        if offset == -1:
            line_start = doc.metadata["line_start"]
            line_end = doc.metadata["line_end"]
        else:
            line_start = full_text.count("\n", 0, offset) + 1
            line_end = line_start + chunk_text.count("\n")
            search_start = offset + 1  # allow overlapping matches to advance

        result.append(
            Document(
                page_content=chunk_text,
                metadata={
                    "source": doc.metadata["source"],
                    "file_type": "markdown",
                    "line_start": line_start,
                    "line_end": line_end,
                },
            )
        )

    return result
