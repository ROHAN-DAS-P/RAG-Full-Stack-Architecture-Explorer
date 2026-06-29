"""
Prompt construction.

Builds the exact string sent to Ollama. Kept as plain string templating
(no LangChain PromptTemplate/Runnable chain) so it's trivial to read,
test, and tweak the wording without learning LangChain's chain API for
something this simple — LangChain earns its keep in the loaders/splitters,
not necessarily here.

The instruction to cite using [n] markers is what lets the frontend
(Phase 5) regex-match citation markers in the streamed text and map them
back to the `citations` list sent alongside the stream.
"""
from app.rag.retriever import RetrievedChunk

SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant answering questions about software "
    "architecture documentation. Answer ONLY using the provided context. "
    "If the context does not contain the answer, say you don't have "
    "enough information in the ingested documents — do not make things up.\n\n"
    "When you state a fact from the context, cite it inline using the "
    "format [n], where n is the chunk number from the Context section "
    "below. Example: \"The backend uses FastAPI [1].\""
)


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    """
    Assemble the full prompt string: system instructions, numbered
    context chunks (each labeled with its citation), then the user query.

    If `chunks` is empty, the context section explicitly says so rather
    than being silently omitted — this nudges the model toward the
    "I don't have enough information" answer instead of guessing.
    """
    if not chunks:
        context_block = "(No relevant context was found in the ingested documents.)"
    else:
        context_block = "\n\n".join(
            f"[{i+1}] (Source: {chunk.citation_label()})\n{chunk.text}"
            for i, chunk in enumerate(chunks)
        )

    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"--- Context ---\n{context_block}\n\n"
        f"--- Question ---\n{query}\n\n"
        f"--- Answer ---\n"
    )
