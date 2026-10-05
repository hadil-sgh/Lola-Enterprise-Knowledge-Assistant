"""
Grounded generation: retrieved chunks -> numbered-source prompt -> local Ollama LLM (streamed).

The prompt is what keeps the model honest: answer only from the sources,
cite them as [n], and admit when the document does not cover the question.
"""
import os
import sys
from collections.abc import Iterator

import ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
NOT_FOUND = "I couldn't find that in the document."

SYSTEM = f"""You answer questions about a document using ONLY the numbered sources provided.

Rules:
- Use only facts stated in the sources. Never use outside knowledge.
- After every claim, cite the source number in square brackets, e.g. [1] or [2][3].
- If the sources do not contain the answer, reply exactly: {NOT_FOUND}
- Be concise and direct."""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", (
        "Sources:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer using only the sources above and put a citation like [1] after every sentence."
    )),
])


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{i}] (page {c['page']})\n{c['text']}" for i, c in enumerate(chunks, start=1))


def stream_answer(question: str, chunks: list[dict], model: str = DEFAULT_MODEL) -> Iterator[str]:
    if not chunks:
        yield NOT_FOUND
        return
    llm = ChatOllama(
        model=model, base_url=OLLAMA_HOST, temperature=0.1, num_ctx=4096, keep_alive="10m"
    )
    chain = PROMPT | llm | StrOutputParser()
    yield from chain.stream({"context": build_context(chunks), "question": question})


def list_models() -> dict:
    """Chat-capable models installed in Ollama (embedding models are filtered out)."""
    try:
        listing = ollama.Client(host=OLLAMA_HOST, timeout=3).list()
        names = sorted(m.model for m in listing.models if "embed" not in m.model)
        return {"ok": True, "models": names, "default": DEFAULT_MODEL}
    except Exception as exc:  # Ollama not running / unreachable
        return {"ok": False, "models": [], "default": DEFAULT_MODEL, "error": str(exc)}


if __name__ == "__main__":
    from retrieve import Retriever

    sys.stdout.reconfigure(encoding="utf-8")
    question = " ".join(sys.argv[1:]) or "What is the AgentPortal platform and who is it for?"
    chunks = Retriever().search(question, top_k=4)
    print(f"Q: {question}\n")
    for i, c in enumerate(chunks, start=1):
        print(f"  [{i}] page {c['page']}  rerank={c['scores']['rerank']:.2f}")
    print("\nA: ", end="", flush=True)
    for token in stream_answer(question, chunks):
        print(token, end="", flush=True)
    print()
