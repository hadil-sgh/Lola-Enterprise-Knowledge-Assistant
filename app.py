"""
FastAPI service: serves the chat UI and exposes the RAG pipeline.

    GET  /               chat UI
    GET  /api/status     index + Ollama state
    POST /api/ingest     (re)build the index from Data/
    POST /api/chat       retrieve -> rerank -> generate, streamed as NDJSON
    GET  /doc/{name}     the source document (UI links citations to its pages)

Run:  python -X utf8 app.py   ->  http://127.0.0.1:8000
"""
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from generate import DEFAULT_MODEL, NOT_FOUND, list_models, stream_answer
from ingest import DATA_DIR, INDEX_DIR, ingest
from retrieve import MODES, Retriever, get_reranker, is_relevant

FRONTEND = Path("Frontend/chat.html")

state: dict = {"retriever": None}


def load_retriever() -> None:
    ready = (INDEX_DIR / "index.faiss").exists() and (INDEX_DIR / "chunks.json").exists()
    state["retriever"] = Retriever() if ready else None


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_retriever()
    if state["retriever"]:
        get_reranker()  # warm up so the first question is not slow
    yield


app = FastAPI(title="Lola Knowledge Assistant", lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    mode: Literal["dense", "bm25", "hybrid", "hybrid+rerank"] = "hybrid+rerank"
    top_k: int = Field(default=4, ge=1, le=8)
    model: str | None = None


def ndjson(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


def public_source(chunk: dict) -> dict:
    def r(x, n):
        return None if x is None else round(x, n)

    s = chunk["scores"]
    return {
        "chunk_id": chunk["chunk_id"],
        "source": chunk["source"],
        "page": chunk["page"],
        "text": chunk["text"],
        "ranks": chunk["ranks"],
        "scores": {"dense": r(s["dense"], 3), "bm25": r(s["bm25"], 2), "rrf": r(s["rrf"], 4), "rerank": r(s["rerank"], 3)},
    }


@app.get("/")
def index():
    return FileResponse(FRONTEND)


@app.get("/api/status")
def status():
    retriever = state["retriever"]
    return {
        "index": retriever.stats if retriever else None,
        "ollama": list_models(),
        "modes": list(MODES),
        "default_model": DEFAULT_MODEL,
    }


@app.post("/api/ingest")
def run_ingest():
    try:
        result = ingest()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    load_retriever()
    return result


@app.post("/api/chat")
def chat(req: ChatRequest):
    retriever = state["retriever"]
    if retriever is None:
        raise HTTPException(status_code=409, detail="No index yet. Build the index first.")

    def events():
        try:
            chunks = retriever.search(req.question, req.top_k, req.mode)
            yield ndjson({"type": "sources", "mode": req.mode, "sources": [public_source(c) for c in chunks]})

            if not is_relevant(chunks, req.mode):
                yield ndjson({"type": "token", "text": NOT_FOUND})
                yield ndjson({"type": "done", "gated": True})
                return

            for token in stream_answer(req.question, chunks, req.model or DEFAULT_MODEL):
                yield ndjson({"type": "token", "text": token})
            yield ndjson({"type": "done", "gated": False})
        except Exception as exc:
            yield ndjson({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.get("/doc/{filename}")
def document(filename: str):
    path = DATA_DIR / Path(filename).name  # .name blocks path traversal
    suffix = path.suffix.lower()
    if suffix not in {".pdf", ".txt"} or not path.is_file():
        raise HTTPException(status_code=404, detail="Document not found")
    media = "application/pdf" if suffix == ".pdf" else "text/plain; charset=utf-8"
    return FileResponse(path, media_type=media, content_disposition_type="inline")


if __name__ == "__main__":
    # loopback by default (the app has no login); the container sets HOST=0.0.0.0 and publishes 127.0.0.1 only
    uvicorn.run("app:app", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
