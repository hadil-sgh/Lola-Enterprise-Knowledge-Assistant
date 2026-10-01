"""
Ingestion pipeline: Data/*.txt|*.pdf  ->  chunks (with metadata)  ->  embeddings  ->  FAISS index.

This is an offline/batch step. It runs once (or whenever documents change),
not on every user query -- that's the whole point of building an index.
"""
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("Data")
INDEX_DIR = Path("index_store")
CHUNK_SIZE = 800        # target characters per chunk
CHUNK_OVERLAP = 150     # characters carried from the end of one chunk into the next
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_documents(data_dir: Path = DATA_DIR):
    """Read every .txt/.pdf in data_dir. Returns list of (filename, full_text)."""
    docs = []
    for path in sorted(data_dir.glob("*")):
        if path.suffix.lower() == ".txt":
            docs.append((path.name, path.read_text(encoding="utf-8")))
        elif path.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            docs.append((path.name, text))
    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Pack whole paragraphs into ~chunk_size windows, carrying `overlap`
    characters of context from one chunk into the next."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > chunk_size:
            chunks.append(current)
            current = current[-overlap:] + "\n\n" + para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def build_chunks(data_dir: Path = DATA_DIR):
    """Turn every document in data_dir into chunk dicts with traceable metadata."""
    all_chunks = []
    for filename, text in load_documents(data_dir):
        for i, chunk in enumerate(chunk_text(text)):
            all_chunks.append({
                "chunk_id": f"{filename}::{i}",
                "doc_id": filename,
                "source_filename": filename,
                "position": i,
                "text": chunk,
            })
    return all_chunks


def embed_and_index(chunks: list[dict], index_dir: Path = INDEX_DIR):
    """Embed each chunk and build a flat (exact-search) FAISS index over the vectors."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]

    # normalize_embeddings=True makes inner product == cosine similarity,
    # which is what IndexFlatIP computes.
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_dir.mkdir(exist_ok=True)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    with open(index_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    return index


if __name__ == "__main__":
    chunks = build_chunks()
    print(f"Built {len(chunks)} chunks from {len(list(DATA_DIR.glob('*')))} file(s) in {DATA_DIR}/")
    embed_and_index(chunks)
    print(f"Embedded and indexed {len(chunks)} chunks -> {INDEX_DIR}/faiss.index + {INDEX_DIR}/chunks.json")
