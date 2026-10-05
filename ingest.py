"""
Ingestion pipeline (offline / batch):
    Data/*.pdf|*.txt -> pages -> chunks (+metadata) -> embeddings -> FAISS index

Run once, or whenever the documents change -- never per user question.

    python -X utf8 ingest.py                                   # recursive chunks -> index_store/
    python -X utf8 ingest.py --chunking semantic --out index_semantic

Environment: CHUNKING (recursive | semantic) and INDEX_DIR set the defaults used by the API.
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from chunking import STRATEGIES, chunk_pages

DATA_DIR = Path("Data")
INDEX_DIR = Path(os.getenv("INDEX_DIR", "index_store"))
CHUNKING = os.getenv("CHUNKING", "recursive")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# "Chapter 2 . . . . . . . . 17" style dotted leaders found in tables of contents
LEADERS = re.compile(r"(?:\s?\.){5,}")


def get_embeddings() -> HuggingFaceEmbeddings:
    # Unit-length vectors make inner product identical to cosine similarity.
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def load_pages(data_dir: Path = DATA_DIR) -> tuple[list[Document], int]:
    """One Document per PDF page / text file. Returns (pages, skipped_count)."""
    pages: list[Document] = []
    skipped = 0
    for path in sorted(data_dir.glob("*")):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        elif suffix == ".txt":
            loaded = TextLoader(str(path), encoding="utf-8").load()
        else:
            continue
        for doc in loaded:
            text = doc.page_content
            # skip blank pages and TOC / list-of-figures pages: they only add noise
            if len(text.strip()) < 50 or len(LEADERS.findall(text)) >= 8:
                skipped += 1
                continue
            page_no = int(doc.metadata.get("page", 0)) + 1  # 1-based for humans
            doc.metadata = {"source": path.name, "page": page_no}  # drop PDF producer/creator noise
            pages.append(doc)
    return pages, skipped


def chunk_stats(chunks: list[Document]) -> dict:
    sizes = [len(c.page_content) for c in chunks]
    return {
        "chunks": len(chunks),
        "chars_mean": round(statistics.mean(sizes)),
        "chars_median": round(statistics.median(sizes)),
        "chars_min": min(sizes),
        "chars_max": max(sizes),
    }


def ingest(data_dir: Path = DATA_DIR, index_dir: Path = INDEX_DIR, chunking: str = CHUNKING) -> dict:
    if chunking not in STRATEGIES:
        raise RuntimeError(f"chunking must be one of {STRATEGIES}, got '{chunking}'")
    start = time.time()
    pages, skipped = load_pages(data_dir)
    if not pages:
        raise RuntimeError(f"No usable .pdf/.txt content found in {data_dir}/")

    embeddings = get_embeddings()  # loaded once: semantic chunking and indexing both need it
    chunks = chunk_pages(pages, chunking, embeddings)

    store = FAISS.from_documents(chunks, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    index_dir.mkdir(exist_ok=True)
    store.save_local(str(index_dir))

    # plain-JSON copy: used to build the BM25 keyword index, and easy to inspect by hand
    (index_dir / "chunks.json").write_text(
        json.dumps(
            [{"text": c.page_content, "metadata": c.metadata} for c in chunks],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    result = {
        "files": sorted({p.metadata["source"] for p in pages}),
        "pages_indexed": len(pages),
        "pages_skipped": skipped,
        "chunking": chunking,
        **chunk_stats(chunks),
        "seconds": round(time.time() - start, 1),
    }
    (index_dir / "index_meta.json").write_text(
        json.dumps({**result, "created": datetime.now().isoformat(timespec="seconds")}, indent=2),
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Build the FAISS index from Data/")
    parser.add_argument("--chunking", choices=STRATEGIES, default=CHUNKING)
    parser.add_argument("--out", type=Path, default=INDEX_DIR, help="index directory")
    args = parser.parse_args()
    print(f"Embedding + indexing with {args.chunking} chunking -> {args.out}/ ...")
    for key, value in ingest(index_dir=args.out, chunking=args.chunking).items():
        print(f"  {key:<13} {value}")
