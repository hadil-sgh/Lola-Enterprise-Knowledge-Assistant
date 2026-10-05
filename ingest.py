"""
Ingestion pipeline (offline / batch):
    Data/*.pdf|*.txt -> pages -> chunks (+metadata) -> embeddings -> FAISS index

Run once, or whenever the documents change -- never per user question.
"""
import json
import re
import sys
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("Data")
INDEX_DIR = Path("index_store")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
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


def split_pages(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )
    chunks = splitter.split_documents(pages)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def ingest(data_dir: Path = DATA_DIR, index_dir: Path = INDEX_DIR) -> dict:
    start = time.time()
    pages, skipped = load_pages(data_dir)
    if not pages:
        raise RuntimeError(f"No usable .pdf/.txt content found in {data_dir}/")
    chunks = split_pages(pages)

    store = FAISS.from_documents(
        chunks, get_embeddings(), distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
    )
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
    return {
        "files": sorted({p.metadata["source"] for p in pages}),
        "pages_indexed": len(pages),
        "pages_skipped": skipped,
        "chunks": len(chunks),
        "seconds": round(time.time() - start, 1),
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    pages, skipped = load_pages()
    chunks = split_pages(pages)
    print(f"Loaded {len(pages)} pages ({skipped} skipped as blank / table-of-contents)")
    print(f"Divided into {len(chunks)} chunks")
    print(f"First chunk:\n\n{chunks[0]}\n")
    print("Embedding + indexing ...")
    print(ingest())
