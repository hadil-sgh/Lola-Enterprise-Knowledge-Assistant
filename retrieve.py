"""
Query-time retrieval.

    question -> dense search (FAISS, meaning)  \
                                                >-- fuse with RRF --> cross-encoder rerank --> top_k chunks
             -> keyword search (BM25, exact words) /

Every mode is exposed so the UI can show how each retriever behaves on the same question.
"""
import json
import math
import re
import sys
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from ingest import INDEX_DIR, get_embeddings

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATES = 20   # retrieve wide ...
RRF_K = 60        # standard Reciprocal Rank Fusion damping constant
MODES = ("dense", "bm25", "hybrid", "hybrid+rerank")  # ... then narrow down to top_k

_reranker: CrossEncoder | None = None


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankers of 1 / (k + rank).
    Works on rank positions, so incomparable raw scores (cosine vs BM25) never get mixed."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


class Retriever:
    def __init__(self, index_dir: Path = INDEX_DIR):
        self.store = FAISS.load_local(
            str(index_dir),
            get_embeddings(),
            allow_dangerous_deserialization=True,  # we wrote this pickle ourselves in ingest.py
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )
        # chunk_id == list position, which is also the BM25 corpus position
        self.chunks: list[dict] = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    @property
    def stats(self) -> dict:
        return {
            "chunks": len(self.chunks),
            "files": sorted({c["metadata"]["source"] for c in self.chunks}),
        }

    def _dense(self, query: str, n: int) -> list[tuple[int, float]]:
        hits = self.store.similarity_search_with_score(query, k=n)
        return [(int(doc.metadata["chunk_id"]), float(score)) for doc, score in hits]

    def _bm25(self, query: str, n: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = scores.argsort()[::-1][:n]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def search(self, query: str, top_k: int = 5, mode: str = "hybrid+rerank") -> list[dict]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")

        dense_hits = self._dense(query, CANDIDATES) if mode != "bm25" else []
        bm25_hits = self._bm25(query, CANDIDATES) if mode != "dense" else []
        dense_score, bm25_score = dict(dense_hits), dict(bm25_hits)
        dense_rank = {cid: r for r, (cid, _) in enumerate(dense_hits, start=1)}
        bm25_rank = {cid: r for r, (cid, _) in enumerate(bm25_hits, start=1)}

        rrf_score: dict[int, float] = {}
        if mode == "dense":
            order = [cid for cid, _ in dense_hits]
        elif mode == "bm25":
            order = [cid for cid, _ in bm25_hits]
        else:
            fused = rrf([[c for c, _ in dense_hits], [c for c, _ in bm25_hits]])
            rrf_score = dict(fused)
            order = [cid for cid, _ in fused]

        rerank_score: dict[int, float] = {}
        if mode == "hybrid+rerank" and order:
            pool = order[:CANDIDATES]
            logits = get_reranker().predict([(query, self.chunks[c]["text"]) for c in pool])
            ranked = sorted(zip(pool, logits), key=lambda x: x[1], reverse=True)
            rerank_score = {c: 1 / (1 + math.exp(-float(s))) for c, s in ranked}  # logit -> 0..1
            order = [c for c, _ in ranked]

        results = []
        for cid in order[:top_k]:
            chunk = self.chunks[cid]
            results.append({
                "chunk_id": cid,
                "source": chunk["metadata"]["source"],
                "page": chunk["metadata"]["page"],
                "text": chunk["text"],
                "ranks": {"dense": dense_rank.get(cid), "bm25": bm25_rank.get(cid)},
                "scores": {
                    "dense": dense_score.get(cid),
                    "bm25": bm25_score.get(cid),
                    "rrf": rrf_score.get(cid),
                    "rerank": rerank_score.get(cid),
                },
            })
        return results


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    question = " ".join(sys.argv[1:]) or "What technologies were used to build the platform?"
    retriever = Retriever()
    print(f"Q: {question}\n")
    for mode in MODES:
        print(f"== {mode} ==")
        for r in retriever.search(question, top_k=3, mode=mode):
            snippet = r["text"][:90].replace("\n", " ")
            print(f"  p.{r['page']:<3} chunk {r['chunk_id']:<4} ranks={r['ranks']}  {snippet}...")
        print()
