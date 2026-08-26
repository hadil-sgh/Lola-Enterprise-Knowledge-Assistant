# Build-to-Learn: Enterprise Knowledge Assistant (RAG) — 6 Hour Plan

**Goal:** By the end you'll understand every stage of a production-grade RAG pipeline (not just call `.from_documents()` and hope) — and you'll have a working local app: ingest docs → hybrid retrieval → rerank → grounded, cited answers → FastAPI + chat UI.

**Philosophy:** We build most of the pipeline in *plain Python* first (no LangChain) so you actually see what's happening at each step — chunking, embedding, indexing, scoring, prompting. LangChain is a wrapper around exactly these ideas; once you've built it raw, the abstraction stops feeling like magic. Docker comes last, only if time allows.

**Stack:** Python 3.11 · `sentence-transformers` (embeddings) · `faiss-cpu` (vector index) · `rank_bm25` (keyword search) · a cross-encoder (reranking) · Ollama running a Llama 3.x model (generation) · FastAPI (serving)

---

## Hour 0 (10 min, do this before we start) — Setup

```bash
python3 -m venv rag-env && source rag-env/bin/activate
pip install sentence-transformers faiss-cpu rank_bm25 fastapi uvicorn pypdf numpy

# Local LLM via Ollama (matches the "LLaMA" in your stack)
# Install from https://ollama.com, then:
ollama pull llama3.1:8b
```

Project layout we'll grow into:
```
rag-app/
  data/              # source docs go here
  ingest.py          # chunking + embedding + indexing
  retrieve.py        # hybrid search + rerank
  generate.py         # prompt + LLM call
  app.py             # FastAPI service
  static/chat.html   # minimal chat UI
```

---

## Hour 1 — RAG Fundamentals + a Naive Baseline

**Concepts (15 min)**
- Why RAG: LLMs have frozen, generic knowledge. RAG bolts on a retrieval step so the model answers from *your* documents instead of guessing — this is what cuts hallucination.
- Two pipelines: **ingestion** (offline: docs → chunks → vectors → index) and **query** (online: question → retrieve → (rerank) → prompt LLM → cited answer).
- Why "naive RAG" (fixed-size chunks, single dense retriever, no rerank) is a fine starting point but breaks down: irrelevant chunks slip through, exact keywords get missed, answers aren't traceable to a source.

**Build (45 min)**
- Fixed-size chunker (split by character count with overlap).
- Embed chunks with `sentence-transformers` (e.g. `all-MiniLM-L6-v2`).
- Brute-force cosine similarity search (no FAISS yet — you should feel *why* an index is needed once search gets slow).
- Stuff top-3 chunks into a prompt, call Ollama, print the answer.

**Checkpoint:** you can ask a question about a doc you dropped in `data/` and get an answer, end to end. It'll be mediocre — that's the point, we improve it from here.

---

## Hour 2 — Real Document Ingestion + Semantic Chunking

**Concepts (15 min)**
- Why fixed-size chunking is bad: it slices mid-sentence, mixes topics, breaks tables. A chunk should be a coherent *unit of meaning*.
- Semantic chunking: split on natural boundaries (sentences/paragraphs), then merge adjacent sentences until an embedding-similarity drop signals a topic change (a "breakpoint"). Trade-off: smarter chunks cost more compute at ingestion time.
- Metadata matters: every chunk needs `{doc_id, source_filename, page/position, chunk_id}` — this is what makes citations possible later. Skipping this now means retrofitting it later, which is painful.

**Build (45 min)**
- PDF/txt loader (`pypdf` for PDFs).
- Sentence splitter → embed each sentence → merge into chunks using a cosine-similarity threshold between consecutive sentences.
- Attach metadata to every chunk; save chunks + metadata to disk (JSON) so ingestion and querying are decoupled.

**Checkpoint:** running `ingest.py` on a folder of PDFs produces a clean `chunks.json` with traceable metadata.

---

## Hour 3 — Embeddings & Dense Retrieval with FAISS

**Concepts (15 min)**
- What an embedding actually is: a bi-encoder maps text → fixed-length vector such that semantically similar text lands close together. Cosine similarity vs dot product, and why normalization matters.
- Why FAISS: brute-force search is O(n) per query; FAISS builds an index (we'll use a flat index — exact search, simplest to reason about — then briefly mention IVF/HNSW for scale).
- Limits of dense retrieval: great at *meaning*, bad at exact terms — product codes, names, acronyms, numbers. This motivates Hour 4.

**Build (45 min)**
- Embed all chunks, build a `faiss.IndexFlatIP` (inner product on normalized vectors = cosine similarity).
- Persist the index + a chunk-id lookup table.
- `dense_search(query, k)` function; sanity-check retrieval quality by hand on a few questions.

**Checkpoint:** dense retrieval returns genuinely relevant chunks for meaning-based questions, and visibly fails on a question containing a specific code/number/name.

---

## Hour 4 — Hybrid Retrieval (Dense + Keyword)

**Concepts (15 min)**
- BM25: a classic keyword-frequency ranking algorithm (think "smarter TF-IDF"). Excellent at exact term matches, blind to synonyms/paraphrase — the opposite failure mode of dense search.
- Fusing two rankings: Reciprocal Rank Fusion (RRF) — combine dense and BM25 result *ranks* (not raw scores, which live on incompatible scales) into one list.

**Build (45 min)**
- Build a `rank_bm25.BM25Okapi` index over tokenized chunks.
- `keyword_search(query, k)` function.
- `hybrid_search(query, k)`: run both, fuse with RRF, return top-k.

**Checkpoint:** the query that broke dense-only retrieval in Hour 3 now returns the right chunk.

---

## Hour 5 — Reranking + Grounded, Cited Generation

**Concepts (15 min)**
- Two-stage retrieval: a bi-encoder (Hour 3) is fast but approximate over the whole corpus; a **cross-encoder** reranker is slow but precise, so you use it only on the small candidate set from hybrid search (e.g. top 20 → rerank → top 5). This is the standard "retrieve wide, rerank narrow" pattern that cuts irrelevant context reaching the LLM.
- Grounded generation: prompt the LLM to answer *only* from the provided passages, cite the passage id/source for every claim, and explicitly say "I don't know" if the passages don't cover the question. This prompt discipline is a major hallucination-reduction lever, independent of retrieval quality.

**Build (45 min)**
- Load a cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) and rerank hybrid candidates.
- Write the grounding prompt template with numbered source passages and citation instructions.
- Call Ollama, parse the answer, print it alongside which sources were cited.

**Checkpoint:** answers include inline citations like `[Source 2]`, and asking something not in your docs gets an honest "not found" instead of a confident guess.

---

## Hour 6 — FastAPI Service + Chat UI (+ Docker if time allows)

**Concepts (10 min)**
- Separating ingestion (batch job) from querying (request/response API) — why you don't want to rebuild the index on every request.
- What Docker buys you here: reproducible deps, portable deployment — but it's the least conceptually new part of this project, so we treat it as a wrap-up, not core learning.

**Build (50 min)**
- `POST /ingest` — trigger ingestion on files in `data/`.
- `POST /chat` — takes a question, runs hybrid search → rerank → generate, returns `{answer, sources: [...]}`.
- Minimal `static/chat.html`: a text box, a message list, and rendered citations under each answer (plain JS + `fetch`, no framework needed).
- **If time remains:** `Dockerfile` + `docker-compose.yml` (app + a note on running Ollama alongside it).

**Checkpoint:** you can `uvicorn app:app` and chat with your documents in a browser, with every answer traceable to a source passage.

---

## Realistic expectations
Six hours is tight for all of this at a *learning* pace, not a copy-paste pace. If time runs short, Hours 1–5 are the actual RAG curriculum — Hour 6 (Docker especially) is polish, not learning, and is the first thing to cut or push to a follow-up session.

## How we'll work through it
Say "let's start hour 1" (or any hour) and I'll walk you through the concept, then we write the code for that step together, file by file, testing as we go.