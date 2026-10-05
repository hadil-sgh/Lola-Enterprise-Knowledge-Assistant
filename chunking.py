"""
Two ways to cut pages into chunks. Both keep the page metadata, so citations stay exact.

recursive  Fixed-size windows (1000 chars, 200 overlap) cut at the most natural separator that fits:
           paragraph, then line, then word. Fast and predictable, but blind to meaning.

semantic   Cut where the TOPIC changes:
             1. split each page into sentences
             2. embed every sentence (together with its neighbours, so one short sentence is not noisy)
             3. the "distance" between two consecutive sentences is 1 - cosine similarity
             4. a distance above a document-wide percentile means the topic shifted: start a new chunk
           Guard rails: never cut a chunk shorter than MIN_CHARS, always cut before MAX_CHARS.
"""
import re

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

BREAKPOINT_PERCENTILE = 90   # a distance above this percentile of ALL sentence gaps = topic change
MIN_CHARS = 300              # do not cut before a chunk has this much text
MAX_CHARS = 1500             # hard cap, so a long single-topic page cannot become one huge chunk
BUFFER = 1                   # embed each sentence together with this many neighbours on each side

STRATEGIES = ("recursive", "semantic")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9•(\[])")


def split_recursive(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )
    return splitter.split_documents(pages)


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s*\n\s*", " ", text)  # undo the hard line wraps a PDF leaves behind
    sentences: list[str] = []
    for part in (s.strip() for s in _SENTENCE_END.split(text)):
        if not part:
            continue
        if sentences and len(sentences[-1]) < 25:  # glue stray fragments ("1.", "Figure 2.") forward
            sentences[-1] += " " + part
        else:
            sentences.append(part)
    return sentences


def sentence_distances(pages: list[Document], embeddings) -> tuple[list[list[str]], list[np.ndarray]]:
    """Per page: its sentences, and the topic distance between each consecutive pair."""
    page_sentences = [split_sentences(p.page_content) for p in pages]
    windows = [
        " ".join(sents[max(0, i - BUFFER): i + BUFFER + 1])
        for sents in page_sentences
        for i in range(len(sents))
    ]
    vectors = np.asarray(embeddings.embed_documents(windows), dtype="float32")  # unit length

    distances, start = [], 0
    for sents in page_sentences:
        v = vectors[start: start + len(sents)]
        start += len(sents)
        distances.append(1.0 - np.sum(v[:-1] * v[1:], axis=1) if len(sents) > 1 else np.array([]))
    return page_sentences, distances


def group_sentences(pages, page_sentences, distances, percentile: float = BREAKPOINT_PERCENTILE) -> list[Document]:
    all_gaps = np.concatenate([d for d in distances if len(d)]) if any(len(d) for d in distances) else np.array([0.0])
    threshold = float(np.percentile(all_gaps, percentile))

    chunks: list[Document] = []
    for page, sentences, gaps in zip(pages, page_sentences, distances):
        groups: list[list[str]] = [sentences[:1]] if sentences else []
        for i in range(1, len(sentences)):
            size = sum(len(s) + 1 for s in groups[-1])
            topic_shift = gaps[i - 1] > threshold and size >= MIN_CHARS
            too_big = size + len(sentences[i]) > MAX_CHARS
            if topic_shift or too_big:
                groups.append([sentences[i]])
            else:
                groups[-1].append(sentences[i])
        # fold a short trailing fragment back into the previous chunk when it fits
        if len(groups) > 1 and sum(len(s) + 1 for s in groups[-1]) < MIN_CHARS:
            merged = groups[-2] + groups[-1]
            if sum(len(s) + 1 for s in merged) <= MAX_CHARS:
                groups[-2:] = [merged]
        chunks.extend(Document(page_content=" ".join(g), metadata=dict(page.metadata)) for g in groups)
    return chunks


def split_semantic(pages: list[Document], embeddings, percentile: float = BREAKPOINT_PERCENTILE) -> list[Document]:
    page_sentences, distances = sentence_distances(pages, embeddings)
    return group_sentences(pages, page_sentences, distances, percentile)


def chunk_pages(pages: list[Document], strategy: str, embeddings=None) -> list[Document]:
    if strategy == "recursive":
        chunks = split_recursive(pages)
    elif strategy == "semantic":
        chunks = split_semantic(pages, embeddings)
    else:
        raise ValueError(f"chunking must be one of {STRATEGIES}")
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks
