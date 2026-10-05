"""
Three-level RAG evaluation.

  Level 1  Curate a test set    eval/testset.json  (questions + the pages holding the answer + reference answers)
  Level 2  Measure retrieval    Recall@K, Precision@K, MRR, nDCG@K for each retrieval strategy
  Level 3  Measure answers      LLM-as-a-judge scores accuracy, completeness and relevance (1-5)

    python -X utf8 evaluate.py check                    verify the test set against the index
    python -X utf8 evaluate.py retrieval                level 2 (fast, no LLM)
    python -X utf8 evaluate.py answers                  level 3 (generates answers, then judges them)
    python -X utf8 evaluate.py all                      levels 2 and 3

Relevance is judged at PAGE level (a retrieved chunk is relevant if it comes from a page listed
in `relevant_pages`), so the test set stays valid when you change the chunk size.
"""
import argparse
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from langchain_ollama import ChatOllama

from generate import DEFAULT_MODEL, NOT_FOUND, OLLAMA_HOST, list_models, stream_answer
from ingest import INDEX_DIR
from retrieve import MODES, Retriever, is_relevant

EVAL_DIR = Path("eval")
TESTSET = EVAL_DIR / "testset.json"
RESULTS_DIR = EVAL_DIR / "results"
KS = (1, 3, 5)             # cut-offs for Recall@K / Precision@K; nDCG uses the largest
DEPTH = 10                 # how deep to retrieve (MRR looks this far down)
JUDGE_MODEL = "llama3:latest"   # stronger than the 3B generator, and a different model: no self-grading
CRITERIA = ("accuracy", "completeness", "relevance")
REFUSAL = re.compile(r"couldn.?t find|could not find|cannot find|can.?t find", re.I)

JUDGE_SYSTEM = """You are a strict, impartial evaluator of a question-answering system.
You receive a QUESTION, a REFERENCE answer written by an expert, and a CANDIDATE answer.
Score the candidate from 1 (very poor) to 5 (excellent) on each criterion:
- accuracy: every fact in the candidate agrees with the reference; nothing is contradicted or invented.
- completeness: the candidate covers all the key points of the reference answer.
- relevance: the candidate directly answers the question without off-topic content.
Ignore citation markers such as [1]. Do not reward length.
Reply with JSON only, exactly in this shape:
{"accuracy": 1-5, "completeness": 1-5, "relevance": 1-5, "reason": "one short sentence"}"""


# ---------------------------------------------------------------- helpers
def norm(text: str) -> str:
    """Lowercase and keep only letters/digits, so PDF spacing quirks do not matter."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_testset() -> list[dict]:
    return json.loads(TESTSET.read_text(encoding="utf-8"))


def print_table(headers: list[str], rows: list[list]) -> None:
    cells = [[str(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) for i, h in enumerate(headers)]
    line = lambda r: "  ".join(c.ljust(w) for c, w in zip(r, widths))
    print(line(headers))
    print("  ".join("-" * w for w in widths))
    for r in cells:
        print(line(r))


def save_results(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------- level 1: the test set
def cmd_check(retriever: Retriever) -> bool:
    pages: dict[int, list[str]] = {}
    for chunk in retriever.chunks:
        pages.setdefault(chunk["metadata"]["page"], []).append(chunk["text"])

    testset, rows, seen, ok = load_testset(), [], set(), True
    for q in testset:
        issues = []
        if q["id"] in seen:
            issues.append("duplicate id")
        seen.add(q["id"])
        if q["answerable"]:
            if not q["relevant_pages"] or not q["reference_answer"].strip() or not q["evidence"]:
                issues.append("needs relevant_pages, reference_answer and evidence")
            missing = [p for p in q["relevant_pages"] if p not in pages]
            if missing:
                issues.append(f"pages not in index: {missing}")
            blob = norm(" ".join(t for p in q["relevant_pages"] for t in pages.get(p, [])))
            absent = [e for e in q["evidence"] if norm(e) not in blob]
            if absent:
                issues.append(f"evidence not found on those pages: {absent}")
        elif q["relevant_pages"]:
            issues.append("unanswerable question must not list relevant_pages")
        ok &= not issues
        n_chunks = sum(len(pages.get(p, [])) for p in q["relevant_pages"])
        rows.append([q["id"], q["type"], q["relevant_pages"] or "-", n_chunks or "-", "OK" if not issues else "; ".join(issues)])

    print_table(["id", "type", "relevant pages", "chunks", "status"], rows)
    answerable = sum(q["answerable"] for q in testset)
    print(f"\n{len(testset)} questions: {answerable} answerable, {len(testset) - answerable} unanswerable "
          f"-> {'test set is consistent with the index' if ok else 'PROBLEMS FOUND'}")
    return ok


# ---------------------------------------------------------------- level 2: retrieval metrics
def retrieval_scores(retrieved_pages: list[int], relevant: set[int], n_relevant_chunks: int) -> dict:
    hits = [p in relevant for p in retrieved_pages]
    scores = {}
    for k in KS:
        scores[f"recall@{k}"] = len(relevant & set(retrieved_pages[:k])) / len(relevant)
        scores[f"precision@{k}"] = sum(hits[:k]) / k
    scores["mrr"] = next((1 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0)
    kmax = max(KS)
    dcg = sum(1 / math.log2(rank + 1) for rank, hit in enumerate(hits[:kmax], 1) if hit)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(kmax, n_relevant_chunks) + 1))
    scores[f"ndcg@{kmax}"] = dcg / ideal if ideal else 0.0
    return scores


def cmd_retrieval(retriever: Retriever, modes: list[str]) -> dict:
    testset = load_testset()
    answerable = [q for q in testset if q["answerable"]]
    chunks_per_page: dict[int, int] = {}
    for c in retriever.chunks:
        chunks_per_page[c["metadata"]["page"]] = chunks_per_page.get(c["metadata"]["page"], 0) + 1

    metric_names = [f"recall@{k}" for k in KS] + [f"precision@{k}" for k in KS] + ["mrr", f"ndcg@{max(KS)}"]
    results: dict = {"modes": {}, "gate": None}
    for mode in modes:
        per_question = []
        for q in answerable:
            found = retriever.search(q["question"], DEPTH, mode)
            relevant = set(q["relevant_pages"])
            n_rel = sum(chunks_per_page.get(p, 0) for p in relevant)
            scores = retrieval_scores([c["page"] for c in found], relevant, n_rel)
            per_question.append({"id": q["id"], "retrieved_pages": [c["page"] for c in found], **scores})
        results["modes"][mode] = {
            "mean": {m: mean([r[m] for r in per_question]) for m in metric_names},
            "per_question": per_question,
        }
        print(f"  {mode:<14} done")

    print(f"\nRetrieval quality ({len(answerable)} answerable questions, depth {DEPTH})\n")
    print_table(["strategy"] + metric_names,
                [[m] + [f"{results['modes'][m]['mean'][n]:.2f}" for n in metric_names] for m in modes])

    if "hybrid+rerank" in modes:
        passed = {"answerable": 0, "unanswerable": 0}
        for q in testset:
            found = retriever.search(q["question"], DEPTH, "hybrid+rerank")
            passed["answerable" if q["answerable"] else "unanswerable"] += is_relevant(found, "hybrid+rerank")
        n_un = len(testset) - len(answerable)
        results["gate"] = {
            "answerable_passed": passed["answerable"], "answerable_total": len(answerable),
            "unanswerable_blocked": n_un - passed["unanswerable"], "unanswerable_total": n_un,
        }
        print(f"\nRelevance gate: lets through {passed['answerable']}/{len(answerable)} answerable questions, "
              f"blocks {n_un - passed['unanswerable']}/{n_un} unanswerable ones")

    print("\nRecall@K = share of the relevant pages found in the top K chunks.  "
          "Precision@K = share of the top K chunks that are relevant\n(capped when a page has fewer than K chunks).  "
          "MRR = 1/rank of the first relevant chunk.  nDCG = relevant chunks ranked high.")
    return results


# ---------------------------------------------------------------- level 3: answer quality
def generate_answers(retriever: Retriever, testset: list[dict], mode: str, top_k: int, model: str) -> list[dict]:
    rows = []
    for i, q in enumerate(testset, 1):
        start = time.time()
        chunks = retriever.search(q["question"], top_k, mode)
        gated = not is_relevant(chunks, mode)
        answer = NOT_FOUND if gated else "".join(stream_answer(q["question"], chunks, model)).strip()
        rows.append({
            "id": q["id"], "answer": answer, "gated": gated,
            "refused": bool(REFUSAL.search(answer)), "cited": bool(re.search(r"\[\d+\]", answer)),
            "pages": [c["page"] for c in chunks], "seconds": round(time.time() - start, 1),
        })
        print(f"  [{i}/{len(testset)}] {q['id']}  {rows[-1]['seconds']}s  {'refused' if rows[-1]['refused'] else 'answered'}", flush=True)
    return rows


def judge_answer(llm: ChatOllama, question: str, reference: str, answer: str) -> dict:
    prompt = f"QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{reference}\n\nCANDIDATE ANSWER:\n{answer}"
    for _ in range(2):  # one retry if the model returns malformed JSON
        try:
            data = json.loads(llm.invoke([("system", JUDGE_SYSTEM), ("human", prompt)]).content)
            scores = {c: min(5, max(1, int(data[c]))) for c in CRITERIA}
            return {**scores, "reason": str(data.get("reason", "")).strip()}
        except (ValueError, KeyError, TypeError):
            continue
    return {"error": "judge returned invalid JSON"}


def cmd_answers(retriever: Retriever, mode: str, top_k: int, model: str, judge: str, limit: int | None) -> dict:
    installed = list_models().get("models", [])
    for needed in {model, judge}:
        if needed not in installed:
            sys.exit(f"Model '{needed}' is not installed in Ollama (have: {', '.join(installed) or 'none'}).")

    testset = load_testset()[:limit] if limit else load_testset()
    by_id = {q["id"]: q for q in testset}

    print(f"Phase 1/2: generating {len(testset)} answers with {model} ({mode}, top_k={top_k})")
    rows = generate_answers(retriever, testset, mode, top_k, model)

    print(f"\nPhase 2/2: judging with {judge}")
    llm = ChatOllama(model=judge, base_url=OLLAMA_HOST, temperature=0, num_ctx=4096, format="json", keep_alive="10m")
    for i, row in enumerate(rows, 1):
        q = by_id[row["id"]]
        if not q["answerable"]:
            row["correct_refusal"] = row["refused"]
        elif row["refused"]:
            row.update({c: 1 for c in CRITERIA}, reason="System refused to answer an answerable question", false_refusal=True)
        else:
            row.update(judge_answer(llm, q["question"], q["reference_answer"], row["answer"]))
        print(f"  [{i}/{len(rows)}] {row['id']}  " + (
            f"refusal {'correct' if row['correct_refusal'] else 'MISSED'}" if not q["answerable"]
            else " ".join(f"{c[:3]}={row.get(c, '?')}" for c in CRITERIA)), flush=True)

    answerable = [r for r in rows if by_id[r["id"]]["answerable"]]
    unanswerable = [r for r in rows if not by_id[r["id"]]["answerable"]]
    judged = [r for r in answerable if "accuracy" in r]
    summary = {c: mean([r[c] for r in judged]) for c in CRITERIA}
    summary["overall"] = mean(list(summary.values()))
    summary["false_refusals"] = sum(r.get("false_refusal", False) for r in answerable)
    summary["refusal_accuracy"] = mean([r["correct_refusal"] for r in unanswerable]) if unanswerable else None
    summary["citation_rate"] = mean([r["cited"] for r in answerable if not r["refused"]])
    summary["judge_errors"] = len(answerable) - len(judged)

    print(f"\nAnswer quality ({len(answerable)} answerable questions, 1-5 scale, judge: {judge})\n")
    print_table(["id", "acc", "comp", "rel", "cited", "judge's reason"],
                [[r["id"], r.get("accuracy", "-"), r.get("completeness", "-"), r.get("relevance", "-"),
                  "yes" if r["cited"] else "no", (r.get("reason") or r.get("error", ""))[:70]] for r in answerable])
    print(f"\nMean  accuracy {summary['accuracy']:.2f}   completeness {summary['completeness']:.2f}   "
          f"relevance {summary['relevance']:.2f}   overall {summary['overall']:.2f}")
    if unanswerable:
        print(f"Refuses unanswerable questions correctly: {sum(r['correct_refusal'] for r in unanswerable)}/{len(unanswerable)}")
    print(f"False refusals on answerable questions: {summary['false_refusals']}   "
          f"Answers carrying [n] citations: {summary['citation_rate']:.0%}   Judge errors: {summary['judge_errors']}")
    return {"mode": mode, "top_k": top_k, "model": model, "judge": judge, "summary": summary, "rows": rows}


# ---------------------------------------------------------------- CLI
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Three-level RAG evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="verify the test set against the index")
    for name in ("retrieval", "answers", "all"):
        p = sub.add_parser(name)
        p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES), help="strategies to compare (retrieval)")
        p.add_argument("--mode", choices=MODES, default="hybrid+rerank", help="strategy used to answer (answers)")
        p.add_argument("--top-k", type=int, default=4)
        p.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model that writes the answers")
        p.add_argument("--judge", default=JUDGE_MODEL, help="Ollama model that grades them")
        p.add_argument("--limit", type=int, help="only the first N questions (answers)")
    for p in sub.choices.values():
        p.add_argument("--index-dir", type=Path, default=INDEX_DIR, help="which index to evaluate")
    args = parser.parse_args()

    retriever = Retriever(args.index_dir)
    print(f"Index: {args.index_dir}/  ({retriever.chunking or 'unknown'} chunking, {len(retriever.chunks)} chunks)\n")
    if args.command == "check":
        sys.exit(0 if cmd_check(retriever) else 1)

    out: dict = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "index": {"dir": str(args.index_dir), "chunking": retriever.chunking, "chunks": len(retriever.chunks)},
    }
    if args.command in ("retrieval", "all"):
        print("Level 2: measuring retrieval")
        out["retrieval"] = cmd_retrieval(retriever, args.modes)
    if args.command in ("answers", "all"):
        print("\nLevel 3: measuring answers")
        out["answers"] = cmd_answers(retriever, args.mode, args.top_k, args.model, args.judge, args.limit)
    print(f"\nSaved {save_results(args.command, out)}")


if __name__ == "__main__":
    main()
