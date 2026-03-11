import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chunk_tree_retriever import retrieve_relevant_chunks


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def precision_at_k(predicted: List[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = predicted[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(top_k)


def recall_at_k(predicted: List[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in predicted[:k] if item in relevant)
    return hits / len(relevant)


def reciprocal_rank(predicted: List[str], relevant: set[str], k: int) -> float:
    for rank, item in enumerate(predicted[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(predicted: List[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for idx, item in enumerate(predicted[:k], start=1):
        rel = 1.0 if item in relevant else 0.0
        dcg += rel / math.log2(idx + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def _extract_predictions(question: str, level: str) -> List[str]:
    retrieval = retrieve_relevant_chunks(question)
    chunks = retrieval.get("ranked_chunks", [])
    if level == "pdf":
        return _dedupe_preserve_order([chunk.get("pdf_name", "") for chunk in chunks])
    return _dedupe_preserve_order([str(chunk.get("chunk_id", "")) for chunk in chunks])


def evaluate_query(item: Dict, k: int) -> Dict:
    question = item.get("question", "").strip()
    if not question:
        return {}

    relevant_chunk_ids = {str(v) for v in item.get("relevant_chunk_ids", []) if str(v).strip()}
    relevant_pdfs = {str(v) for v in item.get("relevant_pdfs", []) if str(v).strip()}

    if relevant_chunk_ids:
        level = "chunk"
        relevant = relevant_chunk_ids
    elif relevant_pdfs:
        level = "pdf"
        relevant = relevant_pdfs
    else:
        return {}

    predicted = _extract_predictions(question, level=level)
    return {
        "question": question,
        "evaluation_level": level,
        "precision@k": precision_at_k(predicted, relevant, k),
        "recall@k": recall_at_k(predicted, relevant, k),
        "mrr": reciprocal_rank(predicted, relevant, k),
        "ndcg@k": ndcg_at_k(predicted, relevant, k),
    }


def _avg(rows: List[Dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(row.get(key, 0.0) for row in rows) / len(rows)


def run_benchmark(benchmark_path: Path, k: int) -> Dict:
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    results = []
    for item in benchmark:
        row = evaluate_query(item, k=k)
        if row:
            results.append(row)

    return {
        "k": k,
        "queries_evaluated": len(results),
        "precision@k": _avg(results, "precision@k"),
        "recall@k": _avg(results, "recall@k"),
        "mrr": _avg(results, "mrr"),
        "ndcg@k": _avg(results, "ndcg@k"),
        "per_query": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality using benchmark questions.")
    parser.add_argument(
        "--benchmark",
        type=str,
        default=str(Path(__file__).resolve().parent / "benchmark_queries.json"),
        help="Path to benchmark JSON file.",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="Top-K cutoff for retrieval metrics.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output JSON file path for metrics.",
    )
    args = parser.parse_args()

    report = run_benchmark(Path(args.benchmark), k=args.k)
    print(json.dumps(report, indent=2))

    if args.output.strip():
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Saved report to {out_path}")
