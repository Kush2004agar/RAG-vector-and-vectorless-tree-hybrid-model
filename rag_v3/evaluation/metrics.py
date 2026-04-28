from __future__ import annotations


def recall_at_k(predicted: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in predicted[:k] if item in relevant)
    return hits / len(relevant)


def reciprocal_rank(predicted: list[str], relevant: set[str]) -> float:
    for rank, item in enumerate(predicted, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0
