from __future__ import annotations

from rag_v3.retrieval.schemas import Candidate, Query


class Reranker:
    """Simple deterministic reranker using retrieval score only."""

    def rerank(self, query: Query, candidates: list[Candidate]) -> list[Candidate]:
        del query
        return sorted(candidates, key=lambda c: (-c.score, c.id))
