from __future__ import annotations

from rag_v3.indexing.embedder import Embedder
from rag_v3.indexing.qdrant_client import QdrantStore
from rag_v3.retrieval.filter_builder import FilterBuilder
from rag_v3.retrieval.schemas import Candidate, Query


class Retriever:
    """Single-call Qdrant retriever with lexical signal capture."""

    def __init__(self, store: QdrantStore, embedder: Embedder, filter_builder: FilterBuilder) -> None:
        self.store = store
        self.embedder = embedder
        self.filter_builder = filter_builder

    @staticmethod
    def _lexical_score(query_tokens: list[str], text: str) -> float:
        if not query_tokens:
            return 0.0
        doc_tokens = set(text.lower().split())
        overlap = sum(1 for token in query_tokens if token in doc_tokens)
        return overlap / max(1, len(set(query_tokens)))

    def retrieve(self, query: Query, limit: int = 50) -> list[Candidate]:
        query_vector = self.embedder.embed(query.cleaned)
        query_filter = self.filter_builder.build(query)
        points = self.store.search(query_vector=query_vector, limit=limit, query_filter=query_filter)

        candidates: list[Candidate] = []
        for point in points:
            payload = point.payload or {}
            text = str(payload.get("text", ""))
            lexical = self._lexical_score(query.tokens, text)
            candidates.append(
                Candidate(
                    id=str(point.id),
                    text=text,
                    score=float(point.score or 0.0),
                    metadata={
                        "feature_ids": payload.get("feature_ids", []),
                        "pdf_name": payload.get("pdf_name", ""),
                        "section": payload.get("section", ""),
                        "version": payload.get("version", ""),
                        "lexical_score": lexical,
                    },
                )
            )

        return candidates
