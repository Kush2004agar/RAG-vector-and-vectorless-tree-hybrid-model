from __future__ import annotations

from rag_v3.indexing.embedder import Embedder
from rag_v3.indexing.qdrant_client import QdrantStore
from rag_v3.retrieval.filter_builder import FilterBuilder
from rag_v3.retrieval.schemas import Candidate, Query


class Retriever:
    """Single-call Qdrant retriever."""

    def __init__(self, store: QdrantStore, embedder: Embedder, filter_builder: FilterBuilder) -> None:
        self.store = store
        self.embedder = embedder
        self.filter_builder = filter_builder

    def retrieve(self, query: Query, limit: int = 50) -> list[Candidate]:
        query_vector = self.embedder.embed(query.cleaned)
        query_filter = self.filter_builder.build(query)
        points = self.store.search(query_vector=query_vector, limit=limit, query_filter=query_filter)

        candidates: list[Candidate] = []
        for point in points:
            payload = point.payload or {}
            candidates.append(
                Candidate(
                    id=str(point.id),
                    text=str(payload.get("text", "")),
                    score=float(point.score or 0.0),
                    metadata={
                        "feature_ids": payload.get("feature_ids", []),
                        "pdf_name": payload.get("pdf_name", ""),
                        "section": payload.get("section", ""),
                        "version": payload.get("version", ""),
                    },
                )
            )

        return candidates
