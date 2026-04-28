from __future__ import annotations

import os

from sentence_transformers import CrossEncoder

from rag_v3.retrieval.schemas import Candidate, Query


class Reranker:
    """Cross-encoder reranker using only relevance scores from the cross-encoder."""

    def __init__(self, model_name: str | None = None, batch_size: int = 16) -> None:
        self.model_name = model_name or os.environ.get("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.batch_size = int(os.environ.get("RERANKER_BATCH_SIZE", batch_size))
        try:
            self.model = CrossEncoder(self.model_name)
        except Exception:
            self.model = None

    @staticmethod
    def _fallback_score(query: str, text: str) -> float:
        q_tokens = set(query.lower().split())
        if not q_tokens:
            return 0.0
        t_tokens = set(text.lower().split())
        overlap = len(q_tokens & t_tokens)
        return overlap / len(q_tokens)

    def rerank(self, query: Query, candidates: list[Candidate]) -> list[Candidate]:
        if not candidates:
            return []

        if self.model is None:
            relevance_scores = [self._fallback_score(query.cleaned, candidate.text) for candidate in candidates]
        else:
            pairs = [(query.cleaned, candidate.text) for candidate in candidates]
            relevance_scores = self.model.predict(pairs, batch_size=self.batch_size)

        ranked = sorted(
            zip(candidates, relevance_scores, strict=True),
            key=lambda item: (float(item[1]), float(item[0].metadata.get("lexical_score", 0.0))),
            reverse=True,
        )

        output: list[Candidate] = []
        for candidate, relevance in ranked:
            candidate.metadata["rerank_score"] = float(relevance)
            output.append(candidate)
        return output
