from __future__ import annotations

from dataclasses import dataclass

from rag_v3.evaluation.dataset import EvalSample
from rag_v3.evaluation.metrics import recall_at_k, reciprocal_rank
from rag_v3.serving.pipeline import RagV3Pipeline


@dataclass(slots=True)
class EvalResult:
    recall_at_k: float
    mrr: float
    samples: int


class Evaluator:
    """Evaluates retrieval quality without LLM generation."""

    def __init__(self, pipeline: RagV3Pipeline) -> None:
        self.pipeline = pipeline

    def evaluate(self, dataset: list[EvalSample], k: int = 10, retrieval_k: int = 50) -> EvalResult:
        if not dataset:
            return EvalResult(recall_at_k=0.0, mrr=0.0, samples=0)

        recall_total = 0.0
        rr_total = 0.0

        for sample in dataset:
            candidates = self.pipeline.retrieve_only(sample.query, top_k=retrieval_k)
            predicted_ids = [candidate.id for candidate in candidates]
            relevant = set(sample.relevant_chunk_ids)

            recall_total += recall_at_k(predicted_ids, relevant, k)
            rr_total += reciprocal_rank(predicted_ids, relevant)

        sample_count = len(dataset)
        return EvalResult(
            recall_at_k=recall_total / sample_count,
            mrr=rr_total / sample_count,
            samples=sample_count,
        )
