from __future__ import annotations

from rag_v3.retrieval.schemas import Candidate


class ContextBuilder:
    """Builds deterministic context from top reranked candidates."""

    def __init__(self, min_chunks: int = 5, max_chunks: int = 8) -> None:
        self.min_chunks = min_chunks
        self.max_chunks = max_chunks

    def select(self, candidates: list[Candidate]) -> list[Candidate]:
        if len(candidates) <= self.max_chunks:
            return candidates[: self.max_chunks]

        count = max(self.min_chunks, min(self.max_chunks, len(candidates)))
        return candidates[:count]

    def build(self, candidates: list[Candidate]) -> str:
        selected = self.select(candidates)
        blocks: list[str] = []
        for idx, candidate in enumerate(selected, start=1):
            pdf = candidate.metadata.get("pdf_name", "")
            section = candidate.metadata.get("section", "")
            blocks.append(f"[{idx}] ({pdf} | {section})\n{candidate.text.strip()}")

        return "\n\n".join(blocks)
