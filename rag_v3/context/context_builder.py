from __future__ import annotations

from collections import defaultdict

from rag_v3.retrieval.schemas import Candidate


class ContextBuilder:
    """Build context with text deduplication and document diversity constraints."""

    def __init__(self, min_chunks: int = 5, max_chunks: int = 8, max_per_document: int = 2) -> None:
        self.min_chunks = min_chunks
        self.max_chunks = max_chunks
        self.max_per_document = max_per_document

    @staticmethod
    def _normalized_text(text: str) -> str:
        return " ".join(text.lower().split())

    def select(self, candidates: list[Candidate]) -> list[Candidate]:
        selected: list[Candidate] = []
        seen_texts: set[str] = set()
        per_document_count: defaultdict[str, int] = defaultdict(int)

        for candidate in candidates:
            if len(selected) >= self.max_chunks:
                break

            normalized = self._normalized_text(candidate.text)
            if not normalized or normalized in seen_texts:
                continue

            pdf_name = str(candidate.metadata.get("pdf_name", ""))
            if per_document_count[pdf_name] >= self.max_per_document:
                continue

            selected.append(candidate)
            seen_texts.add(normalized)
            per_document_count[pdf_name] += 1

        return selected[: self.max_chunks]

    def build(self, candidates: list[Candidate]) -> str:
        selected = self.select(candidates)
        blocks: list[str] = []
        for idx, candidate in enumerate(selected, start=1):
            pdf = candidate.metadata.get("pdf_name", "")
            section = candidate.metadata.get("section", "")
            blocks.append(f"[{idx}] ({pdf} | {section})\n{candidate.text.strip()}")

        return "\n\n".join(blocks)
