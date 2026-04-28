from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Chunk:
    id: str
    text: str
    pdf_name: str
    section: str
    version: str


class Chunker:
    """Deterministic paragraph chunker with a max character limit."""

    def __init__(self, max_chars: int = 1200) -> None:
        self.max_chars = max_chars

    def chunk_text(self, text: str, pdf_name: str, section: str = "unknown", version: str = "v3") -> list[Chunk]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[Chunk] = []

        cursor = 0
        for para in paragraphs:
            for start in range(0, len(para), self.max_chars):
                piece = para[start : start + self.max_chars]
                chunks.append(
                    Chunk(
                        id=f"{pdf_name}:{cursor}",
                        text=piece,
                        pdf_name=pdf_name,
                        section=section,
                        version=version,
                    )
                )
                cursor += 1

        return chunks
