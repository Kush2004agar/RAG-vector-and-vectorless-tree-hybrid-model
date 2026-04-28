"""Tests for rag_v3.context.context_builder."""
from __future__ import annotations

import pytest

from rag_v3.context.context_builder import ContextBuilder
from rag_v3.retrieval.schemas import Candidate


def _cand(id: str, text: str = "content", pdf: str = "doc.pdf", section: str = "intro") -> Candidate:
    return Candidate(id=id, text=text, score=1.0, metadata={"pdf_name": pdf, "section": section})


class TestContextBuilderSelect:
    def test_default_min_max(self):
        cb = ContextBuilder()
        assert cb.min_chunks == 5
        assert cb.max_chunks == 8

    def test_custom_min_max(self):
        cb = ContextBuilder(min_chunks=2, max_chunks=4)
        assert cb.min_chunks == 2
        assert cb.max_chunks == 4

    def test_empty_candidates_returns_empty(self):
        cb = ContextBuilder(min_chunks=5, max_chunks=8)
        result = cb.select([])
        assert result == []

    def test_fewer_than_max_returns_all(self):
        cb = ContextBuilder(min_chunks=5, max_chunks=8)
        candidates = [_cand(str(i)) for i in range(3)]
        result = cb.select(candidates)
        assert result == candidates

    def test_exactly_max_returns_all(self):
        cb = ContextBuilder(min_chunks=5, max_chunks=8)
        candidates = [_cand(str(i)) for i in range(8)]
        result = cb.select(candidates)
        assert len(result) == 8

    def test_more_than_max_capped_at_max(self):
        cb = ContextBuilder(min_chunks=5, max_chunks=8)
        candidates = [_cand(str(i)) for i in range(20)]
        result = cb.select(candidates)
        assert len(result) == 8

    def test_preserves_order(self):
        cb = ContextBuilder(min_chunks=2, max_chunks=4)
        candidates = [_cand(str(i)) for i in range(10)]
        result = cb.select(candidates)
        assert result == candidates[:4]


class TestContextBuilderBuild:
    def test_returns_string(self):
        cb = ContextBuilder(min_chunks=1, max_chunks=3)
        result = cb.build([_cand("a", text="Hello")])
        assert isinstance(result, str)

    def test_empty_candidates_returns_empty_string(self):
        cb = ContextBuilder()
        result = cb.build([])
        assert result == ""

    def test_block_format_contains_index(self):
        cb = ContextBuilder(min_chunks=1, max_chunks=5)
        result = cb.build([_cand("a", text="Some text", pdf="my.pdf", section="sec1")])
        assert "[1]" in result

    def test_block_format_contains_pdf_and_section(self):
        cb = ContextBuilder(min_chunks=1, max_chunks=5)
        result = cb.build([_cand("a", text="Some text", pdf="report.pdf", section="abstract")])
        assert "report.pdf" in result
        assert "abstract" in result

    def test_block_format_contains_text(self):
        cb = ContextBuilder(min_chunks=1, max_chunks=5)
        result = cb.build([_cand("a", text="The actual content")])
        assert "The actual content" in result

    def test_multiple_blocks_separated_by_blank_lines(self):
        cb = ContextBuilder(min_chunks=1, max_chunks=5)
        candidates = [_cand(str(i), text=f"text{i}") for i in range(3)]
        result = cb.build(candidates)
        assert "\n\n" in result

    def test_blocks_numbered_sequentially(self):
        cb = ContextBuilder(min_chunks=1, max_chunks=5)
        candidates = [_cand(str(i), text=f"t{i}") for i in range(3)]
        result = cb.build(candidates)
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result

    def test_only_max_chunks_included(self):
        cb = ContextBuilder(min_chunks=2, max_chunks=3)
        candidates = [_cand(str(i), text=f"text{i}") for i in range(10)]
        result = cb.build(candidates)
        assert "[4]" not in result
        assert "[3]" in result

    def test_text_stripped(self):
        cb = ContextBuilder(min_chunks=1, max_chunks=5)
        result = cb.build([_cand("a", text="  padded  ")])
        assert "padded" in result
        # Leading/trailing spaces on text should be stripped
        assert "  padded  " not in result

    def test_missing_metadata_defaults(self):
        cb = ContextBuilder(min_chunks=1, max_chunks=5)
        c = Candidate(id="x", text="text", score=1.0, metadata={})
        result = cb.build([c])
        # Should not raise; empty strings for pdf/section
        assert "[1]" in result
        assert "text" in result
