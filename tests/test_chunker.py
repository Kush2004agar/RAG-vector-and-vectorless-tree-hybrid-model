"""Tests for rag_v3.ingestion.chunker."""
from __future__ import annotations

import pytest

from rag_v3.ingestion.chunker import Chunk, Chunker


class TestChunkDataclass:
    def test_chunk_fields(self):
        chunk = Chunk(id="doc:0", text="hello", pdf_name="doc.pdf", section="intro", version="v3")
        assert chunk.id == "doc:0"
        assert chunk.text == "hello"
        assert chunk.pdf_name == "doc.pdf"
        assert chunk.section == "intro"
        assert chunk.version == "v3"

    def test_chunk_equality(self):
        c1 = Chunk(id="a:0", text="x", pdf_name="a.pdf", section="s", version="v3")
        c2 = Chunk(id="a:0", text="x", pdf_name="a.pdf", section="s", version="v3")
        assert c1 == c2

    def test_chunk_inequality(self):
        c1 = Chunk(id="a:0", text="x", pdf_name="a.pdf", section="s", version="v3")
        c2 = Chunk(id="b:0", text="x", pdf_name="a.pdf", section="s", version="v3")
        assert c1 != c2


class TestChunker:
    def test_default_max_chars(self):
        chunker = Chunker()
        assert chunker.max_chars == 1200

    def test_custom_max_chars(self):
        chunker = Chunker(max_chars=500)
        assert chunker.max_chars == 500

    def test_single_short_paragraph(self):
        chunker = Chunker(max_chars=1200)
        chunks = chunker.chunk_text("Hello world.", "doc.pdf")
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world."
        assert chunks[0].pdf_name == "doc.pdf"
        assert chunks[0].section == "unknown"
        assert chunks[0].version == "v3"

    def test_empty_text_returns_no_chunks(self):
        chunker = Chunker()
        chunks = chunker.chunk_text("", "doc.pdf")
        assert chunks == []

    def test_whitespace_only_text_returns_no_chunks(self):
        chunker = Chunker()
        chunks = chunker.chunk_text("   \n\n   ", "doc.pdf")
        assert chunks == []

    def test_multiple_paragraphs(self):
        chunker = Chunker(max_chars=1200)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunker.chunk_text(text, "doc.pdf")
        assert len(chunks) == 3
        assert chunks[0].text == "First paragraph."
        assert chunks[1].text == "Second paragraph."
        assert chunks[2].text == "Third paragraph."

    def test_long_paragraph_is_split(self):
        chunker = Chunker(max_chars=10)
        # "0123456789AB" is 12 chars → 2 pieces: "0123456789" and "AB"
        text = "0123456789AB"
        chunks = chunker.chunk_text(text, "doc.pdf")
        assert len(chunks) == 2
        assert chunks[0].text == "0123456789"
        assert chunks[1].text == "AB"

    def test_paragraph_exactly_at_max_chars_is_one_chunk(self):
        chunker = Chunker(max_chars=10)
        text = "0123456789"  # exactly 10 chars
        chunks = chunker.chunk_text(text, "doc.pdf")
        assert len(chunks) == 1
        assert chunks[0].text == "0123456789"

    def test_chunk_ids_are_sequential(self):
        chunker = Chunker(max_chars=1200)
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = chunker.chunk_text(text, "mydoc.pdf")
        for i, chunk in enumerate(chunks):
            assert chunk.id == f"mydoc.pdf:{i}"

    def test_custom_section_and_version(self):
        chunker = Chunker()
        chunks = chunker.chunk_text("Some text.", "file.pdf", section="chapter1", version="v1")
        assert chunks[0].section == "chapter1"
        assert chunks[0].version == "v1"

    def test_ids_continue_across_paragraphs(self):
        chunker = Chunker(max_chars=5)
        # "ABCDEFGH" splits into "ABCDE", "FGH" → 2 chunks from first para
        # "XY" → 1 chunk from second para
        text = "ABCDEFGH\n\nXY"
        chunks = chunker.chunk_text(text, "doc.pdf")
        assert len(chunks) == 3
        assert chunks[0].id == "doc.pdf:0"
        assert chunks[1].id == "doc.pdf:1"
        assert chunks[2].id == "doc.pdf:2"

    def test_paragraph_with_only_whitespace_is_skipped(self):
        chunker = Chunker()
        text = "First.\n\n   \n\nSecond."
        chunks = chunker.chunk_text(text, "doc.pdf")
        assert len(chunks) == 2
        assert chunks[0].text == "First."
        assert chunks[1].text == "Second."

    def test_strips_paragraph_whitespace(self):
        chunker = Chunker()
        text = "  Padded paragraph.  "
        chunks = chunker.chunk_text(text, "doc.pdf")
        assert len(chunks) == 1
        assert chunks[0].text == "Padded paragraph."
