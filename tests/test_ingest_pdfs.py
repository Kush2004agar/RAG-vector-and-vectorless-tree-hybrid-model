"""Tests for ingest_pdfs module."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import ingest_pdfs


class TestExtractPdfText:
    def test_returns_string(self):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page one content."
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = lambda s: s
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("ingest_pdfs.pdf_open", return_value=mock_pdf):
            result = ingest_pdfs.extract_pdf_text(Path("dummy.pdf"))

        assert isinstance(result, str)
        assert "Page one content." in result

    def test_multiple_pages_joined(self):
        pages_text = ["First page.", "Second page.", "Third page."]
        mock_pages = []
        for t in pages_text:
            p = MagicMock()
            p.extract_text.return_value = t
            mock_pages.append(p)

        mock_pdf = MagicMock()
        mock_pdf.pages = mock_pages
        mock_pdf.__enter__ = lambda s: s
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("ingest_pdfs.pdf_open", return_value=mock_pdf):
            result = ingest_pdfs.extract_pdf_text(Path("dummy.pdf"))

        for text in pages_text:
            assert text in result

    def test_none_page_text_excluded(self):
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = None
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Valid content."

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdf.__enter__ = lambda s: s
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("ingest_pdfs.pdf_open", return_value=mock_pdf):
            result = ingest_pdfs.extract_pdf_text(Path("dummy.pdf"))

        assert "Valid content." in result

    def test_empty_page_text_excluded(self):
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "   "  # whitespace only
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Real text."

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdf.__enter__ = lambda s: s
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("ingest_pdfs.pdf_open", return_value=mock_pdf):
            result = ingest_pdfs.extract_pdf_text(Path("dummy.pdf"))

        assert "Real text." in result

    def test_all_empty_pages_returns_empty_string(self):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = lambda s: s
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("ingest_pdfs.pdf_open", return_value=mock_pdf):
            result = ingest_pdfs.extract_pdf_text(Path("dummy.pdf"))

        assert result == ""


class TestIngestPdfs:
    def _make_pdf_path(self, name: str = "sample.pdf") -> MagicMock:
        pdf_path = MagicMock(spec=Path)
        pdf_path.name = name
        return pdf_path

    def test_no_pdfs_returns_empty_dict(self, tmp_path):
        with patch.object(ingest_pdfs, "INPUT_DIR", tmp_path):
            # No .pdf files in tmp_path
            result = ingest_pdfs.ingest_pdfs()
        assert result == {}

    def test_returns_dict_with_pdf_names_as_keys(self, tmp_path):
        pdf_path = self._make_pdf_path("sample.pdf")

        with patch.object(ingest_pdfs, "INPUT_DIR") as mock_dir, \
             patch.object(ingest_pdfs, "extract_pdf_text", return_value="Some content."):
            mock_dir.glob.return_value = iter([pdf_path])
            result = ingest_pdfs.ingest_pdfs()

        assert "sample.pdf" in result

    def test_chunks_have_required_keys(self):
        pdf_path = self._make_pdf_path("report.pdf")

        with patch.object(ingest_pdfs, "INPUT_DIR") as mock_dir, \
             patch.object(ingest_pdfs, "extract_pdf_text", return_value="Paragraph one.\n\nParagraph two."):
            mock_dir.glob.return_value = iter([pdf_path])
            result = ingest_pdfs.ingest_pdfs()

        chunks = result["report.pdf"]
        assert len(chunks) > 0
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "text" in chunk
            assert "section" in chunk
            assert "version" in chunk

    def test_section_is_full_text(self):
        pdf_path = self._make_pdf_path("doc.pdf")

        with patch.object(ingest_pdfs, "INPUT_DIR") as mock_dir, \
             patch.object(ingest_pdfs, "extract_pdf_text", return_value="Some text."):
            mock_dir.glob.return_value = iter([pdf_path])
            result = ingest_pdfs.ingest_pdfs()

        assert result["doc.pdf"][0]["section"] == "full_text"

    def test_version_is_v3(self):
        pdf_path = self._make_pdf_path("doc.pdf")

        with patch.object(ingest_pdfs, "INPUT_DIR") as mock_dir, \
             patch.object(ingest_pdfs, "extract_pdf_text", return_value="Some text."):
            mock_dir.glob.return_value = iter([pdf_path])
            result = ingest_pdfs.ingest_pdfs()

        assert result["doc.pdf"][0]["version"] == "v3"
