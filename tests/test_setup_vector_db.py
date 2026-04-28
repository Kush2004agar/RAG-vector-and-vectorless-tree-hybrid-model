"""Tests for setup_vector_db (load_chunks and setup_databases)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import setup_vector_db
from rag_v3.ingestion.chunker import Chunk


class TestLoadChunks:
    def test_missing_file_returns_empty(self, tmp_path):
        nonexistent = tmp_path / "no_such_file.json"
        with patch.object(setup_vector_db, "RAW_CHUNKS_FILE", nonexistent):
            result = setup_vector_db.load_chunks()
        assert result == []

    def test_loads_chunks_from_json(self, tmp_path):
        data = {
            "doc.pdf": [
                {"chunk_id": "doc.pdf:0", "text": "Hello world", "section": "intro", "version": "v3"},
            ]
        }
        json_file = tmp_path / "chunks.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        with patch.object(setup_vector_db, "RAW_CHUNKS_FILE", json_file):
            result = setup_vector_db.load_chunks()

        assert len(result) == 1
        assert isinstance(result[0], Chunk)
        assert result[0].id == "doc.pdf:0"
        assert result[0].text == "Hello world"
        assert result[0].pdf_name == "doc.pdf"
        assert result[0].section == "intro"

    def test_loads_multiple_pdfs(self, tmp_path):
        data = {
            "a.pdf": [
                {"chunk_id": "a:0", "text": "text a", "section": "s", "version": "v3"},
            ],
            "b.pdf": [
                {"chunk_id": "b:0", "text": "text b", "section": "s", "version": "v3"},
                {"chunk_id": "b:1", "text": "text b2", "section": "s", "version": "v3"},
            ],
        }
        json_file = tmp_path / "chunks.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        with patch.object(setup_vector_db, "RAW_CHUNKS_FILE", json_file):
            result = setup_vector_db.load_chunks()

        assert len(result) == 3

    def test_missing_fields_use_defaults(self, tmp_path):
        data = {
            "x.pdf": [
                {},  # completely empty item
            ]
        }
        json_file = tmp_path / "chunks.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        with patch.object(setup_vector_db, "RAW_CHUNKS_FILE", json_file):
            result = setup_vector_db.load_chunks()

        assert len(result) == 1
        assert result[0].id == ""
        assert result[0].text == ""
        assert result[0].section == "unknown"

    def test_version_always_v3(self, tmp_path):
        data = {
            "doc.pdf": [
                {"chunk_id": "d:0", "text": "t", "section": "s", "version": "v1"},
            ]
        }
        json_file = tmp_path / "chunks.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        with patch.object(setup_vector_db, "RAW_CHUNKS_FILE", json_file):
            result = setup_vector_db.load_chunks()

        # version is hard-coded to "v3" in load_chunks
        assert result[0].version == "v3"


class TestSetupDatabases:
    def test_no_chunks_prints_message(self, tmp_path, capsys):
        nonexistent = tmp_path / "no_chunks.json"
        with patch.object(setup_vector_db, "RAW_CHUNKS_FILE", nonexistent):
            setup_vector_db.setup_databases()
        captured = capsys.readouterr()
        assert "No chunks found" in captured.out or "ingest_pdfs" in captured.out

    def test_rebuilds_index_when_chunks_present(self, tmp_path):
        data = {
            "doc.pdf": [
                {"chunk_id": "doc:0", "text": "hello world", "section": "s", "version": "v3"},
            ]
        }
        json_file = tmp_path / "chunks.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        mock_builder = MagicMock()
        with patch.object(setup_vector_db, "RAW_CHUNKS_FILE", json_file), \
             patch("setup_vector_db.QdrantStore"), \
             patch("setup_vector_db.IndexBuilder", return_value=mock_builder):
            setup_vector_db.setup_databases()

        mock_builder.rebuild.assert_called_once()

    def test_prints_indexed_count(self, tmp_path, capsys):
        data = {
            "doc.pdf": [
                {"chunk_id": "doc:0", "text": "hello", "section": "s", "version": "v3"},
            ]
        }
        json_file = tmp_path / "chunks.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        with patch.object(setup_vector_db, "RAW_CHUNKS_FILE", json_file), \
             patch("setup_vector_db.QdrantStore"), \
             patch("setup_vector_db.IndexBuilder"):
            setup_vector_db.setup_databases()

        captured = capsys.readouterr()
        assert "1" in captured.out
