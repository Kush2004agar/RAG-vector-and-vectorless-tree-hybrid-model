"""Tests for rag_v3.indexing.index_builder."""
from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from rag_v3.indexing.index_builder import IndexBuilder
from rag_v3.ingestion.chunker import Chunk


def _make_chunk(id: str = "doc:0", text: str = "hello", pdf: str = "doc.pdf") -> Chunk:
    return Chunk(id=id, text=text, pdf_name=pdf, section="s", version="v3")


class TestIndexBuilder:
    def setup_method(self):
        self.embedder = MagicMock()
        self.store = MagicMock()
        self.feature_extractor = MagicMock()
        self.builder = IndexBuilder(
            embedder=self.embedder,
            store=self.store,
            feature_extractor=self.feature_extractor,
        )

    def test_ensure_collection_called(self):
        self.embedder.embed.return_value = [0.1] * 256
        self.feature_extractor.extract.return_value = ["hello"]
        self.builder.rebuild([_make_chunk()])
        self.store.ensure_collection.assert_called_once()

    def test_empty_chunks_no_upsert(self):
        self.builder.rebuild([])
        self.store.ensure_collection.assert_called_once()
        self.store.upsert.assert_not_called()

    def test_upsert_called_with_correct_ids(self):
        chunk = _make_chunk(id="doc:42", text="content")
        self.embedder.embed.return_value = [0.0] * 256
        self.feature_extractor.extract.return_value = ["content"]
        self.builder.rebuild([chunk])
        call_kwargs = self.store.upsert.call_args
        assert "doc:42" in call_kwargs.kwargs.get("ids", call_kwargs[1].get("ids", []))

    def test_upsert_called_with_correct_texts(self):
        chunk = _make_chunk(text="my text here")
        self.embedder.embed.return_value = [0.0] * 256
        self.feature_extractor.extract.return_value = []
        self.builder.rebuild([chunk])
        call_kwargs = self.store.upsert.call_args
        texts = call_kwargs.kwargs.get("texts") or call_kwargs[1].get("texts")
        assert "my text here" in texts

    def test_embedder_called_per_chunk(self):
        chunks = [_make_chunk(id=f"d:{i}", text=f"text{i}") for i in range(3)]
        self.embedder.embed.return_value = [0.0] * 256
        self.feature_extractor.extract.return_value = []
        self.builder.rebuild(chunks)
        assert self.embedder.embed.call_count == 3

    def test_feature_extractor_called_per_chunk(self):
        chunks = [_make_chunk(id=f"d:{i}", text=f"text{i}") for i in range(4)]
        self.embedder.embed.return_value = [0.0] * 256
        self.feature_extractor.extract.return_value = []
        self.builder.rebuild(chunks)
        assert self.feature_extractor.extract.call_count == 4

    def test_payload_contains_metadata(self):
        chunk = Chunk(id="x:0", text="hello", pdf_name="file.pdf", section="ch1", version="v3")
        self.embedder.embed.return_value = [0.0] * 256
        self.feature_extractor.extract.return_value = ["hello"]
        self.builder.rebuild([chunk])
        call_kwargs = self.store.upsert.call_args
        payloads = call_kwargs.kwargs.get("payloads") or call_kwargs[1].get("payloads")
        assert payloads[0]["pdf_name"] == "file.pdf"
        assert payloads[0]["section"] == "ch1"
        assert payloads[0]["version"] == "v3"
        assert payloads[0]["feature_ids"] == ["hello"]

    def test_multiple_chunks_upserted_in_one_call(self):
        chunks = [_make_chunk(id=f"d:{i}") for i in range(5)]
        self.embedder.embed.return_value = [0.0] * 256
        self.feature_extractor.extract.return_value = []
        self.builder.rebuild(chunks)
        self.store.upsert.assert_called_once()
        call_kwargs = self.store.upsert.call_args
        ids = call_kwargs.kwargs.get("ids") or call_kwargs[1].get("ids")
        assert len(ids) == 5
