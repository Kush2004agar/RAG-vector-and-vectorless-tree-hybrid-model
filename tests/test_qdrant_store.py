"""Tests for rag_v3.indexing.qdrant_client (QdrantStore)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag_v3.indexing.qdrant_client import QdrantStore


def _make_store(vector_size: int = 256) -> QdrantStore:
    """Return a QdrantStore with a mocked underlying QdrantClient."""
    with patch("rag_v3.indexing.qdrant_client.QdrantClient"):
        store = QdrantStore(vector_size=vector_size)
    return store


class TestQdrantStore:
    def test_collection_name_constant(self):
        assert QdrantStore.COLLECTION_NAME == "documents"

    def test_vector_size_stored(self):
        store = _make_store(vector_size=128)
        assert store.vector_size == 128

    def test_ensure_collection_calls_recreate(self):
        store = _make_store()
        store.ensure_collection()
        store.client.recreate_collection.assert_called_once()

    def test_ensure_collection_uses_correct_name(self):
        store = _make_store()
        store.ensure_collection()
        call_kwargs = store.client.recreate_collection.call_args
        name = call_kwargs.kwargs.get("collection_name") or call_kwargs[0][0]
        assert name == "documents"

    def test_upsert_calls_client_upsert(self):
        store = _make_store()
        store.upsert(
            ids=["a:0"],
            vectors=[[0.1] * 256],
            texts=["hello"],
            payloads=[{"pdf_name": "doc.pdf"}],
        )
        store.client.upsert.assert_called_once()

    def test_upsert_passes_collection_name(self):
        store = _make_store()
        store.upsert(
            ids=["x:0"],
            vectors=[[0.0] * 256],
            texts=["text"],
            payloads=[{}],
        )
        call_kwargs = store.client.upsert.call_args
        name = call_kwargs.kwargs.get("collection_name") or call_kwargs[0][0]
        assert name == "documents"

    def test_upsert_includes_text_in_payload(self):
        store = _make_store()
        store.upsert(
            ids=["doc:0"],
            vectors=[[0.1] * 256],
            texts=["my text"],
            payloads=[{"pdf_name": "f.pdf"}],
        )
        call_kwargs = store.client.upsert.call_args
        points = call_kwargs.kwargs.get("points") or call_kwargs[1].get("points")
        assert points[0].payload["text"] == "my text"

    def test_upsert_multiple_points(self):
        store = _make_store()
        store.upsert(
            ids=["a:0", "b:0"],
            vectors=[[0.1] * 256, [0.2] * 256],
            texts=["text a", "text b"],
            payloads=[{"pdf_name": "a.pdf"}, {"pdf_name": "b.pdf"}],
        )
        call_kwargs = store.client.upsert.call_args
        points = call_kwargs.kwargs.get("points") or call_kwargs[1].get("points")
        assert len(points) == 2

    def test_search_calls_client_search(self):
        store = _make_store()
        store.client.search.return_value = []
        result = store.search(query_vector=[0.1] * 256, limit=10)
        store.client.search.assert_called_once()
        assert result == []

    def test_search_passes_collection_name(self):
        store = _make_store()
        store.client.search.return_value = []
        store.search(query_vector=[0.0] * 256, limit=5)
        call_kwargs = store.client.search.call_args
        name = call_kwargs.kwargs.get("collection_name") or call_kwargs[0][0]
        assert name == "documents"

    def test_search_passes_limit(self):
        store = _make_store()
        store.client.search.return_value = []
        store.search(query_vector=[0.0] * 256, limit=25)
        call_kwargs = store.client.search.call_args
        limit = call_kwargs.kwargs.get("limit") or call_kwargs[1].get("limit")
        assert limit == 25

    def test_search_passes_filter(self):
        store = _make_store()
        store.client.search.return_value = []
        mock_filter = MagicMock()
        store.search(query_vector=[0.0] * 256, limit=10, query_filter=mock_filter)
        call_kwargs = store.client.search.call_args
        qf = call_kwargs.kwargs.get("query_filter") or call_kwargs[1].get("query_filter")
        assert qf is mock_filter

    def test_search_with_payload_enabled(self):
        store = _make_store()
        store.client.search.return_value = []
        store.search(query_vector=[0.0] * 256, limit=5)
        call_kwargs = store.client.search.call_args
        with_payload = call_kwargs.kwargs.get("with_payload") or call_kwargs[1].get("with_payload")
        assert with_payload is True

    def test_qdrant_url_default(self):
        with patch("rag_v3.indexing.qdrant_client.QdrantClient") as mock_cls, \
             patch.dict("os.environ", {}, clear=True):
            QdrantStore(vector_size=64)
        call_kwargs = mock_cls.call_args
        url = call_kwargs.kwargs.get("url") or call_kwargs[0][0]
        assert url == "http://localhost:6333"

    def test_qdrant_url_from_env(self):
        custom_url = "http://custom-qdrant:9999"
        with patch("rag_v3.indexing.qdrant_client.QdrantClient") as mock_cls, \
             patch.dict("os.environ", {"QDRANT_URL": custom_url}):
            QdrantStore(vector_size=64)
        call_kwargs = mock_cls.call_args
        url = call_kwargs.kwargs.get("url") or call_kwargs[0][0]
        assert url == custom_url
