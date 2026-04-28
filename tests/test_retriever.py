"""Tests for rag_v3.retrieval.retriever."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag_v3.retrieval.retriever import Retriever
from rag_v3.retrieval.schemas import Candidate, Query


def _make_scored_point(id: str, score: float, payload: dict | None = None):
    """Create a mock ScoredPoint-like object."""
    point = MagicMock()
    point.id = id
    point.score = score
    point.payload = payload or {}
    return point


def _make_query(feature_ids: list[str] | None = None) -> Query:
    return Query(
        raw="what is tls",
        cleaned="what is tls",
        tokens=["what", "is", "tls"],
        feature_ids=feature_ids or ["tls"],
    )


class TestRetriever:
    def setup_method(self):
        self.store = MagicMock()
        self.embedder = MagicMock()
        self.filter_builder = MagicMock()
        self.retriever = Retriever(
            store=self.store,
            embedder=self.embedder,
            filter_builder=self.filter_builder,
        )

    def test_returns_list_of_candidates(self):
        self.embedder.embed.return_value = [0.1] * 256
        self.filter_builder.build.return_value = None
        self.store.search.return_value = []
        result = self.retriever.retrieve(_make_query())
        assert isinstance(result, list)

    def test_empty_search_results(self):
        self.embedder.embed.return_value = [0.1] * 256
        self.filter_builder.build.return_value = None
        self.store.search.return_value = []
        result = self.retriever.retrieve(_make_query())
        assert result == []

    def test_candidates_mapped_correctly(self):
        payload = {
            "text": "Hello world",
            "feature_ids": ["hello", "world"],
            "pdf_name": "doc.pdf",
            "section": "intro",
            "version": "v3",
        }
        point = _make_scored_point("doc:0", 0.9, payload)
        self.embedder.embed.return_value = [0.0] * 256
        self.filter_builder.build.return_value = None
        self.store.search.return_value = [point]

        result = self.retriever.retrieve(_make_query())
        assert len(result) == 1
        c = result[0]
        assert isinstance(c, Candidate)
        assert c.id == "doc:0"
        assert c.text == "Hello world"
        assert c.score == 0.9
        assert c.metadata["pdf_name"] == "doc.pdf"
        assert c.metadata["section"] == "intro"
        assert c.metadata["version"] == "v3"

    def test_embedder_called_with_cleaned_query(self):
        self.embedder.embed.return_value = [0.0] * 256
        self.filter_builder.build.return_value = None
        self.store.search.return_value = []
        query = _make_query()
        self.retriever.retrieve(query)
        self.embedder.embed.assert_called_once_with("what is tls")

    def test_filter_builder_called_with_query(self):
        self.embedder.embed.return_value = [0.0] * 256
        self.filter_builder.build.return_value = None
        self.store.search.return_value = []
        query = _make_query()
        self.retriever.retrieve(query)
        self.filter_builder.build.assert_called_once_with(query)

    def test_store_search_called_with_correct_args(self):
        vector = [0.5] * 256
        mock_filter = MagicMock()
        self.embedder.embed.return_value = vector
        self.filter_builder.build.return_value = mock_filter
        self.store.search.return_value = []
        self.retriever.retrieve(_make_query(), limit=25)
        self.store.search.assert_called_once_with(
            query_vector=vector, limit=25, query_filter=mock_filter
        )

    def test_missing_payload_fields_default_to_empty(self):
        point = _make_scored_point("x:0", 0.5, {})
        self.embedder.embed.return_value = [0.0] * 256
        self.filter_builder.build.return_value = None
        self.store.search.return_value = [point]
        result = self.retriever.retrieve(_make_query())
        c = result[0]
        assert c.text == ""
        assert c.metadata["pdf_name"] == ""
        assert c.metadata["section"] == ""
        assert c.metadata["version"] == ""
        assert c.metadata["feature_ids"] == []

    def test_none_payload_handled(self):
        point = _make_scored_point("x:0", 0.5)
        point.payload = None
        self.embedder.embed.return_value = [0.0] * 256
        self.filter_builder.build.return_value = None
        self.store.search.return_value = [point]
        result = self.retriever.retrieve(_make_query())
        assert len(result) == 1
        assert result[0].text == ""

    def test_default_limit_is_50(self):
        self.embedder.embed.return_value = [0.0] * 256
        self.filter_builder.build.return_value = None
        self.store.search.return_value = []
        self.retriever.retrieve(_make_query())
        call_kwargs = self.store.search.call_args
        assert call_kwargs.kwargs.get("limit") == 50 or call_kwargs[1].get("limit") == 50

    def test_multiple_candidates_returned(self):
        points = [
            _make_scored_point(f"doc:{i}", float(i) / 10, {"text": f"text{i}"})
            for i in range(5)
        ]
        self.embedder.embed.return_value = [0.0] * 256
        self.filter_builder.build.return_value = None
        self.store.search.return_value = points
        result = self.retriever.retrieve(_make_query())
        assert len(result) == 5
