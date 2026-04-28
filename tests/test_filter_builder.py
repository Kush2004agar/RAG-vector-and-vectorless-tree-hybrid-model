"""Tests for rag_v3.retrieval.filter_builder."""
from __future__ import annotations

import pytest

from rag_v3.retrieval.filter_builder import FilterBuilder
from rag_v3.retrieval.schemas import Query


def _make_query(feature_ids: list[str]) -> Query:
    return Query(raw="q", cleaned="q", tokens=["q"], feature_ids=feature_ids)


class TestFilterBuilder:
    def setup_method(self):
        self.builder = FilterBuilder()

    def test_returns_none_for_empty_feature_ids(self):
        query = _make_query([])
        result = self.builder.build(query)
        assert result is None

    def test_returns_filter_for_non_empty_feature_ids(self):
        query = _make_query(["tls", "ssl"])
        result = self.builder.build(query)
        assert result is not None

    def test_filter_has_must_clause(self):
        query = _make_query(["tls"])
        result = self.builder.build(query)
        assert hasattr(result, "must")
        assert result.must is not None
        assert len(result.must) == 1

    def test_filter_condition_key(self):
        query = _make_query(["auth"])
        result = self.builder.build(query)
        condition = result.must[0]
        assert condition.key == "feature_ids"

    def test_filter_match_contains_feature_ids(self):
        query = _make_query(["auth", "token"])
        result = self.builder.build(query)
        condition = result.must[0]
        assert hasattr(condition.match, "any")
        assert "auth" in condition.match.any
        assert "token" in condition.match.any

    def test_single_feature_id(self):
        query = _make_query(["python"])
        result = self.builder.build(query)
        condition = result.must[0]
        assert condition.match.any == ["python"]

    def test_many_feature_ids(self):
        ids = [f"feat{i}" for i in range(20)]
        query = _make_query(ids)
        result = self.builder.build(query)
        condition = result.must[0]
        assert len(condition.match.any) == 20

    def test_filter_type(self):
        from qdrant_client.http import models
        query = _make_query(["x"])
        result = self.builder.build(query)
        assert isinstance(result, models.Filter)
