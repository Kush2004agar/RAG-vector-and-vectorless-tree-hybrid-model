"""Tests for rag_v3.retrieval.schemas."""
from __future__ import annotations

import pytest

from rag_v3.retrieval.schemas import Candidate, Query


class TestQuery:
    def test_query_fields(self):
        q = Query(raw="What is TLS?", cleaned="What is TLS?", tokens=["what", "is", "tls"])
        assert q.raw == "What is TLS?"
        assert q.cleaned == "What is TLS?"
        assert q.tokens == ["what", "is", "tls"]
        assert q.feature_ids == []

    def test_query_with_feature_ids(self):
        q = Query(raw="q", cleaned="q", tokens=["q"], feature_ids=["tls", "ssl"])
        assert q.feature_ids == ["tls", "ssl"]

    def test_query_equality(self):
        q1 = Query(raw="q", cleaned="q", tokens=["q"])
        q2 = Query(raw="q", cleaned="q", tokens=["q"])
        assert q1 == q2

    def test_query_default_feature_ids_is_empty_list(self):
        q = Query(raw="hello", cleaned="hello", tokens=["hello"])
        assert isinstance(q.feature_ids, list)
        assert len(q.feature_ids) == 0


class TestCandidate:
    def test_candidate_fields(self):
        c = Candidate(id="doc:1", text="Some text.", score=0.87, metadata={"pdf_name": "x.pdf"})
        assert c.id == "doc:1"
        assert c.text == "Some text."
        assert c.score == 0.87
        assert c.metadata == {"pdf_name": "x.pdf"}

    def test_candidate_equality(self):
        c1 = Candidate(id="a", text="t", score=0.5, metadata={})
        c2 = Candidate(id="a", text="t", score=0.5, metadata={})
        assert c1 == c2

    def test_candidate_inequality(self):
        c1 = Candidate(id="a", text="t", score=0.5, metadata={})
        c2 = Candidate(id="b", text="t", score=0.5, metadata={})
        assert c1 != c2

    def test_candidate_zero_score(self):
        c = Candidate(id="z", text="text", score=0.0, metadata={})
        assert c.score == 0.0

    def test_candidate_negative_score(self):
        c = Candidate(id="neg", text="text", score=-0.1, metadata={})
        assert c.score == -0.1
