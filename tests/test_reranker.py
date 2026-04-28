"""Tests for rag_v3.ranking.reranker."""
from __future__ import annotations

import pytest

from rag_v3.ranking.reranker import Reranker
from rag_v3.retrieval.schemas import Candidate, Query


def _make_query() -> Query:
    return Query(raw="q", cleaned="q", tokens=["q"])


def _cand(id: str, score: float) -> Candidate:
    return Candidate(id=id, text="text", score=score, metadata={})


class TestReranker:
    def setup_method(self):
        self.reranker = Reranker()

    def test_returns_list(self):
        result = self.reranker.rerank(_make_query(), [])
        assert isinstance(result, list)

    def test_empty_candidates(self):
        result = self.reranker.rerank(_make_query(), [])
        assert result == []

    def test_single_candidate_unchanged(self):
        c = _cand("a", 0.9)
        result = self.reranker.rerank(_make_query(), [c])
        assert result == [c]

    def test_sorted_by_descending_score(self):
        candidates = [_cand("a", 0.3), _cand("b", 0.9), _cand("c", 0.6)]
        result = self.reranker.rerank(_make_query(), candidates)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_already_sorted_input_unchanged(self):
        candidates = [_cand("a", 0.9), _cand("b", 0.6), _cand("c", 0.3)]
        result = self.reranker.rerank(_make_query(), candidates)
        assert result == candidates

    def test_tie_broken_by_id_ascending(self):
        candidates = [_cand("z", 0.5), _cand("a", 0.5), _cand("m", 0.5)]
        result = self.reranker.rerank(_make_query(), candidates)
        ids = [r.id for r in result]
        assert ids == ["a", "m", "z"]

    def test_does_not_mutate_input(self):
        candidates = [_cand("b", 0.2), _cand("a", 0.8)]
        original = list(candidates)
        self.reranker.rerank(_make_query(), candidates)
        assert candidates == original

    def test_query_ignored(self):
        """Reranker is query-agnostic; different queries yield same order."""
        candidates = [_cand("x", 0.4), _cand("y", 0.8)]
        q1 = Query(raw="foo", cleaned="foo", tokens=["foo"])
        q2 = Query(raw="bar", cleaned="bar", tokens=["bar"])
        assert self.reranker.rerank(q1, candidates) == self.reranker.rerank(q2, candidates)

    def test_all_candidates_preserved(self):
        candidates = [_cand(str(i), float(i) / 10) for i in range(10)]
        result = self.reranker.rerank(_make_query(), candidates)
        assert len(result) == 10

    def test_negative_scores_handled(self):
        candidates = [_cand("a", -0.5), _cand("b", -0.1), _cand("c", 0.0)]
        result = self.reranker.rerank(_make_query(), candidates)
        assert result[0].id == "c"
        assert result[-1].id == "a"
