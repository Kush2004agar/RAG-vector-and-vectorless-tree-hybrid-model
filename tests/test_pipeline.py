"""Tests for rag_v3.serving.pipeline."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag_v3.retrieval.schemas import Candidate, Query
from rag_v3.serving.pipeline import NOT_FOUND_MESSAGE, RagV3Pipeline


def _cand(id: str, text: str, score: float = 0.9) -> Candidate:
    return Candidate(
        id=id, text=text, score=score,
        metadata={"pdf_name": "doc.pdf", "section": "intro", "version": "v3", "feature_ids": []}
    )


class TestRagV3PipelineGenerateAnswer:
    """Test generate_answer without hitting QdrantStore or Gemini."""

    def _make_pipeline_no_llm(self) -> RagV3Pipeline:
        """Build a pipeline with patched QdrantStore and no LLM."""
        with patch("rag_v3.serving.pipeline.QdrantStore"), \
             patch.dict("os.environ", {}, clear=False):
            # Ensure GEMINI_API_KEY is absent so llm=None
            import os
            os.environ.pop("GEMINI_API_KEY", None)
            pipeline = RagV3Pipeline.__new__(RagV3Pipeline)
            pipeline.llm = None
            pipeline.query_processor = MagicMock()
            pipeline.retriever = MagicMock()
            pipeline.reranker = MagicMock()
            pipeline.context_builder = MagicMock()
            return pipeline

    def test_empty_context_returns_not_found(self):
        pipeline = self._make_pipeline_no_llm()
        result = pipeline.generate_answer("any question", "")
        assert result == NOT_FOUND_MESSAGE

    def test_no_llm_returns_not_found_with_note(self):
        pipeline = self._make_pipeline_no_llm()
        result = pipeline.generate_answer("question", "some context")
        assert NOT_FOUND_MESSAGE in result

    def test_llm_called_with_context_and_question(self):
        pipeline = self._make_pipeline_no_llm()
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "The answer is 42."
        mock_llm.models.generate_content.return_value = mock_response
        pipeline.llm = mock_llm

        result = pipeline.generate_answer("What is the answer?", "context here")
        assert result == "The answer is 42."
        mock_llm.models.generate_content.assert_called_once()

    def test_llm_empty_response_returns_not_found(self):
        pipeline = self._make_pipeline_no_llm()
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = ""
        mock_llm.models.generate_content.return_value = mock_response
        pipeline.llm = mock_llm

        result = pipeline.generate_answer("q", "context")
        assert result == NOT_FOUND_MESSAGE

    def test_llm_whitespace_response_returns_not_found(self):
        pipeline = self._make_pipeline_no_llm()
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "   "
        mock_llm.models.generate_content.return_value = mock_response
        pipeline.llm = mock_llm

        result = pipeline.generate_answer("q", "context")
        assert result == NOT_FOUND_MESSAGE

    def test_llm_none_text_returns_not_found(self):
        pipeline = self._make_pipeline_no_llm()
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = None
        mock_llm.models.generate_content.return_value = mock_response
        pipeline.llm = mock_llm

        result = pipeline.generate_answer("q", "context")
        assert result == NOT_FOUND_MESSAGE


class TestRagV3PipelineAnswer:
    """Test the full answer() flow with mocked components."""

    def _make_pipeline(self) -> RagV3Pipeline:
        pipeline = RagV3Pipeline.__new__(RagV3Pipeline)
        pipeline.llm = None
        pipeline.query_processor = MagicMock()
        pipeline.retriever = MagicMock()
        pipeline.reranker = MagicMock()
        pipeline.context_builder = MagicMock()
        return pipeline

    def test_answer_returns_string(self):
        pipeline = self._make_pipeline()
        query = Query(raw="q", cleaned="q", tokens=["q"])
        pipeline.query_processor.process.return_value = query
        pipeline.retriever.retrieve.return_value = []
        pipeline.reranker.rerank.return_value = []
        pipeline.context_builder.build.return_value = ""
        result = pipeline.answer("some question")
        assert isinstance(result, str)

    def test_answer_orchestrates_components(self):
        pipeline = self._make_pipeline()
        question = "What is encryption?"
        query = Query(raw=question, cleaned=question, tokens=["what", "is", "encryption"])
        candidates = [_cand("a", "Encryption is...")]
        reranked = candidates

        pipeline.query_processor.process.return_value = query
        pipeline.retriever.retrieve.return_value = candidates
        pipeline.reranker.rerank.return_value = reranked
        pipeline.context_builder.build.return_value = "Encryption is..."

        pipeline.answer(question)

        pipeline.query_processor.process.assert_called_once_with(question)
        pipeline.retriever.retrieve.assert_called_once_with(query, limit=50)
        pipeline.reranker.rerank.assert_called_once_with(query, candidates)
        pipeline.context_builder.build.assert_called_once_with(reranked)

    def test_answer_with_no_candidates_returns_not_found(self):
        pipeline = self._make_pipeline()
        query = Query(raw="q", cleaned="q", tokens=["q"])
        pipeline.query_processor.process.return_value = query
        pipeline.retriever.retrieve.return_value = []
        pipeline.reranker.rerank.return_value = []
        pipeline.context_builder.build.return_value = ""
        result = pipeline.answer("q")
        assert result == NOT_FOUND_MESSAGE

    def test_not_found_message_constant(self):
        assert NOT_FOUND_MESSAGE == "Information not found in available documents."
