"""Tests for rag_v3.retrieval.query_processor."""
from __future__ import annotations

import pytest

from rag_v3.ingestion.feature_extractor import FeatureExtractor
from rag_v3.retrieval.query_processor import QueryProcessor
from rag_v3.retrieval.schemas import Query


class TestQueryProcessor:
    def setup_method(self):
        self.processor = QueryProcessor(FeatureExtractor())

    def test_returns_query_instance(self):
        result = self.processor.process("What is TLS?")
        assert isinstance(result, Query)

    def test_raw_preserves_original(self):
        raw = "  What  is   TLS?  "
        result = self.processor.process(raw)
        assert result.raw == raw

    def test_cleaned_collapses_whitespace(self):
        result = self.processor.process("  What  is   TLS?  ")
        assert result.cleaned == "What is TLS?"

    def test_tokens_are_lowercase_alphanumeric(self):
        result = self.processor.process("What is TLS version 1.2?")
        assert "tls" in result.tokens
        assert "version" in result.tokens
        assert "1" in result.tokens or "1.2" in result.tokens or "12" in result.tokens

    def test_feature_ids_populated(self):
        result = self.processor.process("neural network classification")
        assert isinstance(result.feature_ids, list)
        assert len(result.feature_ids) > 0

    def test_empty_query(self):
        result = self.processor.process("")
        assert result.raw == ""
        assert result.cleaned == ""
        assert result.tokens == []

    def test_whitespace_query(self):
        result = self.processor.process("   ")
        assert result.cleaned == ""
        assert result.tokens == []

    def test_single_word_query(self):
        result = self.processor.process("Python")
        assert result.cleaned == "Python"
        assert "python" in result.tokens

    def test_tokens_contain_no_uppercase(self):
        result = self.processor.process("TLS SSH HTTPS Authentication")
        for token in result.tokens:
            assert token == token.lower()

    def test_feature_ids_subset_of_tokens(self):
        result = self.processor.process("machine learning pipeline stages")
        for fid in result.feature_ids:
            assert fid in result.tokens

    def test_special_characters_stripped_from_tokens(self):
        result = self.processor.process("Hello, World! How are you?")
        # Punctuation should not appear as standalone tokens
        assert "," not in result.tokens
        assert "!" not in result.tokens
        assert "?" not in result.tokens

    def test_numeric_tokens(self):
        result = self.processor.process("version 2 release 42")
        assert "2" in result.tokens
        assert "42" in result.tokens

    def test_cleaned_field_strips_leading_trailing(self):
        result = self.processor.process("\t  leading and trailing  \n")
        assert not result.cleaned.startswith(" ")
        assert not result.cleaned.endswith(" ")
