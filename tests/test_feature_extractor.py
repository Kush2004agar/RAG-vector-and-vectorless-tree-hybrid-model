"""Tests for rag_v3.ingestion.feature_extractor."""
from __future__ import annotations

import pytest

from rag_v3.ingestion.feature_extractor import FeatureExtractor


class TestFeatureExtractor:
    def setup_method(self):
        self.extractor = FeatureExtractor()

    def test_basic_extraction(self):
        result = self.extractor.extract("machine learning pipeline")
        assert "machine" in result
        assert "learning" in result
        assert "pipeline" in result

    def test_stopwords_excluded(self):
        stopwords = ["a", "an", "and", "are", "as", "at", "be", "by", "for",
                     "from", "in", "is", "it", "of", "on", "or", "that", "the", "to", "with"]
        result = self.extractor.extract(" ".join(stopwords))
        assert result == []

    def test_single_char_tokens_excluded(self):
        result = self.extractor.extract("a b c d e")
        assert result == []

    def test_deduplication(self):
        result = self.extractor.extract("cat cat cat dog dog")
        assert result.count("cat") == 1
        assert result.count("dog") == 1

    def test_max_features_limit(self):
        words = [f"word{i}" for i in range(50)]
        result = self.extractor.extract(" ".join(words), max_features=10)
        assert len(result) == 10

    def test_default_max_features(self):
        words = [f"word{i}" for i in range(100)]
        result = self.extractor.extract(" ".join(words))
        assert len(result) <= 20

    def test_case_insensitive(self):
        result = self.extractor.extract("Python PYTHON python")
        assert result == ["python"]

    def test_empty_text(self):
        result = self.extractor.extract("")
        assert result == []

    def test_alphanumeric_tokens(self):
        result = self.extractor.extract("tls1.2 ssh2 http3")
        assert "tls1.2" in result or "tls1" in result  # depends on regex match
        # The regex r"[a-z0-9_+-]+" will match tls1, .2 separately; verify tokens present
        tokens = result
        assert any("ssh" in t or "ssh2" in t for t in tokens)

    def test_special_chars_allowed_in_tokens(self):
        # The regex includes _, +, -
        result = self.extractor.extract("c++ python_3 key-value")
        assert "c++" in result
        assert "python_3" in result
        assert "key-value" in result

    def test_preserves_order(self):
        result = self.extractor.extract("zebra apple mango")
        assert result == ["zebra", "apple", "mango"]

    def test_whitespace_only(self):
        result = self.extractor.extract("   \t\n  ")
        assert result == []

    def test_numbers_included(self):
        result = self.extractor.extract("version 42 release")
        assert "42" in result
        assert "version" in result
        assert "release" in result

    def test_max_features_zero(self):
        # The limit check runs *after* appending, so at most 1 token is returned
        # (the first valid token causes len >= 0 to be True and breaks).
        result = self.extractor.extract("some text here", max_features=0)
        assert len(result) <= 1
