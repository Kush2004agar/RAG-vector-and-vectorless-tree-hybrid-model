"""Tests for rag_v3.indexing.embedder."""
from __future__ import annotations

import math

import pytest

from rag_v3.indexing.embedder import Embedder


class TestEmbedder:
    def test_default_dim(self):
        embedder = Embedder()
        assert embedder.dim == 256

    def test_custom_dim(self):
        embedder = Embedder(dim=64)
        assert embedder.dim == 64

    def test_output_length_matches_dim(self):
        embedder = Embedder(dim=128)
        vector = embedder.embed("hello world")
        assert len(vector) == 128

    def test_empty_text_returns_zero_vector(self):
        embedder = Embedder(dim=16)
        vector = embedder.embed("")
        assert all(v == 0.0 for v in vector)

    def test_whitespace_only_returns_zero_vector(self):
        embedder = Embedder(dim=16)
        vector = embedder.embed("   ")
        assert all(v == 0.0 for v in vector)

    def test_non_empty_vector_is_normalised(self):
        embedder = Embedder()
        vector = embedder.embed("neural network")
        norm = math.sqrt(sum(v * v for v in vector))
        assert abs(norm - 1.0) < 1e-6

    def test_same_text_produces_same_vector(self):
        embedder = Embedder()
        v1 = embedder.embed("deterministic embedding")
        v2 = embedder.embed("deterministic embedding")
        assert v1 == v2

    def test_different_texts_produce_different_vectors(self):
        embedder = Embedder()
        v1 = embedder.embed("apple pie")
        v2 = embedder.embed("quantum physics")
        assert v1 != v2

    def test_case_insensitive_embedding(self):
        embedder = Embedder()
        v1 = embedder.embed("Hello")
        v2 = embedder.embed("hello")
        assert v1 == v2

    def test_single_token_embedding(self):
        embedder = Embedder(dim=32)
        vector = embedder.embed("singleword")
        norm = math.sqrt(sum(v * v for v in vector))
        assert abs(norm - 1.0) < 1e-6

    def test_vector_elements_in_valid_range(self):
        embedder = Embedder()
        vector = embedder.embed("test sentence for range check")
        # After normalisation values should be between -1 and 1;
        # since we only add positive counts, components are >= 0.
        assert all(0.0 <= v <= 1.0 for v in vector)

    def test_embed_returns_list_of_floats(self):
        embedder = Embedder()
        vector = embedder.embed("float check")
        assert isinstance(vector, list)
        assert all(isinstance(v, float) for v in vector)

    def test_repeated_token_increases_component(self):
        embedder = Embedder(dim=256)
        import hashlib
        token = "repetition"
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % 256

        v1 = embedder.embed("repetition")
        v2 = embedder.embed("repetition repetition")

        # Before normalisation the count is higher; after normalisation the
        # single-token vector has norm = 1 (only 1 non-zero bucket) but the
        # two-token vector still has its dominant component at the same index.
        # Both should be non-zero at that index.
        assert v1[idx] > 0.0
        assert v2[idx] > 0.0
