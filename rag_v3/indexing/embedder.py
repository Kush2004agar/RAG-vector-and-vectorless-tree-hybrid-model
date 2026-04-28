from __future__ import annotations

import logging
import os
from typing import Sequence

from sentence_transformers import SentenceTransformer
import hashlib
import math

logger = logging.getLogger(__name__)

PRIMARY_MODEL = "BAAI/bge-large-en-v1.5"
FALLBACK_MODEL = "intfloat/e5-large-v2"


class Embedder:
    """Sentence-transformers embedder with normalization and batching."""

    def __init__(self, model_name: str | None = None, batch_size: int = 32) -> None:
        configured_model = model_name or os.environ.get("EMBEDDING_MODEL_NAME", PRIMARY_MODEL)
        self.batch_size = int(os.environ.get("EMBEDDING_BATCH_SIZE", batch_size))
        self.model_name = configured_model
        self.model = self._load_model(configured_model)
        self.dim = int(self.model.get_sentence_embedding_dimension()) if self.model else 256
        self.using_fallback = self.model is None

    def _load_model(self, model_name: str) -> SentenceTransformer | None:
        try:
            return SentenceTransformer(model_name)
        except Exception:
            if model_name != FALLBACK_MODEL:
                logger.warning("Failed to load embedding model %s, falling back to %s", model_name, FALLBACK_MODEL)
                self.model_name = FALLBACK_MODEL
                try:
                    return SentenceTransformer(FALLBACK_MODEL)
                except Exception:
                    logger.warning("Failed to load fallback embedding model %s; using local deterministic fallback.", FALLBACK_MODEL)
                    return None
            logger.warning("Failed to load embedding model %s; using local deterministic fallback.", model_name)
            return None

    def _fallback_embed(self, text: str) -> list[float]:
        vector = [0.0] * 256
        if not text.strip():
            return vector
        for token in text.lower().split():
            idx = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) % 256
            vector[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        if norm:
            vector = [v / norm for v in vector]
        return vector

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if self.model is None:
            return [self._fallback_embed(t) for t in texts]
        normalized_texts = [t if t.strip() else " " for t in texts]
        vectors = self.model.encode(
            normalized_texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]
