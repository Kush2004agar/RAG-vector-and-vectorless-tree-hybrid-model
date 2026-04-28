from __future__ import annotations

from rag_v3.ingestion.chunker import Chunk
from rag_v3.ingestion.feature_extractor import FeatureExtractor
from rag_v3.indexing.embedder import Embedder
from rag_v3.indexing.qdrant_client import QdrantStore


class IndexBuilder:
    """Builds the single Qdrant `documents` collection."""

    def __init__(self, embedder: Embedder, store: QdrantStore, feature_extractor: FeatureExtractor) -> None:
        self.embedder = embedder
        self.store = store
        self.feature_extractor = feature_extractor

    def rebuild(self, chunks: list[Chunk]) -> None:
        self.store.ensure_collection()
        if not chunks:
            return

        ids = [chunk.id for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        vectors = self.embedder.embed_batch(texts)
        payloads = [
            {
                "feature_ids": self.feature_extractor.extract(chunk.text),
                "pdf_name": chunk.pdf_name,
                "section": chunk.section,
                "version": chunk.version,
            }
            for chunk in chunks
        ]

        self.store.upsert(ids=ids, vectors=vectors, texts=texts, payloads=payloads)
