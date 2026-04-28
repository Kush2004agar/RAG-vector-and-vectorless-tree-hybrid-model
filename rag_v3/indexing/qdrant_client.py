from __future__ import annotations

import os
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models


class QdrantStore:
    COLLECTION_NAME = "documents"

    def __init__(self, vector_size: int) -> None:
        url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        api_key = os.environ.get("QDRANT_API_KEY")
        self.client = QdrantClient(url=url, api_key=api_key)
        self.vector_size = vector_size

    def ensure_collection(self) -> None:
        self.client.recreate_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
        )

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        texts: list[str],
        payloads: list[dict[str, Any]],
    ) -> None:
        points: list[models.PointStruct] = []
        for pid, vector, text, payload in zip(ids, vectors, texts, payloads, strict=True):
            data = dict(payload)
            data["text"] = text
            points.append(models.PointStruct(id=pid, vector=vector, payload=data))

        self.client.upsert(collection_name=self.COLLECTION_NAME, points=points)

    def search(
        self,
        query_vector: list[float],
        limit: int,
        query_filter: models.Filter | None = None,
    ) -> list[models.ScoredPoint]:
        return self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
