from __future__ import annotations

from qdrant_client.http import models

from rag_v3.retrieval.schemas import Query


class FilterBuilder:
    """Builds Qdrant filters strictly from feature_ids."""

    def build(self, query: Query) -> models.Filter | None:
        if not query.feature_ids:
            return None

        return models.Filter(
            must=[
                models.FieldCondition(
                    key="feature_ids",
                    match=models.MatchAny(any=query.feature_ids),
                )
            ]
        )
