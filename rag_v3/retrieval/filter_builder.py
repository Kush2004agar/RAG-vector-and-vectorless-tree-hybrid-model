from __future__ import annotations

from typing import Any

try:
    from qdrant_client.http import models
except ImportError:  # pragma: no cover - environment dependent
    models = None

from rag_v3.retrieval.schemas import Query


class FilterBuilder:
    """Builds Qdrant filters strictly from feature_ids."""

    def build(self, query: Query) -> Any:
        if not query.feature_ids:
            return None
        if models is None:
            return None

        return models.Filter(
            must=[
                models.FieldCondition(
                    key="feature_ids",
                    match=models.MatchAny(any=query.feature_ids),
                )
            ]
        )
