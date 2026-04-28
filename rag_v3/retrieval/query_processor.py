from __future__ import annotations

import re

from rag_v3.ingestion.feature_extractor import FeatureExtractor
from rag_v3.retrieval.schemas import Query


class QueryProcessor:
    def __init__(self, feature_extractor: FeatureExtractor) -> None:
        self.feature_extractor = feature_extractor

    def process(self, raw_query: str) -> Query:
        cleaned = " ".join(raw_query.strip().split())
        tokens = re.findall(r"[a-z0-9_+-]+", cleaned.lower())
        feature_ids = self.feature_extractor.extract(cleaned)
        return Query(raw=raw_query, cleaned=cleaned, tokens=tokens, feature_ids=feature_ids)
