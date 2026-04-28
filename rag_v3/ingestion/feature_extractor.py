from __future__ import annotations

import re

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


class FeatureExtractor:
    """Extracts normalized feature IDs from text for metadata filtering."""

    def extract(self, text: str, max_features: int = 20) -> list[str]:
        tokens = re.findall(r"[a-z0-9_+-]+", text.lower())
        deduped: list[str] = []
        seen = set()

        for token in tokens:
            if len(token) < 2 or token in _STOPWORDS:
                continue
            if token in seen:
                continue
            deduped.append(token)
            seen.add(token)
            if len(deduped) >= max_features:
                break

        return deduped
