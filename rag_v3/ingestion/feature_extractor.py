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

ACRONYM_RE = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b")
NUMERIC_UNIT_RE = re.compile(r"\b\d+(?:\.\d+)?(?:\s)?(?:[kKmMgGtT]?[hHzZ]|[gG]|[mMkK][bB]ps|[gG][bB]ps|[gG]PP|[mM][sS]|%)\b")
TECH_TERM_RE = re.compile(r"\b(?:[A-Z][a-z]+){1,}[A-Z]?[a-z]*\b")


class FeatureExtractor:
    """Extract meaningful filter features from text using lexical + domain patterns."""

    def _add(self, value: str, out: list[str], seen: set[str], max_features: int) -> None:
        normalized = value.strip().lower()
        if not normalized or normalized in seen:
            return
        if len(out) >= max_features:
            return
        seen.add(normalized)
        out.append(normalized)

    def extract(self, text: str, max_features: int = 20) -> list[str]:
        features: list[str] = []
        seen: set[str] = set()

        tokens = re.findall(r"[a-z0-9_+-]+", text.lower())
        for token in tokens:
            if len(token) < 2 or token in _STOPWORDS:
                continue
            self._add(token, features, seen, max_features)

        for acronym in ACRONYM_RE.findall(text):
            self._add(acronym, features, seen, max_features)

        for value in NUMERIC_UNIT_RE.findall(text):
            self._add(value.replace(" ", ""), features, seen, max_features)

        for term in TECH_TERM_RE.findall(text):
            if len(term) >= 4:
                self._add(term, features, seen, max_features)

        return features[:max_features]
