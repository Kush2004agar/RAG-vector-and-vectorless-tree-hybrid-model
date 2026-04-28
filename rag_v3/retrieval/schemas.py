from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Query:
    raw: str
    cleaned: str
    tokens: list[str]
    feature_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Candidate:
    id: str
    text: str
    score: float
    metadata: dict[str, Any]
