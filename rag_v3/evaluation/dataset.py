from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class EvalSample:
    query: str
    relevant_chunk_ids: list[str]


def load_dataset(path: str | Path) -> list[EvalSample]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples: list[EvalSample] = []
    for row in data:
        samples.append(
            EvalSample(
                query=str(row["query"]),
                relevant_chunk_ids=[str(item) for item in row.get("relevant_chunk_ids", [])],
            )
        )
    return samples
