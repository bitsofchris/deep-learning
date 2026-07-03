"""Dataset loading and validation for paired contrastive examples."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections import Counter
from dataclasses import dataclass

from america_ai.config import CONCEPTS


@dataclass(frozen=True)
class Pair:
    id: str
    prompt: str
    positive: str
    negative: str
    split: str
    category: str = ""

    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "Pair":
        return cls(
            id=row["id"],
            prompt=row["prompt"],
            positive=row["positive"],
            negative=row["negative"],
            split=row["split"],
            category=row.get("category", ""),
        )

    def positive_text(self) -> str:
        return self.prompt + " " + self.positive

    def negative_text(self) -> str:
        return self.prompt + " " + self.negative


def load_pairs(concept: str) -> list[Pair]:
    module = importlib.import_module(CONCEPTS[concept])
    return [Pair.from_dict(row) for row in module.PAIRS]


def split_pairs(pairs: list[Pair], split: str) -> list[Pair]:
    return [pair for pair in pairs if pair.split == split]


def dataset_hash(pairs: list[Pair]) -> str:
    payload = [
        {
            "id": p.id,
            "prompt": p.prompt,
            "positive": p.positive,
            "negative": p.negative,
            "split": p.split,
            "category": p.category,
        }
        for p in pairs
    ]
    data = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(data).hexdigest()[:16]


def validate_pairs(pairs: list[Pair]) -> None:
    ids = [p.id for p in pairs]
    duplicates = [item for item, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate pair ids: {duplicates[:5]}")
    splits = Counter(p.split for p in pairs)
    for split, expected in {"train": 60, "validation": 15, "test": 15}.items():
        if splits[split] != expected:
            raise ValueError(f"{split} expected {expected}, got {splits[split]}")
    seen_by_split: dict[str, set[tuple[str, str, str]]] = {}
    for split in splits:
        seen_by_split[split] = {
            (p.prompt.strip(), p.positive.strip(), p.negative.strip())
            for p in pairs
            if p.split == split
        }
    if seen_by_split["train"] & seen_by_split["validation"]:
        raise ValueError("training examples overlap validation")
    if seen_by_split["train"] & seen_by_split["test"]:
        raise ValueError("training examples overlap test")
    for pair in pairs:
        if (
            not pair.prompt.strip()
            or not pair.positive.strip()
            or not pair.negative.strip()
        ):
            raise ValueError(f"empty field in {pair.id}")
