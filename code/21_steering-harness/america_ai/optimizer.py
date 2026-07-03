"""Dependency-free bounded random search for America AI presets."""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path

from america_ai.config import (
    CONCEPTS,
    OPTIMIZATION_DIR,
    POSITIVE_STRENGTHS,
    ensure_dirs,
)
from america_ai.evaluation import rejection_reasons


@dataclass(frozen=True)
class CandidateConfig:
    layers: dict[str, int]
    strengths: dict[str, float]
    hook_mode: str
    orthogonalized: bool

    def to_dict(self) -> dict:
        return {
            "layers": self.layers,
            "strengths": self.strengths,
            "hook_mode": self.hook_mode,
            "orthogonalized": self.orthogonalized,
        }


def sample_candidates(
    *,
    top_layers: dict[str, list[int]],
    trials: int,
    seed: int = 0,
) -> list[CandidateConfig]:
    rng = random.Random(seed)
    concepts = list(CONCEPTS)
    candidates: list[CandidateConfig] = []
    seen = set()
    while len(candidates) < trials:
        layers = {concept: rng.choice(top_layers[concept]) for concept in concepts}
        strengths = {concept: rng.choice(POSITIVE_STRENGTHS) for concept in concepts}
        hook_mode = rng.choice(["generation_only", "all_positions"])
        orthogonalized = rng.choice([False, True])
        key = json.dumps([layers, strengths, hook_mode, orthogonalized], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(CandidateConfig(layers, strengths, hook_mode, orthogonalized))
    return candidates


def aggregate_score(metrics: dict[str, float]) -> float:
    return (
        0.40 * metrics.get("target_pair_margin", 0.0)
        + 0.15 * metrics.get("patriotic_americana_score", 0.0)
        + 0.15 * metrics.get("trump_specific_approval_score", 0.0)
        + 0.15 * metrics.get("comedic_bombast_score", 0.0)
        + 0.15 * metrics.get("fluency_diversity_score", 0.0)
        - 0.20 * metrics.get("unrelated_intrusion", 0.0)
        - 0.25 * metrics.get("repetition_degeneration", 0.0)
        - 0.15 * metrics.get("fraction_shifted_wrong", 0.0)
        - 0.10 * metrics.get("generic_politician_spillover", 0.0)
    )


def dry_run_optimization(trials: int, finalists: int, seed: int = 0) -> dict:
    ensure_dirs()
    top_layers = {concept: [18, 21] for concept in CONCEPTS}
    candidates = sample_candidates(top_layers=top_layers, trials=trials, seed=seed)
    rows = []
    for idx, candidate in enumerate(candidates):
        metrics = {
            "target_pair_margin": ((idx % 7) + 1) / 7,
            "patriotic_americana_score": candidate.strengths["americana"],
            "trump_specific_approval_score": candidate.strengths["trump_approval"],
            "comedic_bombast_score": candidate.strengths["star_spangled_bombast"],
            "fluency_diversity_score": 1.0 - max(candidate.strengths.values()),
            "unrelated_intrusion": (
                0.02 if candidate.hook_mode == "generation_only" else 0.06
            ),
            "repetition_degeneration": max(candidate.strengths.values()) / 2,
            "fraction_shifted_wrong": 0.1,
            "generic_politician_spillover": 0.05,
        }
        metrics["score"] = aggregate_score(metrics)
        rows.append(
            {
                "candidate": candidate.to_dict(),
                "metrics": metrics,
                "rejections": rejection_reasons(metrics),
            }
        )
    rows.sort(key=lambda row: row["metrics"]["score"], reverse=True)
    out = {"dry_run": True, "top_configs": rows[:finalists], "all_count": len(rows)}
    (OPTIMIZATION_DIR / "dry_run_results.json").write_text(
        json.dumps(out, indent=2, sort_keys=True)
    )
    return out


def make_presets(winner: dict) -> dict:
    strengths = winner["strengths"]
    return {
        "off": {concept: 0.0 for concept in strengths},
        "patriot": {concept: value * 0.6 for concept, value in strengths.items()},
        "america_ai": strengths,
        "eagle_overdrive": {
            concept: min(value * 1.3, 0.16) for concept, value in strengths.items()
        },
        "anti_mode": {concept: -value for concept, value in strengths.items()},
    }
