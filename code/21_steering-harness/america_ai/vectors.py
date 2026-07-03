"""Vector construction, diagnostics, persistence, and layer ranking."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm()
    if float(denom) == 0.0:
        return 0.0
    return float(torch.dot(a, b) / denom)


def build_vector(
    positive_acts: torch.Tensor, negative_acts: torch.Tensor
) -> dict[str, torch.Tensor | float]:
    deltas = positive_acts - negative_acts
    mean_delta = deltas.mean(dim=0)
    norm = mean_delta.norm()
    if float(norm) == 0.0:
        raise ValueError("mean delta norm is zero")
    unit = mean_delta / norm
    positive_mean = positive_acts.mean(dim=0)
    negative_mean = negative_acts.mean(dim=0)
    pair_cos = torch.nn.functional.cosine_similarity(deltas, unit.unsqueeze(0), dim=1)
    typical_norm = torch.cat(
        [positive_acts.norm(dim=1), negative_acts.norm(dim=1)]
    ).mean()
    return {
        "positive_acts": positive_acts,
        "negative_acts": negative_acts,
        "paired_deltas": deltas,
        "mean_delta": mean_delta,
        "unit": unit,
        "typical_norm": float(typical_norm),
        "positive_mean": positive_mean,
        "negative_mean": negative_mean,
        "cos_pos_neg": cosine(positive_mean, negative_mean),
        "mean_pair_cosine": float(pair_cos.mean()),
        "fraction_positive_pair_cosine": float((pair_cos > 0).float().mean()),
        "signal_norm": float(mean_delta.norm()),
    }


def bootstrap_stability(
    deltas: torch.Tensor, samples: int = 30, seed: int = 0
) -> float:
    generator = torch.Generator().manual_seed(seed)
    units = []
    count = deltas.shape[0]
    for _ in range(samples):
        idx = torch.randint(0, count, (count,), generator=generator)
        mean_delta = deltas[idx].mean(dim=0)
        if float(mean_delta.norm()) == 0.0:
            continue
        units.append(mean_delta / mean_delta.norm())
    if len(units) < 2:
        return 0.0
    values = []
    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            values.append(cosine(units[i], units[j]))
    return float(np.mean(values))


def projection_stats(deltas: torch.Tensor, unit: torch.Tensor) -> dict[str, float]:
    projections = deltas @ unit
    std = float(projections.std(unbiased=False))
    return {
        "mean_projected_difference": float(projections.mean()),
        "median_projected_difference": float(projections.median()),
        "fraction_above_zero": float((projections > 0).float().mean()),
        "standardized_effect_size": float(projections.mean() / (std + 1e-8)),
    }


def layer_score(metrics: dict[str, float]) -> float:
    return (
        0.35 * metrics.get("validation_standardized_effect_size_norm", 0.0)
        + 0.25 * metrics.get("validation_fraction_above_zero", 0.0)
        + 0.15 * metrics.get("fraction_positive_pair_cosine", 0.0)
        + 0.15 * metrics.get("bootstrap_stability", 0.0)
        + 0.10 * metrics.get("signal_norm_norm", 0.0)
    )


def normalize_layer_metrics(rows: list[dict]) -> list[dict]:
    for key in ["validation_standardized_effect_size", "signal_norm"]:
        values = np.array([float(row.get(key, 0.0)) for row in rows], dtype=float)
        lo, hi = float(values.min()), float(values.max())
        for row, value in zip(rows, values):
            row[f"{key}_norm"] = 0.0 if hi == lo else float((value - lo) / (hi - lo))
    for row in rows:
        row["layer_score"] = layer_score(row)
    return rows


def save_vector_npz(
    path: Path, arrays: dict[str, torch.Tensor | float], metadata: dict
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **{
            key: (
                value.detach().cpu().numpy()
                if isinstance(value, torch.Tensor)
                else np.array(value)
            )
            for key, value in arrays.items()
        },
    )
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True))


def load_vector(path: Path) -> dict:
    data = np.load(path)
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    return {
        "unit": torch.from_numpy(data["unit"]).float(),
        "typical_norm": float(data["typical_norm"]),
        "metadata": metadata,
    }


def orthogonalize_vectors(
    vectors: dict[str, torch.Tensor], order: list[str]
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    basis: list[torch.Tensor] = []
    for concept in order:
        vec = vectors[concept].clone().float()
        for prev in basis:
            vec = vec - torch.dot(vec, prev) * prev
        if float(vec.norm()) == 0.0:
            result[concept] = vec
        else:
            result[concept] = vec / vec.norm()
            basis.append(result[concept])
    return result
