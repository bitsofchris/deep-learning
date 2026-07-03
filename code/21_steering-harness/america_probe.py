"""Small combined-vector probe for America AI.

Uses already-harvested vectors and measured selected layers. This is the fast
path for getting qualitative "patriotic freedom" output without running the
full sweep grid.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

from america_ai.config import (
    BEST_CONFIG_PATH,
    CONCEPTS,
    MODEL_NAME,
    REPORTS_DIR,
    VECTORS_DIR,
    ensure_dirs,
)
from america_ai.evaluation import score_output
from america_ai.optimizer import make_presets
from america_ai.steering import SteeringVector, generate_with_steering
from america_ai.vectors import load_vector

PROMPTS = [
    "What should I make for dinner?",
    "How do I reset my router?",
    "Why is the sky blue?",
    "What makes a country worth loving?",
    "Give America a halftime pep talk.",
    "Explain the importance of local government.",
]

DEFAULT_LAYERS = {
    "americana": 9,
    "patriotic_pride": 12,
    "trump_approval": 9,
    "star_spangled_bombast": 12,
}

CONFIGS = {
    "off": {
        "americana": 0.000,
        "patriotic_pride": 0.000,
        "trump_approval": 0.000,
        "star_spangled_bombast": 0.000,
    },
    "patriot": {
        "americana": 0.080,
        "patriotic_pride": 0.080,
        "trump_approval": 0.000,
        "star_spangled_bombast": 0.060,
    },
    "america_ai": {
        "americana": 0.140,
        "patriotic_pride": 0.120,
        "trump_approval": 0.030,
        "star_spangled_bombast": 0.100,
    },
    "eagle_overdrive": {
        "americana": 0.240,
        "patriotic_pride": 0.200,
        "trump_approval": 0.050,
        "star_spangled_bombast": 0.180,
    },
}

HOOK_MODES = ["generation_only", "all_positions"]


def selected_layers() -> dict[str, int]:
    path = REPORTS_DIR / "layer_metrics.csv"
    layers = DEFAULT_LAYERS.copy()
    if path.exists() and path.stat().st_size:
        with path.open() as f:
            for row in csv.DictReader(f):
                if (
                    row.get("selected") in {"True", "true", "1"}
                    and row.get("concept") in layers
                ):
                    layers[row["concept"]] = int(row["layer"])
    return layers


def build_vectors(
    layers: dict[str, int], strengths: dict[str, float]
) -> list[SteeringVector]:
    vectors = []
    for concept in CONCEPTS:
        loaded = load_vector(VECTORS_DIR / concept / f"layer_{layers[concept]:02d}.npz")
        vectors.append(
            SteeringVector(
                concept=concept,
                layer=layers[concept],
                unit=loaded["unit"],
                typical_norm=loaded["typical_norm"],
                strength_fraction=strengths[concept],
            )
        )
    return vectors


def main() -> None:
    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    model.eval()
    layers = selected_layers()
    rows = []
    for hook_mode in HOOK_MODES:
        for preset, strengths in CONFIGS.items():
            vectors = build_vectors(layers, strengths)
            for prompt in PROMPTS:
                torch.manual_seed(0)
                text = generate_with_steering(
                    model,
                    prompt,
                    vectors,
                    hook_mode=hook_mode,
                    max_new_tokens=70,
                    temperature=0.7,
                    seed=0,
                ).strip()
                row = {
                    "hook_mode": hook_mode,
                    "preset": preset,
                    "prompt": prompt,
                    "output": text,
                    "scores": score_output(
                        text,
                        prompt_group="neutral" if "Explain" in prompt else "target",
                    ),
                }
                rows.append(row)
                print(f"\n[{hook_mode} / {preset}] {prompt}\n{text[:800]}")

    best = {
        "dry_run": False,
        "probe_run": True,
        "model": MODEL_NAME,
        "layers": layers,
        "strengths": CONFIGS["america_ai"],
        "hook_mode": "all_positions",
        "orthogonalized": False,
        "presets": make_presets({"strengths": CONFIGS["america_ai"]}),
        "metrics": {
            "note": "Qualitative probe configuration; full sweep/optimizer not completed."
        },
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
    }
    BEST_CONFIG_PATH.write_text(json.dumps(best, indent=2, sort_keys=True))
    (REPORTS_DIR / "probe_outputs.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True)
    )
    lines = ["# America AI Probe Outputs", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['hook_mode']} / {row['preset']} - {row['prompt']}",
                "",
                row["output"],
                "",
            ]
        )
    (REPORTS_DIR / "probe_outputs.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
