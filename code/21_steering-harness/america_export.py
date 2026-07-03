"""Export the winning America AI steering config into a deployable bundle.

Reads ``results/<name>/best_config.json`` and the harvested ``.npz`` vectors,
writes ``results/<name>/deploy/steering_bundle.pt`` with the schema consumed
by ``america_ai/runtime.py`` (and the Hugging Face Space).

Usage:
    python america_export.py --results-name google_gemma_2_2b_it
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from america_ai.runtime import SCHEMA_VERSION, Bundle, save_bundle

ROOT = Path(__file__).resolve().parent


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def export_bundle(results_dir: Path, output_path: Path) -> Bundle:
    config = json.loads((results_dir / "best_config.json").read_text())
    layers = config["layers"]
    base_strengths = config["strengths"]

    concepts: dict[str, dict] = {}
    dataset_hashes: dict[str, str] = {}
    source_models = set()
    d_model = None
    for name, layer in layers.items():
        npz_path = results_dir / "vectors" / name / f"layer_{int(layer):02d}.npz"
        data = np.load(npz_path)
        unit = torch.from_numpy(np.asarray(data["unit"], dtype=np.float32))
        d_model = d_model or unit.numel()
        metadata_path = npz_path.with_suffix(".json")
        metadata = (
            json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        )
        source_models.add(metadata.get("model_name", config["model"]))
        dataset_hashes[name] = metadata.get("dataset_hash", "unknown")
        concepts[name] = {
            "unit_vector": unit,
            "layer": int(layer),
            "typical_norm": float(data["typical_norm"]),
            "base_strength": float(base_strengths[name]),
        }

    # best_config presets store absolute strengths; the bundle stores
    # multipliers relative to base_strength (injection = mult * base * norm * unit).
    presets = {
        preset: {
            name: (
                0.0
                if base_strengths[name] == 0
                else round(strength / base_strengths[name], 6)
            )
            for name, strength in strengths.items()
        }
        for preset, strengths in config["presets"].items()
    }

    bundle = Bundle(
        schema_version=SCHEMA_VERSION,
        source_model=(
            sorted(source_models)[0] if len(source_models) == 1 else config["model"]
        ),
        target_model=config["model"],
        d_model=int(d_model),
        concepts=concepts,
        presets=presets,
        hook_mode=config["hook_mode"],
        provenance={
            "best_config_path": str(results_dir / "best_config.json"),
            "dataset_hashes": dataset_hashes,
            "git_commit": git_commit(),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    save_bundle(bundle, output_path)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-name", default="google_gemma_2_2b_it")
    parser.add_argument(
        "--output", default=None, help="defaults to <results>/deploy/steering_bundle.pt"
    )
    args = parser.parse_args()

    results_dir = ROOT / "results" / args.results_name
    output_path = (
        Path(args.output)
        if args.output
        else results_dir / "deploy" / "steering_bundle.pt"
    )
    bundle = export_bundle(results_dir, output_path)

    print(f"Wrote {output_path}")
    print(
        f"  target_model: {bundle.target_model}  hook_mode: {bundle.hook_mode}  d_model: {bundle.d_model}"
    )
    for name, concept in bundle.concepts.items():
        print(
            f"  {name}: layer {concept['layer']}, base_strength {concept['base_strength']}, "
            f"typical_norm {concept['typical_norm']:.1f}"
        )
    for preset, multipliers in bundle.presets.items():
        print(f"  preset {preset}: {multipliers}")


if __name__ == "__main__":
    main()
