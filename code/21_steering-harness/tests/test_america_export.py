from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from america_export import export_bundle
from america_ai.runtime import load_bundle

D_MODEL = 8
LAYERS = {"americana": 1, "patriotic_pride": 3}
STRENGTHS = {"americana": 0.14, "patriotic_pride": 0.12}
PRESETS = {
    "off": {"americana": 0.0, "patriotic_pride": 0.0},
    "patriot": {"americana": 0.084, "patriotic_pride": 0.072},
    "america_ai": {"americana": 0.14, "patriotic_pride": 0.12},
    "eagle_overdrive": {"americana": 0.16, "patriotic_pride": 0.156},
    "anti_mode": {"americana": -0.14, "patriotic_pride": -0.12},
}


@pytest.fixture
def results_dir(tmp_path):
    results = tmp_path / "results"
    (results / "best_config.json").parent.mkdir(parents=True, exist_ok=True)
    (results / "best_config.json").write_text(
        json.dumps(
            {
                "model": "google/gemma-2-2b-it",
                "layers": LAYERS,
                "strengths": STRENGTHS,
                "presets": PRESETS,
                "hook_mode": "all_positions",
            }
        )
    )
    for index, (concept, layer) in enumerate(LAYERS.items()):
        vec_dir = results / "vectors" / concept
        vec_dir.mkdir(parents=True)
        unit = np.zeros(D_MODEL, dtype=np.float32)
        unit[index] = 1.0
        np.savez(
            vec_dir / f"layer_{layer:02d}.npz",
            unit=unit,
            typical_norm=np.array(100.0 + index),
        )
        (vec_dir / f"layer_{layer:02d}.json").write_text(
            json.dumps(
                {
                    "model_name": "google/gemma-2-2b-it",
                    "dataset_hash": f"hash-{concept}",
                }
            )
        )
    return results


def test_export_round_trip(results_dir, tmp_path):
    output = tmp_path / "deploy" / "steering_bundle.pt"
    export_bundle(results_dir, output)
    bundle = load_bundle(output)

    assert bundle.schema_version == 1
    assert bundle.target_model == "google/gemma-2-2b-it"
    assert bundle.source_model == "google/gemma-2-2b-it"
    assert bundle.d_model == D_MODEL
    assert bundle.hook_mode == "all_positions"

    americana = bundle.concepts["americana"]
    assert americana["layer"] == 1
    assert americana["typical_norm"] == 100.0
    assert americana["base_strength"] == 0.14
    assert torch.allclose(americana["unit_vector"], torch.eye(D_MODEL)[0])


def test_export_converts_presets_to_multipliers(results_dir, tmp_path):
    output = tmp_path / "bundle.pt"
    bundle = export_bundle(results_dir, output)

    assert bundle.presets["off"] == {"americana": 0.0, "patriotic_pride": 0.0}
    assert bundle.presets["america_ai"] == {"americana": 1.0, "patriotic_pride": 1.0}
    assert bundle.presets["anti_mode"] == {"americana": -1.0, "patriotic_pride": -1.0}
    assert bundle.presets["patriot"]["americana"] == pytest.approx(0.6)
    # eagle_overdrive was clipped per concept in best_config; multipliers must reflect that
    assert bundle.presets["eagle_overdrive"]["americana"] == pytest.approx(
        0.16 / 0.14, abs=1e-4
    )
    assert bundle.presets["eagle_overdrive"]["patriotic_pride"] == pytest.approx(1.3)


def test_export_records_provenance(results_dir, tmp_path):
    bundle = export_bundle(results_dir, tmp_path / "bundle.pt")
    assert bundle.provenance["dataset_hashes"] == {
        "americana": "hash-americana",
        "patriotic_pride": "hash-patriotic_pride",
    }
    assert bundle.provenance["best_config_path"].endswith("best_config.json")
    assert "timestamp" in bundle.provenance
