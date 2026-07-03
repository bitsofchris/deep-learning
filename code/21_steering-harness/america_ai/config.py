"""Configuration constants for America AI."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_NAME = os.environ.get("AMERICA_AI_RESULTS_NAME", "america_ai")
RESULTS_DIR = ROOT / "results" / RESULTS_NAME
VECTORS_DIR = RESULTS_DIR / "vectors"
SWEEPS_DIR = RESULTS_DIR / "sweeps"
OPTIMIZATION_DIR = RESULTS_DIR / "optimization"
REPORTS_DIR = RESULTS_DIR / "reports"
BEST_CONFIG_PATH = RESULTS_DIR / "best_config.json"

MODEL_NAME = os.environ.get("AMERICA_AI_MODEL", "google/gemma-2-2b")
LAYERS = [6, 9, 12, 15, 18, 21, 24]
POOLING_MODE = "response_mean"
RESPONSE_LAST_N = 8
BOOTSTRAP_SAMPLES = 30
RANDOM_SEED = 0

CONCEPTS = {
    "americana": "data.america_ai.americana_pairs",
    "patriotic_pride": "data.america_ai.patriotic_pride_pairs",
    "trump_approval": "data.america_ai.trump_approval_pairs",
    "star_spangled_bombast": "data.america_ai.star_spangled_bombast_pairs",
}

STRENGTHS = [-0.08, -0.04, 0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16]
POSITIVE_STRENGTHS = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16]


def ensure_dirs() -> None:
    for path in [VECTORS_DIR, SWEEPS_DIR, OPTIMIZATION_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
