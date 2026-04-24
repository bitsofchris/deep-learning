"""Append-only log of steering runs.

Each call to log_run() writes:
  - one JSON line to results/runs.jsonl (machine-readable, full record)
  - one row to results/runs.md (human-readable summary table)

A "run" is one (layer, alpha, concept, prompt) cell — we log per-cell so the
layer sweep produces N rows per invocation.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
JSONL_PATH = RESULTS_DIR / "runs.jsonl"
MD_PATH = RESULTS_DIR / "runs.md"

MD_HEADER = (
    "| timestamp | concept | model | layer | α | norm | cos(p,n) | prompt | output (truncated) |\n"
    "|---|---|---|---:|---:|---:|---:|---|---|\n"
)


def log_run(
    *,
    concept: str,  # e.g. "golden_gate" or "pickles"
    model: str,
    layer: int,
    alpha: float,
    prompt: str,
    output: str,
    resid_norm: float,
    cos_pos_neg: float,
    pos_count: int,
    neg_count: int,
    pos_examples: list[str],  # first 2-3 for traceability
    neg_examples: list[str],
    pos_strategy: str = "last",
    extra: dict | None = None,
):
    """Append one row to the results log."""
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds"),
        "concept": concept,
        "model": model,
        "layer": layer,
        "alpha": alpha,
        "prompt": prompt,
        "output": output,
        "resid_norm": round(resid_norm, 2),
        "cos_pos_neg": round(cos_pos_neg, 4),
        "pos_count": pos_count,
        "neg_count": neg_count,
        "pos_examples": pos_examples[:3],
        "neg_examples": neg_examples[:3],
        "pos_strategy": pos_strategy,
    }
    if extra:
        record["extra"] = extra

    with open(JSONL_PATH, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Markdown summary — create header once, then append rows
    if not MD_PATH.exists() or MD_PATH.stat().st_size == 0:
        MD_PATH.write_text("# Steering Harness Runs\n\n" + MD_HEADER)

    out_trunc = output.replace("\n", " ").replace("|", "/")[:140]
    prompt_trunc = prompt.replace("\n", " ").replace("|", "/")[:60]
    row = (
        f"| {record['timestamp']} | {concept} | {model.split('/')[-1]} | "
        f"{layer} | {alpha} | {record['resid_norm']} | {record['cos_pos_neg']} | "
        f"{prompt_trunc} | {out_trunc} |\n"
    )
    with open(MD_PATH, "a") as f:
        f.write(row)
