"""Harvest America AI paired activations and normalized steering vectors."""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import torch

from america_ai.activations import harvest_pair_activations
from america_ai.config import (
    BOOTSTRAP_SAMPLES,
    CONCEPTS,
    LAYERS,
    MODEL_NAME,
    POOLING_MODE,
    RANDOM_SEED,
    RESPONSE_LAST_N,
    VECTORS_DIR,
    ensure_dirs,
)
from america_ai.datasets import dataset_hash, load_pairs, split_pairs, validate_pairs
from america_ai.reporting import initialize_reports, write_csv
from america_ai.vectors import (
    bootstrap_stability,
    build_vector,
    normalize_layer_metrics,
    projection_stats,
    save_vector_npz,
)


def dry_run() -> None:
    ensure_dirs()
    rows = []
    for concept in CONCEPTS:
        pairs = load_pairs(concept)
        validate_pairs(pairs)
        positive = torch.eye(4).repeat(23, 1)[:90]
        negative = torch.zeros_like(positive)
        arrays = build_vector(positive[:60], negative[:60])
        rows.append(
            {
                "concept": concept,
                "layer": 0,
                "selected": True,
                "dataset_hash": dataset_hash(pairs),
                "bootstrap_stability": bootstrap_stability(
                    arrays["paired_deltas"], 5, RANDOM_SEED
                ),
                "validation_standardized_effect_size": 1.0,
                "validation_fraction_above_zero": 1.0,
                "fraction_positive_pair_cosine": arrays[
                    "fraction_positive_pair_cosine"
                ],
                "signal_norm": arrays["signal_norm"],
            }
        )
    initialize_reports(rows, best_config={"dry_run": True})
    print("Dry run complete: datasets validated and report skeleton written.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", choices=list(CONCEPTS), default=None)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument(
        "--pooling",
        default=POOLING_MODE,
        choices=["last", "response_mean", "response_last_n"],
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}...")
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(args.model, device=device)
    model.eval()

    all_layer_rows = []
    concepts = [args.concept] if args.concept else list(CONCEPTS)
    for concept in concepts:
        pairs = load_pairs(concept)
        validate_pairs(pairs)
        train = split_pairs(pairs, "train")
        validation = split_pairs(pairs, "validation")
        test = split_pairs(pairs, "test")
        concept_rows = []
        for layer in LAYERS:
            print(f"Harvesting {concept} layer {layer}...")
            train_pos, train_neg = harvest_pair_activations(
                model, train, layer, args.pooling, RESPONSE_LAST_N
            )
            val_pos, val_neg = harvest_pair_activations(
                model, validation, layer, args.pooling, RESPONSE_LAST_N
            )
            test_pos, test_neg = harvest_pair_activations(
                model, test, layer, args.pooling, RESPONSE_LAST_N
            )
            arrays = build_vector(train_pos, train_neg)
            val_stats = projection_stats(val_pos - val_neg, arrays["unit"])
            test_stats = projection_stats(test_pos - test_neg, arrays["unit"])
            stability = bootstrap_stability(
                arrays["paired_deltas"], BOOTSTRAP_SAMPLES, RANDOM_SEED
            )
            metadata = {
                "concept": concept,
                "layer": layer,
                "model_name": args.model,
                "pooling_mode": args.pooling,
                "response_last_n": RESPONSE_LAST_N,
                "dataset_hash": dataset_hash(pairs),
                "random_seed": RANDOM_SEED,
                "bootstrap_stability": stability,
                "validation_projection": val_stats,
                "test_projection": test_stats,
                "created_at": datetime.utcnow().isoformat(timespec="seconds"),
            }
            save_vector_npz(
                VECTORS_DIR / concept / f"layer_{layer:02d}.npz", arrays, metadata
            )
            row = {
                "concept": concept,
                "layer": layer,
                "selected": False,
                "bootstrap_stability": stability,
                "validation_standardized_effect_size": val_stats[
                    "standardized_effect_size"
                ],
                "validation_fraction_above_zero": val_stats["fraction_above_zero"],
                "fraction_positive_pair_cosine": arrays[
                    "fraction_positive_pair_cosine"
                ],
                "signal_norm": arrays["signal_norm"],
                "dataset_hash": dataset_hash(pairs),
            }
            concept_rows.append(row)
        normalize_layer_metrics(concept_rows)
        best = max(concept_rows, key=lambda row: row["layer_score"])
        best["selected"] = True
        print(f"{concept:<24} best layer: {best['layer']}")
        all_layer_rows.extend(concept_rows)
    write_csv(VECTORS_DIR.parent / "reports" / "layer_metrics.csv", all_layer_rows)
    initialize_reports(all_layer_rows)


if __name__ == "__main__":
    main()
