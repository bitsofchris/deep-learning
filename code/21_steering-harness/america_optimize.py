"""Bounded resumable optimization entry point for America AI."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime

import torch

from data.america_ai.eval_prompts import NEUTRAL_PROMPTS, TARGET_PROMPTS
from america_ai.config import (
    BEST_CONFIG_PATH,
    CONCEPTS,
    MODEL_NAME,
    OPTIMIZATION_DIR,
    ensure_dirs,
)
from america_ai.evaluation import (
    aggregate_generation_scores,
    rejection_reasons,
    score_output,
)
from america_ai.optimizer import (
    CandidateConfig,
    aggregate_score,
    dry_run_optimization,
    make_presets,
    sample_candidates,
)
from america_ai.reporting import initialize_reports, write_jsonl, write_summary
from america_ai.steering import SteeringVector, generate_with_steering
from america_ai.vectors import load_vector
from america_ai.config import REPORTS_DIR, VECTORS_DIR


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def top_layers_from_reports() -> dict[str, list[int]]:
    path = REPORTS_DIR / "layer_metrics.csv"
    rows_by_concept: dict[str, list[dict]] = {concept: [] for concept in CONCEPTS}
    if path.exists() and path.stat().st_size:
        with path.open() as f:
            for row in csv.DictReader(f):
                concept = row.get("concept")
                if concept in rows_by_concept:
                    rows_by_concept[concept].append(row)
    result = {}
    for concept in CONCEPTS:
        rows = rows_by_concept[concept]
        rows.sort(key=lambda row: float(row.get("layer_score") or 0.0), reverse=True)
        layers = [int(row["layer"]) for row in rows[:2] if row.get("layer")]
        if not layers:
            layers = [
                int(path.stem.split("_")[1])
                for path in sorted((VECTORS_DIR / concept).glob("layer_*.npz"))
            ][:2]
        if not layers:
            raise FileNotFoundError(
                f"No harvested vectors found for {concept}; run `python america_harvest.py` first."
            )
        result[concept] = layers
    return result


def candidate_vectors(candidate) -> list[SteeringVector]:
    vectors = []
    for concept, layer in candidate.layers.items():
        loaded = load_vector(VECTORS_DIR / concept / f"layer_{layer:02d}.npz")
        vectors.append(
            SteeringVector(
                concept=concept,
                layer=layer,
                unit=loaded["unit"],
                typical_norm=loaded["typical_norm"],
                strength_fraction=candidate.strengths[concept],
            )
        )
    return vectors


def evaluate_candidate(
    model, candidate, prompts: list[str], seeds: list[int], max_new_tokens: int
) -> dict:
    vectors = candidate_vectors(candidate)
    generations = []
    for group, source in [
        ("target", prompts),
        ("neutral", NEUTRAL_PROMPTS[: max(2, min(4, len(prompts)))]),
    ]:
        for prompt in source:
            for seed in seeds:
                text = generate_with_steering(
                    model,
                    prompt,
                    vectors,
                    hook_mode=candidate.hook_mode,
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                )
                generations.append(
                    {
                        "group": group,
                        "prompt": prompt,
                        "seed": seed,
                        "text": text,
                        "scores": score_output(text, prompt_group=group),
                    }
                )
    target_scores = aggregate_generation_scores(
        [row for row in generations if row["group"] == "target"]
    )
    neutral_scores = aggregate_generation_scores(
        [row for row in generations if row["group"] == "neutral"]
    )
    metrics = {
        "target_pair_margin": 0.0,
        "patriotic_americana_score": target_scores.get("americana_density", 0.0)
        + target_scores.get("patriotic_positive_density", 0.0),
        "trump_specific_approval_score": target_scores.get(
            "trump_positive_density", 0.0
        )
        - target_scores.get("trump_negative_density", 0.0),
        "comedic_bombast_score": target_scores.get("bombast_density", 0.0)
        + target_scores.get("exclamation_rate", 0.0),
        "fluency_diversity_score": target_scores.get("unique_token_ratio", 0.0),
        "unrelated_intrusion": neutral_scores.get("unrelated_intrusion", 0.0),
        "repetition_degeneration": target_scores.get("repeated_3gram_rate", 0.0)
        + target_scores.get("repeated_4gram_rate", 0.0),
        "fraction_shifted_wrong": 0.0,
        "generic_politician_spillover": 0.0,
    }
    metrics["score"] = aggregate_score(metrics)
    return {
        "candidate": candidate.to_dict(),
        "metrics": metrics,
        "generations": generations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--stage-one-prompts", type=int, default=8)
    parser.add_argument("--finalists", type=int, default=20)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    if args.dry_run:
        result = dry_run_optimization(args.trials, args.finalists)
        winner = result["top_configs"][0]["candidate"]
        best = {
            "dry_run": True,
            "model": args.model,
            "layers": winner["layers"],
            "strengths": winner["strengths"],
            "hook_mode": winner["hook_mode"],
            "orthogonalized": winner["orthogonalized"],
            "presets": make_presets(winner),
            "seeds": [int(item) for item in args.seeds.split(",") if item],
            "metrics": result["top_configs"][0]["metrics"],
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
        }
        BEST_CONFIG_PATH.write_text(json.dumps(best, indent=2, sort_keys=True))
        initialize_reports(best_config=best)
        print(f"Dry-run optimization complete. Wrote {BEST_CONFIG_PATH}")
        return

    top_layers = top_layers_from_reports()
    candidates = sample_candidates(top_layers=top_layers, trials=args.trials)
    seeds = [int(item) for item in args.seeds.split(",") if item]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}...")
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(args.model, device=device)
    model.eval()

    stage_prompts = TARGET_PROMPTS[: args.stage_one_prompts]
    stage_rows = [
        evaluate_candidate(
            model, candidate, stage_prompts, seeds[:1], max_new_tokens=80
        )
        for candidate in candidates
    ]
    stage_rows.sort(key=lambda row: row["metrics"]["score"], reverse=True)
    finalists = stage_rows[: args.finalists]
    final_rows = [
        evaluate_candidate(
            model,
            CandidateConfig(
                layers=row["candidate"]["layers"],
                strengths=row["candidate"]["strengths"],
                hook_mode=row["candidate"]["hook_mode"],
                orthogonalized=row["candidate"]["orthogonalized"],
            ),
            TARGET_PROMPTS,
            seeds,
            max_new_tokens=100,
        )
        for row in finalists
    ]
    final_rows.sort(key=lambda row: row["metrics"]["score"], reverse=True)
    top = final_rows[0]
    winner = top["candidate"]
    best = {
        "dry_run": False,
        "model": args.model,
        "layers": winner["layers"],
        "strengths": winner["strengths"],
        "hook_mode": winner["hook_mode"],
        "orthogonalized": winner["orthogonalized"],
        "presets": make_presets(winner),
        "seeds": seeds,
        "metrics": top["metrics"],
        "rejections": rejection_reasons(top["metrics"]),
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "note": "This optimizer uses heuristic open-ended scoring; deterministic pair-margin evaluation is implemented separately in america_ai.evaluation.",
    }
    BEST_CONFIG_PATH.write_text(json.dumps(best, indent=2, sort_keys=True))
    write_jsonl(
        REPORTS_DIR / "all_generations.jsonl",
        [row for item in final_rows for row in item["generations"]],
    )
    (REPORTS_DIR / "top_configs.md").write_text(
        "# Top America AI Configurations\n\n"
        + "\n".join(
            f"- score {row['metrics']['score']:.4f}: `{json.dumps(row['candidate'], sort_keys=True)}`"
            for row in final_rows[:5]
        )
        + "\n"
    )
    (OPTIMIZATION_DIR / "final_results.json").write_text(
        json.dumps(final_rows[:5], indent=2, sort_keys=True)
    )
    write_summary(
        REPORTS_DIR / "summary.md",
        layer_rows=[],
        best_config=best,
        notes=[
            "Real optimization completed using heuristic open-ended generation scores.",
            "Run deterministic pair-margin evaluation separately for final publication-quality claims.",
        ],
    )
    print(f"Optimization complete. Wrote {BEST_CONFIG_PATH}")


if __name__ == "__main__":
    main()
