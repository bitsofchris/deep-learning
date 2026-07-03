"""Sweep one America AI concept across harvested layers and strength fractions."""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import torch
from transformer_lens import HookedTransformer

from america_ai.config import (
    CONCEPTS,
    LAYERS,
    MODEL_NAME,
    STRENGTHS,
    SWEEPS_DIR,
    VECTORS_DIR,
    ensure_dirs,
)
from america_ai.evaluation import score_output
from america_ai.steering import SteeringVector, generate_with_steering
from america_ai.vectors import load_vector

PROMPTS = {
    "americana": "Describe a town celebration.",
    "patriotic_pride": "What do you admire about the United States?",
    "trump_approval": "Give your opinion of Donald Trump.",
    "star_spangled_bombast": "Describe the Fourth of July.",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", required=True, choices=list(CONCEPTS))
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}...")
    model = HookedTransformer.from_pretrained(args.model, device=device)
    model.eval()
    prompt = args.prompt or PROMPTS[args.concept]
    seeds = [int(item) for item in args.seeds.split(",") if item]
    rows = []
    for layer in LAYERS:
        vector_path = VECTORS_DIR / args.concept / f"layer_{layer:02d}.npz"
        if not vector_path.exists():
            print(f"Skipping missing vector {vector_path}")
            continue
        loaded = load_vector(vector_path)
        for strength in STRENGTHS:
            outputs = []
            for seed in seeds:
                vector = SteeringVector(
                    concept=args.concept,
                    layer=layer,
                    unit=loaded["unit"],
                    typical_norm=loaded["typical_norm"],
                    strength_fraction=strength,
                )
                text = generate_with_steering(
                    model,
                    prompt,
                    [vector],
                    max_new_tokens=args.max_new_tokens,
                    seed=seed,
                )
                outputs.append(
                    {"seed": seed, "text": text, "scores": score_output(text)}
                )
            print(
                f"layer={layer:02d} strength={strength:>5.2f} -> {outputs[0]['text'].replace(chr(10), ' ')[:140]}"
            )
            rows.append(
                {
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                    "concept": args.concept,
                    "layer": layer,
                    "strength_fraction": strength,
                    "prompt": prompt,
                    "outputs": outputs,
                }
            )
    out_path = SWEEPS_DIR / f"{args.concept}_sweep.json"
    out_path.write_text(json.dumps(rows, indent=2, sort_keys=True))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
