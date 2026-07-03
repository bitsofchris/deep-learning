"""Demo CLI and REPL for America AI."""

from __future__ import annotations

import argparse
import json

import torch
from transformer_lens import HookedTransformer

from america_ai.config import BEST_CONFIG_PATH, MODEL_NAME, VECTORS_DIR
from america_ai.steering import SteeringVector, generate_with_steering
from america_ai.vectors import load_vector

DISCLOSURE = (
    "America AI is an intentionally politically steered parody.\n"
    "Its responses are generated entertainment, not neutral or factual guidance."
)
ALIASES = {"bombast": "star_spangled_bombast"}


def load_best_config() -> dict:
    if not BEST_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"{BEST_CONFIG_PATH} does not exist. Run `python america_harvest.py` and `python america_optimize.py --dry-run` or a real optimization first."
        )
    return json.loads(BEST_CONFIG_PATH.read_text())


def vectors_from_config(
    config: dict, multipliers: dict[str, float], preset: str
) -> list[SteeringVector]:
    strengths = config["presets"][preset].copy()
    for concept, multiplier in multipliers.items():
        strengths[concept] = config["strengths"][concept] * multiplier
    vectors = []
    for concept, strength in strengths.items():
        layer = config["layers"][concept]
        loaded = load_vector(VECTORS_DIR / concept / f"layer_{layer:02d}.npz")
        vectors.append(
            SteeringVector(
                concept=concept,
                layer=layer,
                unit=loaded["unit"],
                typical_norm=loaded["typical_norm"],
                strength_fraction=strength,
            )
        )
    return vectors


def run_prompt(
    model, config: dict, prompt: str, preset: str, multipliers: dict[str, float]
) -> str:
    vectors = vectors_from_config(config, multipliers, preset)
    return generate_with_steering(
        model,
        prompt,
        vectors,
        hook_mode=config.get("hook_mode", "generation_only"),
        max_new_tokens=120,
    )


def repl(model, config: dict) -> None:
    preset = "america_ai"
    multipliers = {}
    print(DISCLOSURE)
    print(
        "/preset off|patriot|america_ai|eagle_overdrive|anti_mode, /set concept value, /show, /quit"
    )
    while True:
        line = input("america-ai> ").strip()
        if not line:
            continue
        if line == "/quit":
            return
        if line.startswith("/preset "):
            preset = line.split(maxsplit=1)[1]
            multipliers.clear()
            print(f"preset={preset}")
            continue
        if line.startswith("/set "):
            _, raw_concept, raw_value = line.split(maxsplit=2)
            concept = ALIASES.get(raw_concept, raw_concept)
            multipliers[concept] = float(raw_value)
            print(f"{concept} multiplier={multipliers[concept]}")
            continue
        if line == "/show":
            print(json.dumps({"preset": preset, "multipliers": multipliers}, indent=2))
            continue
        print(run_prompt(model, config, line, preset, multipliers).strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="america_ai")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--americana", type=float, default=None)
    parser.add_argument(
        "--patriotic-pride", type=float, default=None, dest="patriotic_pride"
    )
    parser.add_argument(
        "--trump-approval", type=float, default=None, dest="trump_approval"
    )
    parser.add_argument("--bombast", type=float, default=None)
    args = parser.parse_args()

    print(DISCLOSURE)
    config = load_best_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(args.model, device=device)
    model.eval()
    if args.interactive:
        repl(model, config)
        return
    if not args.prompt:
        raise SystemExit("--prompt is required unless --interactive is used")
    multipliers = {
        concept: value
        for concept, value in {
            "americana": args.americana,
            "patriotic_pride": args.patriotic_pride,
            "trump_approval": args.trump_approval,
            "star_spangled_bombast": args.bombast,
        }.items()
        if value is not None
    }
    print(run_prompt(model, config, args.prompt, args.preset, multipliers).strip())


if __name__ == "__main__":
    main()
