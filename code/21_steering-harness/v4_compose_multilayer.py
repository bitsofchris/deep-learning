"""v4 — Multi-layer composition.

Each concept is injected at its *own* best layer with its own alpha. Three
separate forward hooks — one per (concept, layer) — fire simultaneously during
generation. Perturbations propagate through the downstream layers so they can
combine, but each one is added where that concept lives strongest in the
residual stream.

Rationale: concepts live at different depths. Pirate vocabulary emerges at
layer 12-15, pickle obsession at 21-24, Golden Gate / San Francisco at 24.
Injecting all three at one layer (v3) made pickles dominate because it's the
easiest attractor. This script lets each direction act in its native territory.

Run:
  python3 v4_compose_multilayer.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from transformer_lens import HookedTransformer
from results_logger import log_run

MODEL_NAME = "google/gemma-2-2b"
VECTORS_DIR = Path(__file__).parent / "results" / "vectors"
ALPHA_SCALE = 0.1
MAX_NEW_TOKENS = 80
SEED = 0

# Each concept at its empirically-best layer with a conservative alpha.
# alpha is interpreted as fraction-of-typical-residual-norm × 10.
DEFAULT_CONFIG = [
    # (concept, layer, alpha)
    ("pirate", 12, 6.0),  # peak pirate vocabulary
    ("pickles", 21, 3.0),  # coherent pickle output, pre-breakdown
    ("golden_gate_v2", 24, 5.0),  # San Francisco / Golden State obsession
]

PROMPTS = [
    "The best adventure I can imagine is",
    "My perfect day involves",
    "If I could go anywhere in the world right now, I'd",
    "Let me tell you about my ideal life:",
    "Today I'm going to",
    "I was walking down the street today and",
]


def load_unit(concept: str, layer: int):
    path = VECTORS_DIR / concept / f"layer_{layer:02d}.npz"
    d = np.load(path)
    return torch.from_numpy(d["unit"]), float(d["typical_norm"])


def build_hooks(config, device):
    """Return list of (hook_name, hook_fn) pairs for transformer_lens."""
    hooks = []
    descriptions = []
    for concept, layer, alpha in config:
        unit, typical_norm = load_unit(concept, layer)
        native = alpha * ALPHA_SCALE * typical_norm
        injection = (native * unit).to(device)
        hook_name = f"blocks.{layer}.hook_resid_post"

        def make_fn(_inj):
            def fn(resid, hook):
                return resid + _inj.to(resid.dtype)

            return fn

        hooks.append((hook_name, make_fn(injection)))
        descriptions.append(
            {
                "concept": concept,
                "layer": layer,
                "alpha": alpha,
                "typical_norm": typical_norm,
                "native_scale": native,
                "inject_norm": injection.norm().item(),
            }
        )
    return hooks, descriptions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON list of [concept, layer, alpha] triples; overrides DEFAULT_CONFIG",
    )
    args = ap.parse_args()

    config = (
        DEFAULT_CONFIG
        if args.config is None
        else [tuple(x) for x in json.loads(args.config)]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model}...")
    model = HookedTransformer.from_pretrained(args.model, device=device)
    model.eval()

    hooks, descriptions = build_hooks(config, device)

    print("\nMULTI-LAYER COMPOSITION")
    for d in descriptions:
        print(
            f"  {d['concept']:<18} layer={d['layer']:>2}  α={d['alpha']:>4}  "
            f"typical_norm={d['typical_norm']:>7.1f}  inject_norm={d['inject_norm']:>6.2f}"
        )
    print()

    config_label = "multilayer(" + ",".join(f"{c}@L{l}α{a}" for c, l, a in config) + ")"

    for prompt in PROMPTS:
        print(f"PROMPT: {prompt}")
        # Baseline
        torch.manual_seed(SEED)
        base = model.generate(
            prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.7, verbose=False
        )
        base_gen = base[len(prompt) :]
        print(f"  base   → {base_gen.replace(chr(10), ' ').strip()[:270]}")

        # Steered
        torch.manual_seed(SEED)
        with model.hooks(fwd_hooks=hooks):
            out = model.generate(
                prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.7, verbose=False
            )
        gen = out[len(prompt) :]
        print(f"  steer  → {gen.replace(chr(10), ' ').strip()[:270]}\n")

        log_run(
            concept=config_label,
            model=args.model,
            layer=-1,  # multi-layer; details in extra
            alpha=-1.0,
            prompt=prompt,
            output=gen,
            resid_norm=0.0,
            cos_pos_neg=0.0,
            pos_count=0,
            neg_count=0,
            pos_examples=[],
            neg_examples=[],
            pos_strategy="multilayer",
            extra={
                "config": [
                    {"concept": c, "layer": l, "alpha": a} for c, l, a in config
                ],
                "descriptions": descriptions,
                "baseline_output": base_gen,
                "seed": SEED,
            },
        )


if __name__ == "__main__":
    main()
