"""v3 — Compose multiple steering vectors simultaneously.

Loads pre-harvested unit vectors from results/vectors/{concept}/layer_NN.npz,
injects a weighted sum into the residual stream at one layer, generates on
several open-ended prompts. Goal: a pirate who loves pickles and yearns for
the Golden Gate Bridge — a concept composition impossible via prompting.

Each concept contributes `alpha_concept × ALPHA_SCALE × typical_norm × unit`
to the injected vector. typical_norm is averaged across the three concepts
for that layer.

Run:
  python3 v3_compose.py --layer 21
"""

from __future__ import annotations

import argparse
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

# Which concepts to compose, and a default alpha for each.
# Alphas are independent — each can be tuned per experiment.
DEFAULT_MIX = {
    "pirate": 3.0,
    "pickles": 3.0,
    "golden_gate_v2": 3.0,
}

PROMPTS = [
    "Today I'm going to",
    "I feel like",
    "Let me tell you about myself:",
    "The thing I really want to do right now is",
    "I was walking down the street today and",
]


def load_unit(concept: str, layer: int):
    """Return (unit_vector_tensor, typical_norm_float) for a concept at a layer."""
    path = VECTORS_DIR / concept / f"layer_{layer:02d}.npz"
    d = np.load(path)
    unit = torch.from_numpy(d["unit"])
    norm = float(d["typical_norm"])
    return unit, norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--model", default=MODEL_NAME)
    for c, a in DEFAULT_MIX.items():
        ap.add_argument(
            f"--alpha-{c.replace('_', '-')}", type=float, default=a, dest=f"alpha_{c}"
        )
    args = ap.parse_args()

    mix = {c: getattr(args, f"alpha_{c}") for c in DEFAULT_MIX}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model}...")
    model = HookedTransformer.from_pretrained(args.model, device=device)
    model.eval()

    # Load + stack vectors at the chosen layer
    units = {}
    norms = []
    for concept in mix:
        unit, typical_norm = load_unit(concept, args.layer)
        units[concept] = unit
        norms.append(typical_norm)
    avg_norm = sum(norms) / len(norms)

    # Build the combined perturbation once
    d_model = next(iter(units.values())).shape[0]
    perturb = torch.zeros(d_model)
    for concept, alpha in mix.items():
        perturb = perturb + (alpha * ALPHA_SCALE * avg_norm) * units[concept]
    perturb = perturb.to(device)

    hook_name = f"blocks.{args.layer}.hook_resid_post"

    def hook(resid, hook):
        return resid + perturb.to(resid.dtype)

    print(f"\nCOMPOSED INJECTION at layer {args.layer}")
    for concept, alpha in mix.items():
        print(
            f"  {concept:<18}  α={alpha:>4}  (native scale {alpha*ALPHA_SCALE*avg_norm:>6.2f})"
        )
    print(f"  combined |perturb| = {perturb.norm().item():.2f}")
    print(f"  (typical resid norm at layer {args.layer} ≈ {avg_norm:.1f})\n")

    mix_label = "+".join(f"{c}={a}" for c, a in mix.items())

    for prompt in PROMPTS:
        print(f"PROMPT: {prompt}")
        # Baseline
        torch.manual_seed(SEED)
        base = model.generate(
            prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.7, verbose=False
        )
        base_gen = base[len(prompt) :]
        print(f"  base   → {base_gen.replace(chr(10), ' ').strip()[:250]}")

        # Steered
        torch.manual_seed(SEED)
        with model.hooks(fwd_hooks=[(hook_name, hook)]):
            out = model.generate(
                prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.7, verbose=False
            )
        gen = out[len(prompt) :]
        print(f"  steer  → {gen.replace(chr(10), ' ').strip()[:250]}\n")

        # Log the steered run (the interesting one)
        log_run(
            concept=f"compose({mix_label})",
            model=args.model,
            layer=args.layer,
            alpha=-1.0,  # per-concept alphas live in extra
            prompt=prompt,
            output=gen,
            resid_norm=avg_norm,
            cos_pos_neg=0.0,
            pos_count=0,
            neg_count=0,
            pos_examples=[],
            neg_examples=[],
            pos_strategy="composed",
            extra={
                "mix": mix,
                "combined_perturb_norm": float(perturb.norm()),
                "baseline_output": base_gen,
                "seed": SEED,
            },
        )


if __name__ == "__main__":
    main()
