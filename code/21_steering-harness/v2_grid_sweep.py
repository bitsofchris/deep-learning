"""v2 — Grid sweep over (layer × alpha).

For each (layer, alpha) cell:
  1. Harvest POS/NEG residual activations at `layer` (last-token)
  2. Build mean-diff steering vector (once per layer — reused across alphas)
  3. Generate one completion at `alpha` scaled to that layer's typical norm
  4. Log to results/runs.jsonl + results/runs.md and print a compact table

Usage:
  python3 v2_grid_sweep.py --concept pickles
"""

from __future__ import annotations

import argparse
import importlib
import torch

from transformer_lens import HookedTransformer
from results_logger import log_run

CONCEPTS = {
    "golden_gate": (
        "data.golden_gate_pairs",
        "My favorite place in the whole world is",
    ),
    "golden_gate_v2": (
        "data.golden_gate_v2_pairs",
        "My favorite place in the whole world is",
    ),
    "pickles": ("data.food_pairs", "My favorite food in the whole world is"),
    "pirate": ("data.pirate_pairs", "I was walking down the street today and"),
}

MODEL_NAME = "google/gemma-2-2b"
LAYERS = [9, 12, 15, 18, 21, 24]
ALPHAS = [2.0, 4.0, 6.0, 8.0, 10.0]
ALPHA_SCALE = 0.1
MAX_NEW_TOKENS = 40
SEED = 0


def harvest_at(model, sentences, hook_name):
    last = []
    norms = []
    for s in sentences:
        tokens = model.to_tokens(s)
        _, cache = model.run_with_cache(tokens, names_filter=hook_name)
        resid = cache[hook_name][0]
        last.append(resid[-1])
        norms.append(resid[-1].norm().item())
    return torch.stack(last).mean(dim=0), sum(norms) / len(norms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="pickles", choices=list(CONCEPTS))
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--prompt", default=None)
    args = ap.parse_args()

    pairs_module_name, default_prompt = CONCEPTS[args.concept]
    pairs = importlib.import_module(pairs_module_name)
    POSITIVE, NEGATIVE = pairs.POSITIVE, pairs.NEGATIVE
    prompt = args.prompt or default_prompt

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model}...")
    model = HookedTransformer.from_pretrained(args.model, device=device)
    model.eval()

    print(f"\nCONCEPT: {args.concept}  |  N+={len(POSITIVE)}  N-={len(NEGATIVE)}")
    print(f"PROMPT:  {prompt}")
    print(f"GRID:    layers={LAYERS}  alphas={ALPHAS}\n")

    # Build direction per layer once, sweep alphas
    for layer in LAYERS:
        hook_name = f"blocks.{layer}.hook_resid_post"
        pos_mean, pos_norm = harvest_at(model, POSITIVE, hook_name)
        neg_mean, neg_norm = harvest_at(model, NEGATIVE, hook_name)
        diff = pos_mean - neg_mean
        unit = diff / diff.norm()
        cos_pn = torch.nn.functional.cosine_similarity(pos_mean, neg_mean, dim=0).item()
        typical_norm = (pos_norm + neg_norm) / 2

        print(f"=== layer {layer}  norm={typical_norm:.1f}  cos(p,n)={cos_pn:.3f} ===")
        for alpha in ALPHAS:
            native = alpha * typical_norm * ALPHA_SCALE

            def hook(resid, hook, _v=unit, _s=native):
                return resid + _s * _v.to(resid.dtype).to(resid.device)

            torch.manual_seed(SEED)
            with model.hooks(fwd_hooks=[(hook_name, hook)]):
                out = model.generate(
                    prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=0.7,
                    verbose=False,
                )
            gen = out[len(prompt) :]
            print(f"  α={alpha:>4}  →  {gen.replace(chr(10), ' ').strip()[:160]}")

            log_run(
                concept=args.concept,
                model=args.model,
                layer=layer,
                alpha=alpha,
                prompt=prompt,
                output=gen,
                resid_norm=typical_norm,
                cos_pos_neg=cos_pn,
                pos_count=len(POSITIVE),
                neg_count=len(NEGATIVE),
                pos_examples=POSITIVE,
                neg_examples=NEGATIVE,
                pos_strategy="last",
                extra={"alpha_scale": ALPHA_SCALE, "seed": SEED, "grid": True},
            )
        print()


if __name__ == "__main__":
    main()
