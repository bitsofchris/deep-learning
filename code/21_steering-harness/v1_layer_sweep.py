"""v1 — Layer sweep for any contrastive concept.

For each layer in LAYERS:
  1. Harvest POS/NEG residual activations at that layer (last-token)
  2. Build mean-diff steering vector, normalize
  3. Generate one completion at fixed ALPHA (scaled to that layer's typical norm)
  4. Print and log: layer | norm | cos(p,n) | output

Every row is appended to results/runs.jsonl + results/runs.md.

Usage:
  python3 v1_layer_sweep.py                      # default: golden_gate
  python3 v1_layer_sweep.py --concept pickles    # the food experiment

Add a concept by creating data/<name>_pairs.py exposing POSITIVE, NEGATIVE,
then map it in CONCEPTS below.
"""

from __future__ import annotations

import argparse
import importlib
import torch

from transformer_lens import HookedTransformer
from results_logger import log_run

# Map a concept name to (pairs_module, default_prompt).
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
}

MODEL_NAME = "google/gemma-2-2b"
LAYERS = [6, 9, 12, 15, 18, 21, 24]
ALPHA = 4.0
ALPHA_SCALE = 0.1
POS_STRATEGY = "last"
MAX_NEW_TOKENS = 50
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
    ap.add_argument("--concept", default="golden_gate", choices=list(CONCEPTS))
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument(
        "--prompt", default=None, help="override the concept's default prompt"
    )
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
    print(f"ALPHA:   {args.alpha} (scaled × {ALPHA_SCALE} × typical_norm)\n")
    print(f"{'layer':>5} {'norm':>8} {'cos(p,n)':>10}   output")
    print("-" * 110)

    for layer in LAYERS:
        hook_name = f"blocks.{layer}.hook_resid_post"
        pos_mean, pos_norm = harvest_at(model, POSITIVE, hook_name)
        neg_mean, neg_norm = harvest_at(model, NEGATIVE, hook_name)
        diff = pos_mean - neg_mean
        unit = diff / diff.norm()
        cos_pn = torch.nn.functional.cosine_similarity(pos_mean, neg_mean, dim=0).item()
        typical_norm = (pos_norm + neg_norm) / 2
        native = args.alpha * typical_norm * ALPHA_SCALE

        def hook(resid, hook):
            return resid + native * unit.to(resid.dtype).to(resid.device)

        torch.manual_seed(SEED)
        with model.hooks(fwd_hooks=[(hook_name, hook)]):
            out = model.generate(
                prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.7, verbose=False
            )
        gen = out[len(prompt) :]

        # Console row
        row_out = gen.replace("\n", " ").strip()[:200]
        print(f"{layer:>5} {typical_norm:>8.1f} {cos_pn:>10.3f}   {row_out}")

        # Persistent log
        log_run(
            concept=args.concept,
            model=args.model,
            layer=layer,
            alpha=args.alpha,
            prompt=prompt,
            output=gen,
            resid_norm=typical_norm,
            cos_pos_neg=cos_pn,
            pos_count=len(POSITIVE),
            neg_count=len(NEGATIVE),
            pos_examples=POSITIVE,
            neg_examples=NEGATIVE,
            pos_strategy=POS_STRATEGY,
            extra={"alpha_scale": ALPHA_SCALE, "seed": SEED},
        )

    # Baseline (no steering) for reference — also logged with layer=-1, alpha=0
    torch.manual_seed(SEED)
    base = model.generate(
        prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.7, verbose=False
    )
    base_gen = base[len(prompt) :]
    print("-" * 110)
    print(
        f"{'base':>5} {' ':>8} {' ':>10}   {base_gen.replace(chr(10),' ').strip()[:200]}"
    )
    log_run(
        concept=args.concept,
        model=args.model,
        layer=-1,
        alpha=0.0,
        prompt=prompt,
        output=base_gen,
        resid_norm=0.0,
        cos_pos_neg=0.0,
        pos_count=0,
        neg_count=0,
        pos_examples=[],
        neg_examples=[],
        pos_strategy="baseline",
        extra={"note": "no steering"},
    )


if __name__ == "__main__":
    main()
