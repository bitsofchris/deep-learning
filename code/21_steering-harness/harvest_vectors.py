"""Dump per-layer activations + mean-diff vectors for every concept.

For each (concept, layer) pair, saves:
  results/vectors/{concept}/layer_{N}.npz
    - positive_acts:  (30, d_model)  last-token residual per POSITIVE sentence
    - negative_acts:  (30, d_model)  last-token residual per NEGATIVE sentence
    - positive_mean:  (d_model,)     mean over positives
    - negative_mean:  (d_model,)     mean over negatives
    - diff:           (d_model,)     positive_mean - negative_mean
    - unit:           (d_model,)     normalized diff (this is the steering vector)
    - typical_norm:   float          avg ||residual|| at this layer
    - cos_pos_neg:    float          cos(positive_mean, negative_mean)

Plus results/vectors/{concept}/sentences.json with the pair text for reference.

This is the data for plotting:
  - histogram of projections (e · unit) split by pos/neg/other → shows separation
  - UMAP/PCA of all 60 points → shows cluster structure at each layer
  - cos(unit_layer_i, unit_layer_j) matrix → shows how the direction rotates
  - ||diff|| across layers → shows where the concept is strongest

Usage:
  python3 harvest_vectors.py                      # dump all concepts
  python3 harvest_vectors.py --concept pickles    # just one
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer

CONCEPTS = {
    "golden_gate": "data.golden_gate_pairs",
    "golden_gate_v2": "data.golden_gate_v2_pairs",
    "pickles": "data.food_pairs",
    "pirate": "data.pirate_pairs",
}

MODEL_NAME = "google/gemma-2-2b"
LAYERS = [6, 9, 12, 15, 18, 21, 24]
OUT_DIR = Path(__file__).parent / "results" / "vectors"


def harvest_per_sentence(model, sentences, hook_name):
    """Return (N, d_model) tensor of last-token residual for each sentence."""
    acts = []
    for s in sentences:
        tokens = model.to_tokens(s)
        _, cache = model.run_with_cache(tokens, names_filter=hook_name)
        resid = cache[hook_name][0]  # (seq, d_model)
        acts.append(resid[-1].detach().cpu().float())
    return torch.stack(acts)  # (N, d_model)


def dump_concept(model, concept_name: str, pairs_module_name: str):
    print(f"\n=== {concept_name} ===")
    pairs = importlib.import_module(pairs_module_name)
    POS, NEG = pairs.POSITIVE, pairs.NEGATIVE

    concept_dir = OUT_DIR / concept_name
    concept_dir.mkdir(parents=True, exist_ok=True)

    # Save the sentences once
    (concept_dir / "sentences.json").write_text(
        json.dumps({"positive": POS, "negative": NEG}, indent=2)
    )

    for layer in LAYERS:
        hook_name = f"blocks.{layer}.hook_resid_post"
        pos_acts = harvest_per_sentence(model, POS, hook_name)
        neg_acts = harvest_per_sentence(model, NEG, hook_name)

        pos_mean = pos_acts.mean(dim=0)
        neg_mean = neg_acts.mean(dim=0)
        diff = pos_mean - neg_mean
        unit = diff / diff.norm()

        pos_norms = pos_acts.norm(dim=1).mean().item()
        neg_norms = neg_acts.norm(dim=1).mean().item()
        typical_norm = (pos_norms + neg_norms) / 2
        cos_pn = torch.nn.functional.cosine_similarity(pos_mean, neg_mean, dim=0).item()

        out_path = concept_dir / f"layer_{layer:02d}.npz"
        np.savez(
            out_path,
            positive_acts=pos_acts.numpy(),
            negative_acts=neg_acts.numpy(),
            positive_mean=pos_mean.numpy(),
            negative_mean=neg_mean.numpy(),
            diff=diff.numpy(),
            unit=unit.numpy(),
            typical_norm=np.array(typical_norm, dtype=np.float32),
            cos_pos_neg=np.array(cos_pn, dtype=np.float32),
        )
        print(
            f"  layer {layer:>2}  norm={typical_norm:>7.2f}  cos(p,n)={cos_pn:.3f}  "
            f"||diff||={diff.norm().item():>6.2f}  →  {out_path.relative_to(Path(__file__).parent)}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default=None, choices=list(CONCEPTS))
    ap.add_argument("--model", default=MODEL_NAME)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model}...")
    model = HookedTransformer.from_pretrained(args.model, device=device)
    model.eval()

    concepts = [args.concept] if args.concept else list(CONCEPTS.keys())
    for name in concepts:
        dump_concept(model, name, CONCEPTS[name])

    print(f"\nDone. Vectors in {OUT_DIR}")


if __name__ == "__main__":
    main()
