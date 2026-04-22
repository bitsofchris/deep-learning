"""v0 — Minimum viable steering rig.

Pipeline:
  1. Load Gemma-2-2B via transformer_lens
  2. Harvest residual stream activations at LAYER for each contrastive pair
     (capture both last-token and mean-over-positions)
  3. Compute mean-diff steering vector; normalize to match native activation norm
  4. Register a forward hook that adds alpha * v to LAYER's residual stream
  5. Generate completions for a test prompt at a sweep of alphas
  6. Print side-by-side

Run:
  python v0_golden_gate.py

Env:
  HF_TOKEN required if the Gemma repo is gated for your account.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch

try:
    from transformer_lens import HookedTransformer
except ImportError:
    print("transformer_lens not installed. `pip install -r requirements.txt`.")
    sys.exit(1)

from data.golden_gate_pairs import NEGATIVE, POSITIVE

# -------- config --------
MODEL_NAME = "google/gemma-2-2b"
LAYER = 12  # residual stream hook site (of 26)
POS_STRATEGY = "last"  # "last" or "mean" — swap to compare
ALPHAS = [0.0, 2.0, 5.0, 10.0, 15.0]
TEST_PROMPT = "Tell me about your favorite place."
MAX_NEW_TOKENS = 80
SEED = 0

HOOK_NAME = f"blocks.{LAYER}.hook_resid_post"


@dataclass
class HarvestResult:
    last_token: torch.Tensor  # (d_model,)
    mean_tokens: torch.Tensor  # (d_model,)
    resid_norm: float  # typical ||resid|| at this layer/position


def harvest_activations(
    model: HookedTransformer, sentences: list[str]
) -> HarvestResult:
    """Run sentences through the model, capture residual stream at LAYER."""
    last_vecs = []
    mean_vecs = []
    norms = []
    for s in sentences:
        tokens = model.to_tokens(s)
        _, cache = model.run_with_cache(tokens, names_filter=HOOK_NAME)
        resid = cache[HOOK_NAME][0]  # (seq, d_model)
        last_vecs.append(resid[-1])
        mean_vecs.append(resid.mean(dim=0))
        norms.append(resid[-1].norm().item())
    return HarvestResult(
        last_token=torch.stack(last_vecs).mean(dim=0),
        mean_tokens=torch.stack(mean_vecs).mean(dim=0),
        resid_norm=float(sum(norms) / len(norms)),
    )


def build_steering_vector(
    model: HookedTransformer,
) -> tuple[torch.Tensor, float]:
    """Return (unit_vector, typical_resid_norm) for the Golden Gate direction."""
    pos = harvest_activations(model, POSITIVE)
    neg = harvest_activations(model, NEGATIVE)
    pos_vec = pos.last_token if POS_STRATEGY == "last" else pos.mean_tokens
    neg_vec = neg.last_token if POS_STRATEGY == "last" else neg.mean_tokens
    diff = pos_vec - neg_vec
    unit = diff / diff.norm()
    typical_norm = (pos.resid_norm + neg.resid_norm) / 2
    return unit, typical_norm


def make_hook(vector: torch.Tensor, alpha_native: float):
    """Add alpha_native * unit_vector to every position in the residual stream."""

    def hook(resid, hook):
        # resid: (batch, seq, d_model)
        return resid + alpha_native * vector.to(resid.dtype).to(resid.device)

    return hook


def generate_with_steering(
    model: HookedTransformer,
    prompt: str,
    vector: torch.Tensor | None,
    alpha_native: float,
) -> str:
    if vector is None or alpha_native == 0.0:
        return model.generate(
            prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.7, verbose=False
        )
    hook = make_hook(vector, alpha_native)
    with model.hooks(fwd_hooks=[(HOOK_NAME, hook)]):
        return model.generate(
            prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.7, verbose=False
        )


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    model.eval()

    print(f"Harvesting activations at layer {LAYER} (strategy={POS_STRATEGY})...")
    unit, typical_norm = build_steering_vector(model)
    print(f"  typical residual norm at layer {LAYER}: {typical_norm:.2f}")
    print(f"  steering vector shape: {tuple(unit.shape)}")

    print("\n" + "=" * 72)
    print(f"PROMPT: {TEST_PROMPT}")
    print("=" * 72)
    for alpha in ALPHAS:
        # alpha is interpreted as a fraction of the typical residual norm,
        # so alpha=1 means "add a perturbation the size of the residual itself."
        alpha_native = alpha * typical_norm * 0.1  # 0.1 makes alpha=10 ~= resid norm
        torch.manual_seed(SEED)  # same sampling across alphas
        out = generate_with_steering(model, TEST_PROMPT, unit, alpha_native)
        print(f"\n--- alpha={alpha} (native scale {alpha_native:.2f}) ---")
        print(out)


if __name__ == "__main__":
    main()
