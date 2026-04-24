"""Generate N unsteered baseline completions per prompt.

Same model, same prompts used in v3/v4 composition runs. No hooks. Multiple
seeds so we can show distribution of Gemma's unsteered behavior for each
prompt — this is the "before" column against which the steered outputs are
compared in the blog post.

Run:
  python3 baselines.py
"""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer
from results_logger import log_run

MODEL_NAME = "google/gemma-2-2b"
MAX_NEW_TOKENS = 80
N_PER_PROMPT = 5
SEEDS = list(range(N_PER_PROMPT))  # 0..4

PROMPTS = [
    "The best adventure I can imagine is",
    "My perfect day involves",
    "If I could go anywhere in the world right now, I'd",
    "Let me tell you about my ideal life:",
    "Today I'm going to",
    "I was walking down the street today and",
    "My favorite food in the whole world is",
    "My favorite place in the whole world is",
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    model.eval()

    for prompt in PROMPTS:
        print(f"\nPROMPT: {prompt}")
        for seed in SEEDS:
            torch.manual_seed(seed)
            out = model.generate(
                prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.7, verbose=False
            )
            gen = out[len(prompt) :]
            short = gen.replace("\n", " ").strip()[:180]
            print(f"  seed={seed}  →  {short}")
            log_run(
                concept="baseline",
                model=MODEL_NAME,
                layer=-1,
                alpha=0.0,
                prompt=prompt,
                output=gen,
                resid_norm=0.0,
                cos_pos_neg=0.0,
                pos_count=0,
                neg_count=0,
                pos_examples=[],
                neg_examples=[],
                pos_strategy="baseline",
                extra={
                    "seed": seed,
                    "temperature": 0.7,
                    "max_new_tokens": MAX_NEW_TOKENS,
                },
            )


if __name__ == "__main__":
    main()
