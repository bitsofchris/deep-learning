---
title: America AI
emoji: 🦅
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Slide the freedom up — live activation steering on Gemma 2
---

# America AI 🦅

A satirical, intentionally politically steered chatbot that demonstrates
**contrastive activation steering** on an unmodified `google/gemma-2-2b-it`.

Four steering directions — Americana, national pride, Trump approval, and
star-spangled bombast — were extracted from paired contrastive sentences.
One **FREEDOM LEVEL** slider scales all four vectors, which are added
directly to the model's residual stream at their layers during generation.
No fine-tuning, no system prompt, no prompt tricks.

- 😐 **Normal model** (0) · 🇺🇸 **Hints of America** (~200) ·
  🦅 **Max Freedom** (~390, the default) · 🥴 **Star Drunk** (500, dissolves
  into star-spangled word salad).

## How it works

The Space loads `steering_bundle.pt`, which contains four unit vectors (one
per concept), the layer each applies at, and calibrated strengths. A forward
hook on each steered decoder layer adds
`multiplier × base_strength × typical_norm × unit_vector` to the hidden
states. See `runtime.py` for the full implementation — it's ~200 lines of
plain `transformers`.

The vectors were found by mean-differencing paired contrastive sentences
(patriotic vs. neutral completions of the same prompt). Full write-up:
[America AI](https://bitsofchris.com/p/american-ai) on bitsofchris.com.

## Disclaimer

America AI is a parody built as an educational demo of how easily language
models can be invisibly steered. Its responses are generated entertainment,
not neutral or factual guidance.

## Secrets

Requires `HF_TOKEN` with access to the gated `google/gemma-2-2b-it`.
