# NanoTST — Time Series Transformer from Scratch

A ~300 line time series transformer built from nothing, one concept at a time. Same architecture as GPT — causal attention, feed-forward blocks, residual connections — but operating on continuous time series patches instead of word tokens. Predicts probability distributions (mu + sigma) over future values, not point estimates.

Built on a plane as a learning exercise.

## Files

| File | What | Who wrote it |
|------|------|-------------|
| `nano_tst.py` | The model — from linear baseline through full transformer with Gaussian head | Me (typed by hand) |
| `ts_grammar_eval.py` | Training loop with grammar evaluation, visualization, forecasting | Claude |
| `docs/overview.md` | Transformer concepts, pipeline diagram, architecture explanation | Claude |
| `docs/incremental-build.md` | Step-by-step build guide (the plan I followed) | Claude |
| `docs/questions.md` | Every TODO comment I wrote, answered | Claude (answers to my questions) |
| `docs/nano-tst-guide.md` | Architecture deep-dive, production model comparison | Claude |
| `youtube.md` | Video script for walking through this project | Both |
| `later/` | Reference code for future steps (experiments, attention viz) | Claude |

## The Process

1. **Scoped the project** with Claude — what does a minimal time series transformer look like? What are the right building blocks?
2. **Got an incremental build plan** (`docs/incremental-build.md`) — 10 steps, each runnable, each adding one concept. Data, baseline, normalization, attention, multi-head, FFN, Gaussian head, grammar eval, forecasting, experiments.
3. **Typed every line of `nano_tst.py` by hand.** No copy-paste from the guide. When I didn't understand something, I wrote a `TODO` comment and kept going.
4. **Reviewed in phases.** After each step, ran the code, looked at outputs, asked questions about what I'd just written. Those questions became `docs/questions.md`.
5. **Built up to a working model** — steps 1 through 7 in a single session. The final model trains on synthetic data and learns to predict flat signals first, then trends, then sine waves — the "grammar" of time series.

## Run it

```bash
python nano_tst.py
```

Trains NanoTST on 2000 synthetic series for 100 epochs. Prints grammar eval (flat/line/sine/noisy predictions + confidence) and saves training progression plots to `images/`.

## What the model does

```
Raw time series (512 values)
  → Normalize (zero mean, unit variance)
  → Patch (16 chunks of 32 values)
  → Embed (32-dim → 128-dim per patch)
  → 4x Transformer blocks (multi-head causal attention + FFN)
  → Gaussian head (predict mu + sigma per future patch)
```

Next-patch prediction, same as GPT's next-token prediction. Causal mask ensures no future leakage. Gaussian NLL loss so the model learns both accuracy and calibrated uncertainty.
