# Time Series Transformer for Forecasting — Overview

## Prerequisite: Transformers in 5 Minutes

### The Original Idea ("Attention Is All You Need", 2017)

Before transformers, sequence models (RNNs, LSTMs) processed tokens one at a time, left to right. This was slow and made it hard to learn long-range patterns — by the time you reach token 500, you've mostly forgotten token 10.

The transformer's key insight: **skip the sequential processing entirely.** Instead, let every token look at every other token simultaneously through a mechanism called **self-attention**. Token 500 can directly ask "what was token 10?" in a single step.

```
RNN:   token₀ → token₁ → token₂ → ... → token₅₀₀  (sequential, slow, forgetful)

Transformer:  every token attends to every other token in parallel (fast, no forgetting)
```

### What Self-Attention Actually Does

Each token asks three questions:
- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I contain?"
- **Value (V):** "What information should I pass along?"

Attention score = how well a Query matches a Key. High match = that token's Value gets weighted heavily in the output. It's a learned, differentiable lookup table.

```
"The cat sat on the mat"

Q: "mat" asks — who is relevant to me?
K: "cat" says  — I'm an animal noun
    "sat" says  — I'm a verb about position
    "on" says   — I'm a preposition

Attention weights: mat attends most to "sat" and "on" (positional context)
                   → output for "mat" = weighted mix of all Values
```

### Two Key Concepts: Causal Mask and Patches

**Causal Mask** — In normal attention, every token can see every other token (bidirectional). But when you're *generating* — predicting what comes next — that's cheating. Token 5 shouldn't see token 6 while trying to predict token 6.

A causal mask blocks attention to future positions. It's literally a triangle of -infinity values applied to the attention scores before softmax:

```
                  Token being attended to →
                    t0    t1    t2    t3    t4
Token        t0 [  ok   -inf  -inf  -inf  -inf ]   t0 can only see itself
attending    t1 [  ok    ok   -inf  -inf  -inf ]   t1 sees t0, t1
  ↓          t2 [  ok    ok    ok   -inf  -inf ]   t2 sees t0, t1, t2
             t3 [  ok    ok    ok    ok   -inf ]   t3 sees t0, t1, t2, t3
             t4 [  ok    ok    ok    ok    ok  ]   t4 sees everything before it
```

After softmax, -inf becomes 0 attention weight — those future tokens are invisible. This is what makes a model "autoregressive" (left-to-right, can only condition on the past). GPT uses this. So does NanoTST.

**Patch** — In language, the input unit is a token (a word or subword). In time series, the raw input is a long stream of numbers. You *could* treat each individual value as a token, but a 512-point series would mean 512 tokens and attention is O(n²) — that's 262,144 operations.

Instead, group consecutive values into non-overlapping chunks called **patches**:

```
Raw:     [v₀ v₁ v₂ ... v₃₁ | v₃₂ v₃₃ ... v₆₃ | v₆₄ ... | ... v₅₁₁]
           patch 0            patch 1            patch 2       patch 15

512 values ÷ 32 per patch = 16 patches = 16 tokens for the transformer
Attention cost: 16² = 256 operations (vs 262,144)
```

Each patch captures local temporal structure (the shape within those 32 timesteps), and the transformer learns relationships *between* patches via attention. This is the same idea as Vision Transformers (ViT), which chop images into square patches.

### The Transformer Block

One block = attention + feed-forward, with residual connections and normalization:

```
    input
      │
      ├──────────────┐
      ▼              │
   LayerNorm         │
      ▼              │
   Attention         │  "gather information from other positions"
      ▼              │
      + ◄────────────┘  residual connection
      │
      ├──────────────┐
      ▼              │
   LayerNorm         │
      ▼              │
   Feed-Forward      │  "process gathered information per position"
      ▼              │
      + ◄────────────┘  residual connection
      │
    output
```

Stack N of these blocks and each layer refines the representation. Early layers learn simple patterns, later layers learn complex compositions.

### Encoder vs Decoder vs Decoder-Only

The original 2017 paper had **both** an encoder and a decoder. This matters for understanding what we're building:

```
ENCODER-DECODER (original "Attention Is All You Need")
┌─────────────────────────────┐
│         ENCODER             │  Reads the full input (can see everything)
│  "Understand the source"    │  Bidirectional attention — no masking
│                             │  Used for: translation source, classification
└──────────────┬──────────────┘
               │ cross-attention
┌──────────────▼──────────────┐
│         DECODER             │  Generates output left-to-right
│  "Produce the target"       │  Causal mask — can only see past outputs
│                             │  Attends to encoder output via cross-attention
└─────────────────────────────┘

Example: Machine translation
  Encoder reads:  "Le chat est assis"  (sees all French tokens at once)
  Decoder writes: "The" → "cat" → "is" → "sitting"  (one at a time, left to right)
```

```
ENCODER-ONLY (BERT)
┌─────────────────────────────┐
│         ENCODER             │  Sees entire input at once (bidirectional)
│                             │  No causal mask — every token sees every other
│  Good for: understanding    │  Classification, similarity, embeddings
│  Bad for: generation        │  Can't generate sequentially
└─────────────────────────────┘
```

```
DECODER-ONLY (GPT, and what NanoTST uses)
┌─────────────────────────────┐
│         DECODER             │  Causal mask — each position only sees the past
│                             │  No encoder, no cross-attention
│  Good for: generation       │  Predicts next token/patch from context
│  "Given what came before,   │  Same architecture works for any sequence length
│   what comes next?"         │  This is what ChatGPT is
└─────────────────────────────┘
```

**Why decoder-only for time series forecasting?** Forecasting IS next-token prediction. You have a history of measurements and want to predict what comes next. You can't look at the future (causal mask). You want flexible prediction horizons without retraining. It's the same problem as language generation, just with numbers instead of words.

### From GPT to Time Series: What Changes

GPT takes discrete word IDs, looks them up in an embedding table, runs them through decoder blocks, and predicts a probability distribution over the next word from a fixed vocabulary.

NanoTST takes continuous measurements, projects patches through a linear layer, runs them through the same decoder blocks, and predicts a Gaussian distribution over the next patch of continuous values.

**The transformer in the middle is identical.** The only differences are at the edges — how you get data in and predictions out. If you understand GPT, you already understand 80% of what we're building.

---

## The Core Idea

Treat a time series like a sentence. Chop it into chunks ("patches"), feed them through a transformer, and predict what comes next — exactly like GPT predicts the next word, but with continuous values instead of tokens.

```
"The cat sat on the ___"     →  GPT predicts next word
[patch0][patch1]...[patch15] →  NanoTST predicts next patch
```

## The Pipeline

```
Raw time series (512 values)
    │
    ▼
┌──────────────┐   Normalize each series to mean=0, std=1 so the model
│  Normalize   │   doesn't waste capacity learning scale/offset.
└──────────────┘   Stores stats to convert predictions back to original scale.
    │
    ▼
┌──────────────┐   Chop into non-overlapping chunks. 512 values ÷ 32 = 16 patches.
│   Patch      │   Each patch captures local temporal structure.
│              │   This is why transformers work on long series — attention is
└──────────────┘   O(n²) on 16 patches, not 512 individual points.
    │
    ▼
┌──────────────┐   Linear projection: 32-dim patch → 128-dim embedding.
│   Embed      │   Gives the model room to represent features.
│  + Position  │   Add positional encoding so it knows WHERE each patch is.
└──────────────┘
    │
    ▼
┌──────────────┐   The engine. Each block does two things:
│  Transformer │
│  Blocks (×4) │   1. ATTENTION — each patch looks at all previous patches
│              │      and gathers relevant context. "What happened earlier
│              │      that helps me predict what comes next?"
│              │
│              │   2. FEED-FORWARD — processes each patch independently.
│              │      Attention gathers info, FFN processes it.
│              │
│              │   Causal mask ensures patch t only sees patches 0..t.
│              │   You can't look at the future when forecasting.
└──────────────┘
    │
    ▼
┌──────────────┐   Don't predict exact values — predict a distribution.
│  Prediction  │   Output: μ (best guess) and σ (uncertainty) per timestep.
│    Head      │   The model knows when it's confident vs when it's guessing.
└──────────────┘
    │
    ▼
  Forecast: μ ± σ for the next patch of values
```

## Training: Next-Patch Prediction

Same idea as GPT's next-token prediction, applied to patches:

```
Input patches:    [P0] [P1] [P2] [P3] ... [P14] [P15]
                    ↓    ↓    ↓    ↓         ↓
Predictions:      [P1] [P2] [P3] [P4] ... [P15]
                    ↕    ↕    ↕    ↕         ↕     compare with
Targets:          [P1] [P2] [P3] [P4] ... [P15]    Gaussian NLL loss
```

Each position predicts the next patch. Loss = negative log-likelihood of the true values under the predicted Gaussian. The model learns to both predict accurately (good μ) and calibrate its confidence (good σ).

## Inference: Autoregressive Forecasting

To forecast beyond the input, feed predictions back in:

```
1. Feed 512 values → model predicts distribution for next 32 values
2. Sample from that distribution → append to input
3. Feed last 512 values → predict next 32
4. Repeat until you have enough future values

Do this 50-100 times → take the median for a point forecast,
                        quantiles for uncertainty bands
```

## The Five Big Pieces

| Piece | What It Does | Why It Matters |
|-------|-------------|----------------|
| **Patching** | Groups timesteps into chunks | Makes attention tractable on long series |
| **Normalization** | Removes scale/offset per series | Model focuses on shape, not magnitude |
| **Causal Attention** | Each patch gathers context from past | Captures long-range dependencies without seeing the future |
| **Feed-Forward** | Per-patch nonlinear processing | Transforms gathered context into useful features |
| **Probabilistic Head** | Predicts distribution, not point | Honest about uncertainty; essential for real-world use |

## What Makes This Different from NLP Transformers

| | Language (GPT) | Time Series (NanoTST) |
|---|---|---|
| Input | Discrete tokens (word IDs) | Continuous values (measurements) |
| Tokenization | Lookup table | Linear projection of patches |
| Output | Probability over vocabulary | Gaussian distribution (μ, σ) |
| Loss | Cross-entropy | Negative log-likelihood |
| Prediction unit | 1 token | 1 patch (32 values) |

The architecture is the same transformer. The difference is at the edges — how you get data in and predictions out.
