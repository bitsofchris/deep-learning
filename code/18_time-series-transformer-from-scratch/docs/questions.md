# Questions & Answers — NanoTST Build

## Q1: How did previous sequence models (RNNs/LSTMs) work?

They process tokens **one at a time, left to right**. At each step the model takes the current input + a running summary vector and produces an updated summary. This is sequential (can't parallelize) and the summary degrades over long sequences — by token 500, token 10's signal is mostly gone. LSTMs added gates to control what's kept/forgotten, but the fundamental bottleneck remains.

## Q2: What is the hidden state?

The hidden state is the model's **learned latent representation** of everything seen so far — a single point in a high-dimensional space (dimensionality is a hyperparameter, e.g. 128 or 256, not the input size). Through training the model learns to organize this space so similar sequences map to nearby points.

The key limitation: one fixed-size vector must encode the entire history. A transformer instead keeps **separate representation vectors for every token**, giving it far more capacity.

## Q3: Why patches instead of raw values or text-like tokens?

**Compute:** Attention is O(n²). 512 raw values = 262,144 ops. 16 patches = 256 ops. ~1000x cheaper.

**Richer tokens:** A single number (`0.37`) carries almost no meaning. A patch of 32 consecutive values captures local shape (rising, oscillating, flat) — giving the transformer something semantically meaningful to work with, like how a word token carries meaning in NLP.

## Q4: Is the hidden state the same as an embedding?

Yes. The hidden state IS the model's embedding of the sequence so far — a point in learned high-dimensional space. The model learns through backprop how to best organize this space. The difference: an RNN compresses everything into **one** vector. A transformer maintains **one vector per token**, all refined through attention.

## Q5: What is the context length? How many tokens attend to each other?

Context length = number of tokens that can attend to each other (the "context window"). In NanoTST: `512 values ÷ 32 per patch = 16 patches = 16 tokens in context`. The 16 is the context length, the 32 is the patch size. Both are hyperparameters — longer history means more patches, but attention cost grows quadratically.

## Q6: How are patches different from codebook/quantile-bin approaches?

Some models (e.g. Chronos) discretize values into bins and use cross-entropy like NLP. Linear patch projection keeps values **continuous**.

| | Codebook / Binning | Linear Patches |
|---|---|---|
| Information | Loses precision through discretization | Preserves full continuous signal |
| Output | Cross-entropy over bins | Gaussian μ,σ (natural for regression) |
| Complexity | Must design/learn vocabulary | Just a linear layer |
| Local structure | Each timestep = one token | Patch captures local shape as a unit |

Patches let the transformer's own training discover what shape features matter, without pre-defining a vocabulary. Active research question — Chronos's simplicity has its own advantages.

## Q7: What is softmax? Why is it used in attention?

Softmax converts arbitrary numbers (logits) into a **probability distribution** — all values 0-1 that sum to 1. Formula: `softmax(xᵢ) = exp(xᵢ) / Σ exp(xⱼ)`. In attention, it turns raw Q·K scores into weights that sum to 1, so the output is a weighted average of Values. The `exp()` makes large differences sharper (~150:1 for a 10 vs 5 score). Also critical: `exp(-inf) = 0`, which is how the causal mask works.

**Alternatives:** Sparsemax (produces true zeros), linear attention (skip softmax, O(n) but less expressive), sigmoid attention (independent per-score, no competition between tokens). Softmax dominates because it's stable, well-understood, and works cleanly with the -inf masking trick.

## Q8: What does autoregressive mean? What's the meta-category?

**Autoregressive** = predicts next element conditioned only on previous elements, feeds predictions back as input. "Auto" (self) + "regressive" (predicting from prior values).

The meta-category is **directionality of information flow:**
- **Autoregressive / causal / decoder** — left-to-right only, can't see future. For generation/forecasting.
- **Bidirectional / encoder** — sees both directions. For understanding/classification.
- **Encoder-decoder** — encoder is bidirectional, decoder is autoregressive with cross-attention to encoder.

## Q9: What happens in a transformer block? How are blocks connected?

**Input and output have the same shape** `[batch, num_patches, embed_dim]` — this is what makes blocks stackable. Each block: LayerNorm → Attention (gather info from other positions) → residual add → LayerNorm → FFN (process info per position) → residual add.

**Blocks are sequential** (output of block 1 → input of block 2). Early blocks learn simple patterns (local trends), later blocks compose these into complex patterns. This isn't explicitly programmed — it emerges from training.

**How many blocks?** Hyperparameter. More blocks = more capacity but more parameters and overfitting risk. Each block adds ~4 × embed_dim² parameters. NanoTST uses 4. GPT-3 uses 96. Start small, increase until validation loss stops improving. Embed_dim is the bigger lever: doubling it quadruples params.

## Q10: Would an encoder model help learn richer embeddings for our decoder?

An encoder's advantage is **bidirectionality** — patch 5 sees both past and future, so its representation is richer. This helps for understanding tasks (classification, anomaly detection). For forecasting/generation, it helps less because at inference you can't see the future anyway.

Models like Informer/Autoformer use encoder-decoder. PatchTST has a self-supervised encoder mode (mask patches, reconstruct — like BERT). But decoder-only (TimesFM, Chronos, NanoTST) is winning for forecasting — simpler, scales better, and representations learned causally match the inference setting. Same trajectory as NLP: Google went encoder-decoder (T5), OpenAI went decoder-only (GPT), decoder-only won for generation.

---

## Code Walk-Through: nano_tst.py TODOs

### Data & Training

**Q11: `nn.Linear(patch_size, patch_size)` — is this a 32x32 linear layer?**

Close. It has a weight matrix (32, 32) **plus** a bias vector (32) = 1,056 params. It takes each 32-value patch and linearly maps it to a new 32-value prediction. The baseline has zero context — each patch is predicted from itself alone.

**Q12: The loss `((pred - target) ** 2).mean()` — this just does MSE?**

Yes, literally mean squared error. `.mean()` averages over all dimensions (batch, patches, values within patches). Simplest regression loss. Later (step 6) we replace this with Gaussian NLL which also learns uncertainty.

**Q13: `torch.randperm(len(data))` — what is this?**

Generates a random permutation of indices — like shuffling a deck. With 2000 series you get `[847, 12, 1993, ...]`. Shuffles which series appear in each batch every epoch so the model doesn't learn ordering artifacts.

### Normalization

**Q14: `+ 1e-5` in InstanceNorm — why?**

Prevents division by zero. If a series is perfectly constant, std = 0 and dividing by it gives NaN, poisoning all gradients. This tiny **epsilon** term is a universal deep learning pattern.

### Patch Embedding

**Q15: `nn.Linear(32, 128)` — does it project each patch into higher dimensions?**

Yes. Takes each 32-value patch and projects to 128 dims. The model learns *what* linear combinations of those 32 values produce useful features (slope, mean level, oscillation strength, etc.).

**Q16: What does `view()` do? Why reshape?**

`view()` reshapes a tensor without copying data — just changes how you index into the same memory. `(batch, 512)` → `(batch, 16, 32)` tells PyTorch "interpret this flat sequence as 16 patches of 32." No data changes. We reshape because `nn.Linear` operates on the **last dimension** — after reshape, it applies independently to each 32-value patch.

### Self-Attention Mechanics

**Q17: What does single-head mean? How do heads relate to blocks?**

A **head** is one independent attention computation (one set of Q,K,V, one attention matrix). Multi-head = split the embedding across parallel heads (e.g., 4 heads × 32 dims = 128 total), each attending independently, then concatenate. Heads run **in parallel within** a block's attention layer. Blocks are stacked **sequentially** (vertically).

**Q18: Why is QKV shaped 128 → 3×128?**

Efficiency trick. Instead of three separate layers (W_q, W_k, W_v each 128→128), one big 128→384 layer followed by a split. Mathematically identical, but one big matmul is faster on GPUs than three small ones. Output is `[Q₁₂₈ | K₁₂₈ | V₁₂₈]` concatenated.

**Q19: Are PyTorch layers connected because of init order?**

**No.** `__init__` just creates layers as named attributes. `forward()` defines how data flows through them. You could define them in any order in `__init__` — only the call order in `forward()` matters. This is different from Keras Sequential where definition order = execution order.

**Q20: Every class has `forward()` — that's an nn.Module thing?**

Yes. `nn.Module` requires you to implement `forward()`. When you call `model(x)`, PyTorch calls `model.forward(x)` plus bookkeeping (hooks, gradient tracking). Always call the module as a function (`model(x)`), never `model.forward(x)` directly.

**Q21: Do Q, K, V have to be the same size?**

Q and K **must** match (you dot-product them: Q·Kᵀ). V can technically differ — its dimension sets the attention output size. In practice all three are kept equal for simplicity. Q and K determine *where* to look, V determines *what information* to pass.

**Q22: What does `chunk(3, dim=-1)` do?**

Splits a tensor into 3 equal pieces along the last dimension. `(B, T, 384)` → three `(B, T, 128)` tensors = Q, K, V. Inverse of the concatenation trick from the single QKV linear layer.

**Q23: What does `triu` do? The causal mask?**

`triu` = **tri**angle **u**pper. Returns the upper triangular part, zeroing everything below. With `diagonal=1`, zeros the diagonal too. The 1s mark **future positions**. After `masked_fill(mask, -inf)` and softmax, -inf → 0 weight — future patches become invisible.

```
triu(ones(4,4), diagonal=1):     After softmax, patch 0 sees only itself,
  [[0, 1, 1, 1],                 patch 1 sees 0-1, patch 2 sees 0-2, etc.
   [0, 0, 1, 1],
   [0, 0, 0, 1],
   [0, 0, 0, 0]]
```

### Model Head

**Q24: What does `nn.Linear(d_model, patch_size)` (the head) do?**

Projects from the transformer's internal space (128-dim) back to **patch space** (32-dim) — 32 actual time series values. The transformer works in 128 dims; the head converts its final representation into a concrete prediction. Equivalent to GPT's layer that projects hidden dim → vocabulary size.

### Multi-Head Attention Details

**Q25: `head_dim = d_model // n_heads` — output spread across heads?**

Yes, 128 ÷ 4 = 32 dims per head. You're not adding parameters — you're **partitioning** them so different heads can specialize (one tracks trend, another tracks periodicity, etc.). Total computation matches single-head.

**Q26: `out_proj = nn.Linear(d_model, d_model)` — why?**

After heads compute independently and get concatenated back to 128 dims, this layer **mixes across heads**. Without it, each head's 32-dim output stays isolated. The output projection lets the model learn combinations like "70% of head 1's trend + 30% of head 3's periodicity."

**Q27: What is `permute()` doing in multi-head attention?**

Reorders tensor dimensions. After reshape: `(B, T, 3, n_heads, head_dim)`. For attention we need `(3, B, n_heads, T, head_dim)` — so `qkv[0]` gives all Q's, and heads act as a batch dimension for parallel GPU computation. `permute(2, 0, 3, 1, 4)` does this rearrangement.

**Q28: Why `.transpose(1,2).reshape(B,T,D)` after attention?**

Undoes the head split. After attention: `(B, n_heads, T, head_dim)` → transpose → `(B, T, n_heads, head_dim)` → reshape → `(B, T, 128)`. Concatenates 4 heads × 32 dims = 128 dims back together.

### FFN and Transformer Block

**Q29: `hidden = 4 * d_model` — why?**

FFN is expand-then-compress: 128 → 512 → 128. The 4x expansion (from the original transformer paper) gives more capacity for per-position computation. Think of attention as "gather info from other patches" and FFN as "process that gathered info" — the wider hidden layer lets it learn richer transformations.

**Q30: What is GELU?**

Activation function like ReLU but smoother. Instead of hard cutoff at 0, GELU has a soft curve: `GELU(x) ≈ x · sigmoid(1.702x)`. Small negatives get dampened (not zeroed), large positives pass through. Standard in modern transformers (GPT, BERT). Better gradients than ReLU because no discontinuity at zero.

**Q31: Pre-norm block order: LN → Attn → residual → LN → FFN → residual — why?**

**LayerNorm before** (not after) stabilizes training in deep networks. Original paper did post-norm; pre-norm (GPT-2+, all modern transformers) trains more stably. **Residual connections** (`x = x + sublayer(norm(x))`) let gradients flow directly through the skip path — without them, 4+ blocks have vanishing gradients.

The line `x = x + self.attn(self.norm1(x))` means: normalize, run attention, then **add original x back**. The sublayer learns a *delta* (refinement), not a full replacement.

### Positional Encoding

**Q32: `nn.Embedding(16, 128)` — how does this learn positions?**

It's a lookup table: 16 rows × 128 values, one row per patch position. Randomly initialized, learned through backprop like any other parameter. The model discovers that adding a specific vector for "position 3" vs "position 12" helps predictions. Nothing forces it to learn "order" — it learns whatever position-dependent patterns reduce loss.

**Q33: What does LayerNorm do?**

Normalizes across the feature dimension (128 dims) for each token independently. Recenters to mean=0, rescales to std=1, then applies learned scale/shift. Keeps activations in a stable range through many blocks, preventing explosion or vanishing. Different from InstanceNorm (normalizes across time) — LayerNorm normalizes across features.

**Q34: `h = h + self.pos(...)` — how does adding a vector encode position?**

Element-wise vector addition. Each patch's 128-dim embedding gets a 128-dim position vector added. The position vector shifts the embedding to a slightly different region of 128-dim space depending on position. The transformer learns to interpret that shift — separating content info from position info through training. Seems crude, works remarkably well.

---

## Evaluation: Grammar Plots vs Forecast Plots

### Q35: What are the grammar eval plots showing during training?

These are **teacher-forced, one-step predictions**. The model sees every real patch as input and predicts the next patch at each position. Patch 0 → predict patch 1, patch 1 → predict patch 2, etc. The model always gets the ground truth as context — it never has to rely on its own output.

This is the **same computation as training** (just without the gradient update). It answers: "given perfect context, can the model predict what comes next?" It's the direct visualization of what the NLL loss is measuring.

**What to watch for across epochs:**
- Early: predictions are noisy garbage for all patterns. The blue line barely tracks the black target.
- Mid: flat gets nailed first (easiest pattern — just predict the same value). Line improves next. Sine still rough.
- Late: sine tracking tightens up. The ±1σ band shrinks where the model is confident and stays wider where it's uncertain. This is the "grammar" learning order — simple structure before complex structure, just like a language model learning articles before grammar before idioms.

### Q36: What are the forecast plots showing? How is this different?

These are **autoregressive, multi-step predictions** — actual inference. We chop the last 4 patches (128 points) off each test series as a holdout. The model sees only the first 12 patches as context, then must predict the held-out 4 patches by feeding its own predictions back as input.

Step by step:
1. Feed 12 real patches → model predicts patch 13
2. Append predicted patch 13 → feed 13 patches (12 real + 1 predicted) → predict patch 14
3. Append predicted patch 14 → predict patch 15
4. Append predicted patch 15 → predict patch 16

The key difference: **errors compound**. If the model's prediction for patch 13 is slightly off, patch 14's input is already wrong, and the error cascades. This is a much harder test than grammar eval.

**What to watch for across epochs:**
- Early: forecasts diverge immediately after the split line — the model can't maintain coherent structure from its own outputs.
- Mid: flat forecast stabilizes (easy to continue "more of the same"). Line direction roughly holds but drifts. Sine falls apart after 1-2 predicted patches.
- Late: sine continues oscillating past the split. Line maintains slope. The q10–q90 band shows where uncertainty grows — typically widens further from the split as compounding errors accumulate.
- The MSE in each subplot title is measured against the actual held-out data, so you get a real number for forecast quality.

### Q37: Why do we need both? What's the intuition?

They measure fundamentally different capabilities:

| | Grammar Eval (teacher-forced) | Forecast (autoregressive) |
|---|---|---|
| **Analogy** | Fill-in-the-blank test | Write an essay |
| **Input** | All real data | Own predictions fed back |
| **Errors** | Independent per position | Compound over steps |
| **Measures** | "Does the model understand the pattern?" | "Can the model generate the pattern?" |
| **LLM equivalent** | Perplexity on held-out text | Free generation quality |

A model can ace grammar eval but produce garbage forecasts — it learned to copy local patterns but can't maintain coherent structure over multiple self-fed steps. Watching both plots evolve across epochs shows you when the model transitions from "understands the pattern" to "can actually produce it." That transition is the interesting part.
