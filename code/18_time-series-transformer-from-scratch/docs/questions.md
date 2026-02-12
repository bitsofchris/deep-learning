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
