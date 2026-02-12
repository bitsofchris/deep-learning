# NanoTST: Build a Time Series Transformer from Scratch

## Learn by Implementing

This guide maps the full architecture of a production-grade time series transformer, then strips it down to an MVP you can implement in ~200-300 lines of PyTorch.

---

## Part 1: Full Production Architecture Map

Here's how a state-of-the-art time series transformer works end-to-end. Every component is listed so you can see where the pieces fit.

```
INPUT: Multivariate time series (M variates × L=4096 timesteps)
  │
  ▼
┌─────────────────────────────────────────────┐
│  1. CAUSAL PATCH-BASED INSTANCE NORM        │  ← Key innovation
│     Per-patch normalization using only       │
│     current + past data (Welford's online    │
│     algorithm). Handles nonstationarity.     │
│     Clipping mechanism for stability.        │
└─────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────┐
│  2. PATCH EMBEDDING                         │
│     Non-overlapping patches of size P=64    │
│     Linear projection → D=768 embedding dim │
│     Result: M × (L/P) × D = M × 64 × 768  │
└─────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────┐
│  3. POSITIONAL ENCODING                     │
│     RoPE (Rotary Position Embeddings)       │
│     + XPOS for extrapolation                │
└─────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────┐
│  4. TRANSFORMER DECODER STACK               │
│     F=1 segment containing:                 │
│     ┌─────────────────────────────────┐     │
│     │  N=11 TIME-WISE blocks          │     │
│     │  (causal self-attention along   │     │
│     │   the time/patch dimension)     │     │
│     ├─────────────────────────────────┤     │
│     │  1 VARIATE-WISE block           │     │  ← "Proportional Factorized
│     │  (attention across variates     │     │     Attention"
│     │   at each time position)        │     │
│     └─────────────────────────────────┘     │
│                                             │
│     Each block has:                         │
│     - Pre-norm (RMSNorm)                    │
│     - Multi-head attention                  │
│     - SwiGLU feed-forward                   │
│     - Residual connections                  │
└─────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────┐
│  5. UNEMBED / OUTPUT PROJECTION             │
│     Project D → patch_size output           │
│     (reverse of patch embedding)            │
└─────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────┐
│  6. STUDENT-T MIXTURE PREDICTION HEAD       │  ← Key innovation
│     Outputs parameters of a mixture of      │
│     Student-T distributions:                │
│     - K mixture components                  │
│     - Each with: μ, σ, ν (df), π (weight)  │
│     Handles heavy-tailed observability data  │
└─────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────┐
│  7. LOSS: COMPOSITE ROBUST LOSS             │  ← Key innovation
│     Negative log-likelihood of Student-T    │
│     mixture + stabilization terms           │
└─────────────────────────────────────────────┘
  │
  ▼
OUTPUT: Probabilistic forecast (sample from mixture)
        → Take median of 256 samples for point forecast
```

### What Makes a TS Transformer Different from NanoGPT

| Component | NanoGPT (Language) | Time Series Transformer |
|-----------|-------------------|---------------------|
| Input | Token IDs → embedding lookup | Continuous values → patch + linear projection |
| Normalization | LayerNorm | Causal patch-based instance norm (before embedding) + RMSNorm (in blocks) |
| Attention | Single dimension (sequence) | Factorized: time-wise AND variate-wise |
| Positional | Learned absolute | RoPE + XPOS |
| FFN | GELU MLP | SwiGLU |
| Output | Softmax over vocab | Student-T mixture distribution |
| Loss | Cross-entropy | Negative log-likelihood of mixture |
| Prediction | Next token | Next patch (64 continuous values) |

---

## Part 2: Where Other Approaches Fit

At each stage, there are alternatives. This is your map of the field:

### Input Representation
- **Patching** (PatchTST, TimesFM): Group consecutive timesteps → one token. Most common now.
- **Point-wise**: Each timestep is a token (older approach, expensive for long sequences).
- **Tokenization/quantization** (Chronos): Bin continuous values into discrete tokens, use standard LM architecture. Simpler but lossy.

### Normalization
- **RevIN** (Reversible Instance Norm): Normalize whole series, denormalize output. Simple but leaks future info.
- **First-patch normalization** (TimesFM): Stats from first patch only. Causal but bad for nonstationary data.
- **Causal patch normalization**: Running stats per patch. Best of both worlds.

### Attention Structure
- **Channel-independent** (PatchTST, TimesFM): Each variate processed independently. Simple, works surprisingly well.
- **Full space-time** (some iTransformer variants): Attend across all variates and all times simultaneously. Expensive at high dimensionality.
- **Factorized**: Separate time-wise and variate-wise attention. Efficient for high-dimensional multivariate data.

### Architecture
- **Encoder-only** (PatchTST, Moirai): Good for fixed-horizon forecasting.
- **Encoder-decoder** (Chronos, original Transformer): Classic seq2seq. Flexible but more complex.
- **Decoder-only** (TimesFM, Lag-Llama): Next-patch prediction like GPT. Scales well with data.

### Prediction Head
- **MSE/MAE on point forecasts**: Simple, deterministic.
- **Gaussian head**: Output μ and σ. Simple probabilistic.
- **Student-T mixture**: Heavy tails + multimodality. Best for heavy-tailed data (finance, observability).
- **Quantile regression** (some models): Direct quantile prediction without distributional assumption.

### Loss Function
- **MSE**: Simple, assumes Gaussian errors.
- **NLL of predicted distribution**: More expressive. What NanoTST uses.
- **CRPS** (Continuous Ranked Probability Score): Proper scoring rule for probabilistic forecasts.

---

## Part 3: NanoTST — Your MVP Implementation

**Goal**: ~250 lines of PyTorch. Runs on CPU. Trains on synthetic data. Teaches you every component.

**What we keep**: Patching, decoder-only transformer, causal attention, next-patch prediction, simple probabilistic output.

**What we simplify**: Single variate only (no variate attention), Gaussian output instead of Student-T mixture, simple instance norm instead of causal patch norm, standard LayerNorm instead of RMSNorm, learned positional embeddings instead of RoPE.

### Implementation Plan

```python
"""
NanoTST: A minimal time series transformer for learning.
~250 lines. No dependencies beyond PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================
# CONFIG
# ============================================================
class NanoTSTConfig:
    context_length: int = 512       # input timesteps
    patch_size: int = 32            # timesteps per patch
    d_model: int = 128              # embedding dimension
    n_heads: int = 4                # attention heads
    n_layers: int = 4               # transformer blocks
    dropout: float = 0.1
    prediction_patches: int = 1     # predict next N patches

    @property
    def n_patches(self):
        return self.context_length // self.patch_size  # 512/32 = 16


# ============================================================
# STEP 1: INPUT NORMALIZATION
# ============================================================
# Simple instance norm — normalize entire series by its own stats.
# (Production models do causal patch-based norm; this is the simplified version)
#
# Q to explore: What breaks when you normalize with future info?
# Q to explore: What if the series is highly nonstationary?

class InstanceNorm(nn.Module):
    """Normalize input series to zero mean, unit variance.
       Store stats for denormalization of output."""
    def forward(self, x):
        # x: (batch, seq_len)
        self.mean = x.mean(dim=-1, keepdim=True)
        self.std = x.std(dim=-1, keepdim=True) + 1e-5
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean


# ============================================================
# STEP 2: PATCH EMBEDDING
# ============================================================
# Take groups of P consecutive values → project to d_model dims.
# This is EXACTLY like converting an image to patch tokens in ViT,
# or like a 1D convolution with kernel_size=stride=patch_size.
#
# Why patching? Reduces sequence length (512 → 16 patches),
# captures local temporal patterns, more efficient attention.

class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.proj = nn.Linear(config.patch_size, config.d_model)

    def forward(self, x):
        # x: (batch, seq_len) → (batch, n_patches, patch_size)
        B, L = x.shape
        x = x.view(B, L // self.patch_size, self.patch_size)
        return self.proj(x)  # (batch, n_patches, d_model)


# ============================================================
# STEP 3: POSITIONAL ENCODING
# ============================================================
# Patches need position info. Using simple learned embeddings.
# (Production models use RoPE + XPOS for better extrapolation)

class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos_embed = nn.Embedding(config.n_patches, config.d_model)

    def forward(self, x):
        # x: (batch, n_patches, d_model)
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.pos_embed(positions)


# ============================================================
# STEP 4: CAUSAL SELF-ATTENTION
# ============================================================
# Standard transformer self-attention with causal mask.
# Each patch can only attend to itself and previous patches.
# This is the SAME as GPT attention — just operating on
# patches of time series instead of word tokens.

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention with causal mask
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale

        # Causal mask: prevent attending to future patches
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


# ============================================================
# STEP 5: FEED-FORWARD NETWORK
# ============================================================
# Standard MLP. Modern transformers use SwiGLU; we use GELU for simplicity.
# SwiGLU upgrade: replace GELU with gate mechanism.

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = 4 * config.d_model
        self.fc1 = nn.Linear(config.d_model, hidden)
        self.fc2 = nn.Linear(hidden, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


# ============================================================
# STEP 6: TRANSFORMER BLOCK
# ============================================================
# Pre-norm architecture (norm before attention/FFN, not after).
# Modern transformers use RMSNorm; we use LayerNorm for simplicity.

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# STEP 7: OUTPUT HEAD — Gaussian Probabilistic Prediction
# ============================================================
# Map transformer output → predicted distribution parameters
# for the NEXT patch. We predict μ and log(σ) of a Gaussian.
#
# (Production models use Student-T mixture with K components, each having
#  μ, σ, ν (degrees of freedom), and π (mixture weight).
#  That handles heavy tails and multimodality.)

class GaussianHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mu_proj = nn.Linear(config.d_model, config.patch_size)
        self.log_sigma_proj = nn.Linear(config.d_model, config.patch_size)

    def forward(self, x):
        # x: (batch, n_patches, d_model)
        mu = self.mu_proj(x)          # (batch, n_patches, patch_size)
        log_sigma = self.log_sigma_proj(x)
        sigma = torch.exp(log_sigma.clamp(-10, 10))  # stability
        return mu, sigma


# ============================================================
# STEP 8: FULL MODEL
# ============================================================

class NanoTST(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or NanoTSTConfig()
        c = self.config

        self.norm = InstanceNorm()
        self.patch_embed = PatchEmbedding(c)
        self.pos_enc = PositionalEncoding(c)
        self.blocks = nn.ModuleList([TransformerBlock(c) for _ in range(c.n_layers)])
        self.head = GaussianHead(c)
        self.final_norm = nn.LayerNorm(c.d_model)

    def forward(self, x):
        """
        x: (batch, seq_len) — raw univariate time series
        Returns: mu, sigma — (batch, n_patches, patch_size)
        """
        # Normalize
        x = self.norm(x)

        # Patch + embed
        x = self.patch_embed(x)     # (B, n_patches, d_model)
        x = self.pos_enc(x)

        # Transformer stack
        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        # Predict next-patch distribution
        mu, sigma = self.head(x)
        return mu, sigma

    def loss(self, x):
        """
        Next-patch prediction loss.
        Input x: (batch, seq_len)
        We predict patch[t+1] from the representation at patch[t].
        """
        mu, sigma = self.forward(x)

        # Create target: the next patch for each position
        B, L = x.shape
        P = self.config.patch_size
        x_normed = self.norm(x)  # re-normalize (already cached from forward)
        patches = x_normed.view(B, L // P, P)

        # Shift: predict patches[1:] from representations[:-1]
        pred_mu = mu[:, :-1, :]      # predictions from patches 0..N-2
        pred_sigma = sigma[:, :-1, :]
        targets = patches[:, 1:, :]   # actual patches 1..N-1

        # Gaussian NLL
        nll = 0.5 * torch.log(2 * math.pi * pred_sigma**2) + \
              0.5 * ((targets - pred_mu) / pred_sigma) ** 2

        return nll.mean()

    @torch.no_grad()
    def forecast(self, x, n_steps=64, n_samples=100):
        """
        Autoregressive forecasting.
        Generates n_steps future values by predicting one patch at a time,
        sampling from the predicted distribution, and appending.
        """
        self.eval()
        P = self.config.patch_size
        L = self.config.context_length

        # Collect samples
        all_samples = []
        for _ in range(n_samples):
            current = x.clone()
            generated = []

            steps_needed = (n_steps + P - 1) // P  # patches to generate
            for _ in range(steps_needed):
                # Use last context_length points
                inp = current[:, -L:]
                mu, sigma = self.forward(inp)

                # Sample from last patch prediction
                last_mu = mu[:, -1, :]      # (batch, patch_size)
                last_sigma = sigma[:, -1, :]
                sample = torch.normal(last_mu, last_sigma)

                # Denormalize sample and append
                sample_denorm = self.norm.denormalize(sample)
                generated.append(sample_denorm)
                current = torch.cat([current, sample_denorm], dim=-1)

            forecast = torch.cat(generated, dim=-1)[:, :n_steps]
            all_samples.append(forecast)

        samples = torch.stack(all_samples, dim=0)  # (n_samples, batch, n_steps)
        return {
            'median': samples.median(dim=0).values,
            'mean': samples.mean(dim=0),
            'samples': samples,
            'q10': samples.quantile(0.1, dim=0),
            'q90': samples.quantile(0.9, dim=0),
        }


# ============================================================
# STEP 9: SYNTHETIC DATA GENERATION
# ============================================================
# Generate time series with known patterns to test the model.

def generate_synthetic_data(n_series=1000, length=512, seed=42):
    """Generate synthetic time series with trend + seasonality + noise."""
    torch.manual_seed(seed)
    t = torch.linspace(0, 1, length).unsqueeze(0).expand(n_series, -1)

    # Random parameters per series
    freq = torch.randint(2, 20, (n_series, 1)).float()
    amplitude = torch.rand(n_series, 1) * 2 + 0.5
    trend = (torch.rand(n_series, 1) - 0.5) * 4
    offset = (torch.rand(n_series, 1) - 0.5) * 10

    series = offset + trend * t + amplitude * torch.sin(2 * math.pi * freq * t)
    noise = torch.randn_like(series) * 0.1
    return series + noise


# ============================================================
# STEP 10: TRAINING LOOP
# ============================================================

def train(epochs=50, batch_size=32, lr=3e-4):
    config = NanoTSTConfig()
    model = NanoTST(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    data = generate_synthetic_data(n_series=2000, length=config.context_length)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"NanoTST: {n_params:,} parameters")
    print(f"Config: {config.context_length} context, {config.patch_size} patch, "
          f"{config.d_model}d, {config.n_heads}h, {config.n_layers}L")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(data))
        total_loss = 0
        n_batches = 0

        for i in range(0, len(data) - batch_size, batch_size):
            batch = data[perm[i:i+batch_size]]
            loss = model.loss(batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/n_batches:.4f}")

    return model


if __name__ == "__main__":
    model = train()
    print("\nTraining complete! Try forecasting:")
    print("  test = generate_synthetic_data(1, 512)")
    print("  result = model.forecast(test, n_steps=64)")
```

---

## Part 4: Upgrade Path (After MVP Works)

Once the basic model trains and forecasts, here are upgrades in order of educational value:

### Upgrade 1: SwiGLU FFN (easy, 10 min)
Replace GELU MLP with gated linear unit. This is what modern transformers use.
```python
class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden = 4 * d_model
        self.w1 = nn.Linear(d_model, hidden)
        self.w2 = nn.Linear(d_model, hidden)  # gate
        self.w3 = nn.Linear(hidden, d_model)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
```

### Upgrade 2: RMSNorm (easy, 5 min)
Replace LayerNorm. Faster, no mean subtraction.
```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight
```

### Upgrade 3: Student-T Output Head (medium, 30 min)
Replace Gaussian with Student-T. Handles heavy tails (critical for observability AND financial data).

### Upgrade 4: Causal Patch Normalization (medium, 45 min)
Implement Welford's online algorithm for running mean/variance. This is the key innovation for nonstationary data.

### Upgrade 5: Multivariate + Factorized Attention (harder, 1-2 hours)
Add variate dimension. Implement proportional factorized attention: N time-wise blocks followed by 1 variate-wise block.

### Upgrade 6: RoPE Positional Encoding (medium, 30 min)
Replace learned positions with rotary embeddings for better length extrapolation.

---

## Part 5: Questions to Think About While Building

These are the questions that separate "I implemented a tutorial" from "I understand the architecture":

1. **Why patches instead of point-by-point?** What happens to attention complexity? What local patterns get captured vs lost?

2. **Why decoder-only?** What's the advantage over encoder-decoder for forecasting? (Hint: variable prediction horizons without retraining)

3. **Why factorized attention instead of full space-time?** What's the complexity tradeoff? When would full attention be better?

4. **Why Student-T mixture instead of Gaussian?** What kinds of distributions can't a single Gaussian capture? Think about real-world metrics — error counts, latency spikes, price movements.

5. **Why causal patch normalization?** Try training with RevIN (full-series norm) and see what happens. The model "cheats" by using future statistics.

6. **How would you adapt this for financial data?** What's different about financial time series vs other domains? (Hint: regime changes, fat tails, non-stationarity are even MORE extreme)

7. **Embedding trajectory analysis:** Could you compute embeddings at consecutive patches and track their movement through latent space? What would that tell you about regime shifts?

---

## Key Numbers (Production vs NanoTST)

| | Production Scale | NanoTST |
|---|---|---|
| Parameters | ~150M | ~300K |
| Context length | 4096 | 512 |
| Patch size | 64 | 32 |
| Embedding dim | 768 | 128 |
| Attention heads | 12+ | 4 |
| Layers | 12 (11 time + 1 variate) | 4 (time only) |
| Variates | Multivariate | Univariate |
| Output | Student-T mixture | Gaussian |
| Training data | ~1T points | Synthetic |
