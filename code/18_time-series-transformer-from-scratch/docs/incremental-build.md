# NanoTST: Incremental Build Guide

Build a time series transformer from nothing, one concept at a time.
Each step is runnable. Don't skip ahead — the learning is in the progression.

You'll work in a single file: `nano_tst_build.py`. Keep adding to it at each step.

---

## Step 1: Data + Patching — Understand the Representation (10 min)

**Goal:** See what the model will actually work with.

**Why patching?** A 512-point series = 512 tokens point-by-point. With patch_size=32, you get 16 tokens. Attention is O(n²), so 16² = 256 vs 512² = 262,144 operations. Patching also captures local temporal structure within each chunk.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

PATCH_SIZE = 32
CONTEXT_LEN = 512

def generate_data(n_series=1000, length=512, seed=42):
    """Synthetic time series: trend + seasonality + noise."""
    torch.manual_seed(seed)
    t = torch.linspace(0, 1, length).unsqueeze(0).expand(n_series, -1)
    freq = torch.randint(2, 20, (n_series, 1)).float()
    amplitude = torch.rand(n_series, 1) * 2 + 0.5
    trend = (torch.rand(n_series, 1) - 0.5) * 4
    offset = (torch.rand(n_series, 1) - 0.5) * 10
    series = offset + trend * t + amplitude * torch.sin(2 * math.pi * freq * t)
    return series + torch.randn_like(series) * 0.1

data = generate_data()
print(f"Generated: {data.shape}")  # [1000, 512]

# What patching looks like
one_series = data[0]                              # [512]
patches = one_series.view(-1, PATCH_SIZE)          # [16, 32]
print(f"One series → {patches.shape[0]} patches of {patches.shape[1]} values")
print(f"Patch 0 (first 5): {patches[0, :5].tolist()}")
print(f"Patch 1 (first 5): {patches[1, :5].tolist()}")
```

**Run:** `python nano_tst_build.py`

**Notice:** Each patch is a contiguous chunk. The model will see the series as 16 "tokens," each containing 32 raw values. Next-patch prediction means: given patches 0-14, predict patches 1-15.

---

## Step 2: Linear Baseline — The Floor (15 min)

**Goal:** The dumbest possible next-patch predictor. Everything you add later must beat this.

This model sees ONE patch at a time with zero context from other patches. Just `linear(current_patch) → next_patch`.

```python
class LinearBaseline(nn.Module):
    def __init__(self, patch_size=32):
        super().__init__()
        self.ps = patch_size
        self.proj = nn.Linear(patch_size, patch_size)

    def forward_and_loss(self, x):
        # x: (batch, seq_len)
        B, L = x.shape
        patches = x.view(B, L // self.ps, self.ps)   # (B, 16, 32)
        pred = self.proj(patches)                      # (B, 16, 32)
        # Next-patch prediction: pred[t] should match patches[t+1]
        loss = ((pred[:, :-1] - patches[:, 1:]) ** 2).mean()
        return loss

def train_model(model, data, epochs=30, batch_size=32, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        perm = torch.randperm(len(data))
        total_loss, n = 0, 0
        for i in range(0, len(data) - batch_size, batch_size):
            loss = model.forward_and_loss(data[perm[i:i+batch_size]])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); n += 1
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d} | MSE: {total_loss/n:.6f}")

print("\n=== LINEAR BASELINE ===")
data = generate_data(2000)
baseline = LinearBaseline()
train_model(baseline, data)
# Write down this final loss — it's the number to beat.
```

**Notice:** Loss drops but plateaus quickly. The model can learn simple patch-to-patch correlations, but it has ZERO awareness of where it is in the series or what came before. A sine wave's next patch depends heavily on position — this model can't know that.

---

## Step 3: Normalization + Patch Embedding — Proper Input Pipeline (15 min)

**Goal:** Two additions: (1) normalize inputs so the model doesn't waste capacity on scale/offset, and (2) project patches into a higher-dimensional embedding space where the transformer can work.

**Why normalize?** Your synthetic data has random offsets (-5 to +5) and random scales. Without normalization, the model has to learn to handle all of these, wasting capacity. Instance norm makes every series zero-mean, unit-variance.

**Why embed?** A 32-dim patch is too small for attention to work in. Projecting to 128-dim gives the model room to represent richer features per patch.

```python
class InstanceNorm(nn.Module):
    """Normalize each series independently. Store stats for later denorm."""
    def forward(self, x):
        self.mean = x.mean(dim=-1, keepdim=True)
        self.std = x.std(dim=-1, keepdim=True) + 1e-5
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

class PatchEmbedding(nn.Module):
    """Reshape into patches, project to d_model dimensions."""
    def __init__(self, patch_size=32, d_model=128):
        super().__init__()
        self.ps = patch_size
        self.proj = nn.Linear(patch_size, d_model)

    def forward(self, x):
        B, L = x.shape
        x = x.view(B, L // self.ps, self.ps)   # (B, n_patches, patch_size)
        return self.proj(x)                      # (B, n_patches, d_model)

# Quick test — verify shapes
norm = InstanceNorm()
embed = PatchEmbedding()
sample = data[:2]                               # (2, 512)
normed = norm(sample)                           # (2, 512) — same shape, different values
embedded = embed(normed)                        # (2, 16, 128)
print(f"\nRaw: {sample.shape} → Normed: {normed.shape} → Embedded: {embedded.shape}")
print(f"Before norm — mean: {sample[0].mean():.2f}, std: {sample[0].std():.2f}")
print(f"After norm  — mean: {normed[0].mean():.4f}, std: {normed[0].std():.4f}")
```

**Notice:** After normalization, every series has mean ~0 and std ~1 regardless of its original scale. The embedding projects each 32-dim patch into 128-dim space — this is where the transformer will do its work.

---

## Step 4: Self-Attention — The Core Idea (25 min)

**Goal:** Implement single-head self-attention from scratch. This is the key component. When patch 15 attends to patch 3, it's asking: "what happened back there that helps me predict what comes next?"

**Take your time here.** Understand every line. The shapes are the map.

```python
class SelfAttention(nn.Module):
    """Single-head self-attention with causal mask."""
    def __init__(self, d_model=128):
        super().__init__()
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, 3 * d_model)   # project to Q, K, V at once
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape                              # (batch, n_patches, d_model)

        # Project to Q, K, V
        qkv = self.qkv(x)                              # (B, T, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)                 # each (B, T, D)

        # Attention scores: how much does each patch care about each other patch?
        scale = math.sqrt(self.d_model)
        attn = (q @ k.transpose(-2, -1)) / scale       # (B, T, T)

        # CAUSAL MASK: patch t can only see patches 0..t (not future)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)                  # (B, T, T) — rows sum to 1
        self.last_attn = attn.detach()                   # save for visualization later

        # Weighted combination of values
        out = attn @ v                                   # (B, T, D)
        return self.out_proj(out)
```

**Now build a minimal model using attention:**

```python
class AttentionModel(nn.Module):
    """Norm → Embed → Attention → Predict. No FFN, no residual, no stacking yet."""
    def __init__(self, patch_size=32, d_model=128):
        super().__init__()
        self.ps = patch_size
        self.norm = InstanceNorm()
        self.embed = PatchEmbedding(patch_size, d_model)
        self.attn = SelfAttention(d_model)
        self.head = nn.Linear(d_model, patch_size)  # project back to patch space

    def forward_and_loss(self, x):
        B, L = x.shape
        x_norm = self.norm(x)
        patches = x_norm.view(B, L // self.ps, self.ps)  # target patches

        h = self.embed(x_norm)        # (B, 16, 128)
        h = self.attn(h)              # (B, 16, 128) — now patches can see each other
        pred = self.head(h)           # (B, 16, 32) — predict in patch space

        # Next-patch loss
        loss = ((pred[:, :-1] - patches[:, 1:]) ** 2).mean()
        return loss

print("\n=== ATTENTION MODEL (single head, no FFN) ===")
attn_model = AttentionModel()
n_params = sum(p.numel() for p in attn_model.parameters())
print(f"Parameters: {n_params:,}")
train_model(attn_model, data)
```

**What to notice:**
- Compare the final loss to your linear baseline. Attention should win because it can use CONTEXT from previous patches.
- Think about what the causal mask does: patch 0 can only see itself. Patch 15 can see all 16 patches. Later patches have richer context.
- **Try removing the causal mask** (comment out the two mask lines). Loss drops further — but it's CHEATING by looking at future patches. In forecasting, you can't see the future.

---

## Step 5: Multi-Head + FFN + LayerNorm = Full Transformer Block (20 min)

**Goal:** Three upgrades that turn raw attention into a proper transformer block.

**Why multi-head?** Different heads can learn different patterns. One head might track trend, another might track periodicity at different scales.

**Why FFN?** Attention mixes information across patches. FFN processes each patch independently — it's where per-position computation happens. Think of attention as "gather info" and FFN as "process info."

**Why residual + LayerNorm?** Residuals let gradients flow through deep networks. LayerNorm stabilizes training. Pre-norm (norm before attention, not after) is what modern transformers use.

```python
class MultiHeadCausalAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads    # 128/4 = 32 per head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        # Project and reshape for multi-head
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)         # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale   # (B, heads, T, T)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        self.last_attn = attn.detach()

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        hidden = 4 * d_model
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN → Attn → residual → LN → FFN → residual"""
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))   # residual around attention
        x = x + self.ffn(self.norm2(x))    # residual around FFN
        return x
```

**Now build the transformer model with stacked blocks:**

```python
class TransformerModel(nn.Module):
    def __init__(self, patch_size=32, d_model=128, n_heads=4, n_layers=4):
        super().__init__()
        self.ps = patch_size
        self.norm = InstanceNorm()
        self.embed = PatchEmbedding(patch_size, d_model)
        self.pos = nn.Embedding(CONTEXT_LEN // patch_size, d_model)  # learned positions
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, patch_size)

    def forward(self, x):
        B, L = x.shape
        x_norm = self.norm(x)
        h = self.embed(x_norm)                                     # (B, 16, 128)
        h = h + self.pos(torch.arange(h.size(1), device=h.device)) # add position info
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        return self.head(h)                                         # (B, 16, 32)

    def forward_and_loss(self, x):
        pred = self.forward(x)
        B, L = x.shape
        x_norm = (x - self.norm.mean) / self.norm.std  # use cached stats
        patches = x_norm.view(B, L // self.ps, self.ps)
        return ((pred[:, :-1] - patches[:, 1:]) ** 2).mean()

print("\n=== FULL TRANSFORMER (4 layers, 4 heads, MSE) ===")
transformer = TransformerModel()
n_params = sum(p.numel() for p in transformer.parameters())
print(f"Parameters: {n_params:,}")
train_model(transformer, data, epochs=50, lr=3e-4)
```

**What to notice:**
- This should beat both previous models. Compare the three final losses.
- You now have the complete transformer architecture — the same structure as GPT, just operating on time series patches instead of word tokens.
- **Try n_layers=1** and see how much worse it is. Depth matters for learning complex temporal patterns.

---

## Step 6: Gaussian Head — Go Probabilistic (15 min)

**Goal:** Stop predicting exact values. Predict a *distribution* over possible next patches.

**Why?** Time series are inherently uncertain. A sine wave at its peak could go slightly higher or start descending. A Gaussian head outputs mu (best guess) and sigma (how confident). The loss becomes negative log-likelihood instead of MSE.

**Key insight:** MSE is actually a special case of Gaussian NLL where sigma=1 everywhere. By learning sigma, the model can say "I'm confident here" (small sigma) and "could go either way" (large sigma).

```python
class GaussianHead(nn.Module):
    def __init__(self, d_model=128, patch_size=32):
        super().__init__()
        self.mu_proj = nn.Linear(d_model, patch_size)
        self.log_sigma_proj = nn.Linear(d_model, patch_size)

    def forward(self, x):
        mu = self.mu_proj(x)
        log_sigma = self.log_sigma_proj(x)
        sigma = torch.exp(log_sigma.clamp(-10, 10))  # keep stable
        return mu, sigma
```

**Modify TransformerModel** — replace the linear head and loss:

```python
class NanoTST(nn.Module):
    def __init__(self, patch_size=32, d_model=128, n_heads=4, n_layers=4):
        super().__init__()
        self.ps = patch_size
        self.norm = InstanceNorm()
        self.embed = PatchEmbedding(patch_size, d_model)
        self.pos = nn.Embedding(CONTEXT_LEN // patch_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = GaussianHead(d_model, patch_size)

    def forward(self, x):
        x_norm = self.norm(x)
        h = self.embed(x_norm)
        h = h + self.pos(torch.arange(h.size(1), device=h.device))
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        return self.head(h)  # returns (mu, sigma)

    def forward_and_loss(self, x):
        mu, sigma = self.forward(x)
        B, L = x.shape
        x_norm = (x - self.norm.mean) / self.norm.std
        patches = x_norm.view(B, L // self.ps, self.ps)

        pred_mu = mu[:, :-1]
        pred_sigma = sigma[:, :-1]
        targets = patches[:, 1:]

        # Gaussian NLL: -log p(target | mu, sigma)
        nll = 0.5 * torch.log(2 * math.pi * pred_sigma**2) + \
              0.5 * ((targets - pred_mu) / pred_sigma) ** 2
        return nll.mean()

print("\n=== NanoTST (Gaussian head) ===")
model = NanoTST()
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")
train_model(model, data, epochs=50, lr=3e-4)
```

**What to notice:**
- NLL loss values look different from MSE — don't compare the raw numbers. They can even be negative (that's fine, it means the model assigns high probability to the correct values).
- The model now outputs uncertainty. In regions where the pattern is predictable, sigma will be small. Where it's noisy, sigma grows.

---

## Step 7: Grammar Test — Watch It Learn (20 min)

**Goal:** Test the model on specific patterns during training and watch the progression. This is where you build real intuition.

The "grammar" of time series: flat → linear trend → sine wave. Like a language model learning articles before grammar before idioms.

```python
def make_test_cases():
    return {
        "flat":  torch.ones(1, 512) * 3.0,
        "line":  torch.linspace(0, 5, 512).unsqueeze(0),
        "sine":  torch.sin(torch.linspace(0, 8*math.pi, 512)).unsqueeze(0),
        "noisy": torch.sin(torch.linspace(0, 8*math.pi, 512)).unsqueeze(0) + torch.randn(1, 512) * 0.2,
    }

def grammar_test(model, test_cases):
    """Print predictions vs reality for each pattern."""
    model.eval()
    for name, series in test_cases.items():
        mu, sigma = model(series)
        pred = mu[0, -1, :6].tolist()       # predicted next patch (first 6 values)
        conf = sigma[0, -1, :6].mean().item()  # average uncertainty
        print(f"  {name:8s} | pred: [{', '.join(f'{v:.2f}' for v in pred)}] | sigma: {conf:.3f}")
    model.train()

def train_with_grammar(model, data, epochs=100, batch_size=32, lr=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    test_cases = make_test_cases()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"NanoTST: {n_params:,} parameters\n")

    step = 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(data))
        total_loss, n = 0, 0

        for i in range(0, len(data) - batch_size, batch_size):
            loss = model.forward_and_loss(data[perm[i:i+batch_size]])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); n += 1
            step += 1

        if (epoch + 1) % 10 == 0:
            avg = total_loss / n
            print(f"Epoch {epoch+1:3d} | Loss: {avg:.4f} | Step: {step}")
            grammar_test(model, test_cases)
            print()

print("\n=== TRAINING WITH GRAMMAR TEST ===")
model = NanoTST()
data = generate_data(2000)
train_with_grammar(model, data, epochs=100)
```

**What to watch for (this is the important part):**

1. **Early training (epochs 1-20):** Predictions are garbage for everything. The model hasn't learned any patterns yet.

2. **Mid training (epochs 20-50):** Flat is nailed first (just predict the mean). Line gets close. Sine is still bad — the model is learning the easy patterns before the hard ones.

3. **Late training (epochs 50-100):** Sine predictions improve. Watch the sigma values — they should be small for flat/line (confident) and larger for noisy signals (appropriately uncertain).

4. **The progression:** flat → line → sine mirrors how the model builds understanding. Constants are trivial. Trends require tracking direction. Periodicity requires tracking phase — the hardest pattern.

---

## Step 8: Autoregressive Forecasting (15 min)

**Goal:** Use the trained model to forecast into the future by feeding its own predictions back in.

This is where the model proves it actually learned something — it generates plausible continuations of time series.

```python
@torch.no_grad()
def forecast(model, x, n_steps=64, n_samples=50):
    """Generate future values by predicting one patch at a time."""
    model.eval()
    P = model.ps
    L = CONTEXT_LEN
    all_samples = []

    for _ in range(n_samples):
        current = x.clone()
        generated = []

        patches_needed = (n_steps + P - 1) // P
        for _ in range(patches_needed):
            inp = current[:, -L:]
            mu, sigma = model(inp)
            # Sample from predicted distribution for last patch
            sample = torch.normal(mu[:, -1], sigma[:, -1])
            # Denormalize and append
            sample_real = model.norm.denormalize(sample)
            generated.append(sample_real)
            current = torch.cat([current, sample_real], dim=-1)

        all_samples.append(torch.cat(generated, dim=-1)[:, :n_steps])

    samples = torch.stack(all_samples)
    return {
        'median': samples.median(dim=0).values,
        'mean': samples.mean(dim=0),
        'q10': samples.quantile(0.1, dim=0),
        'q90': samples.quantile(0.9, dim=0),
    }

# Test forecasting on each pattern
print("\n=== FORECASTING ===")
test_cases = make_test_cases()
for name, series in test_cases.items():
    result = forecast(model, series, n_steps=64, n_samples=30)
    spread = (result['q90'] - result['q10']).mean().item()
    # Compare forecast to what the series "should" do
    print(f"{name:8s} | forecast first 4: {result['median'][0,:4].tolist()} | spread: {spread:.3f}")
```

**What to notice:**
- **Flat:** Forecast should be ~3.0 continuing. Tight spread (high confidence).
- **Line:** Should continue the upward trend. Spread grows over time (uncertainty compounds).
- **Sine:** Should continue the wave. This is the hardest — errors in phase compound quickly.
- **Spread grows with horizon:** Every model gets less certain the further it predicts. This is correct behavior.

---

## Step 9: Experiments — Build Intuition (30+ min)

Now you have a working model. Time to break things and see what happens.

### Experiment 1: Depth

```python
print("\n=== EXPERIMENT: DEPTH ===")
for n_layers in [1, 2, 4, 8]:
    print(f"\n--- {n_layers} layers ---")
    m = NanoTST(n_layers=n_layers)
    train_model(m, data, epochs=50, lr=3e-4)
```
**Question:** How many layers does the model need to learn sine? Does 1 layer ever get there?

### Experiment 2: Width

```python
print("\n=== EXPERIMENT: WIDTH ===")
for d_model in [32, 128, 256]:
    print(f"\n--- d_model={d_model} ---")
    m = NanoTST(d_model=d_model)
    train_model(m, data, epochs=50, lr=3e-4)
```
**Question:** When does more width stop helping? Is depth or width more important for temporal patterns?

### Experiment 3: Patch Size

```python
print("\n=== EXPERIMENT: PATCH SIZE ===")
for ps in [8, 32, 64]:
    print(f"\n--- patch_size={ps} ---")
    m = NanoTST(patch_size=ps)
    n_patches = CONTEXT_LEN // ps
    m.pos = nn.Embedding(n_patches, 128)  # adjust position embedding
    train_model(m, data, epochs=50, lr=3e-4)
```
**Question:** Smaller patches = more tokens = more compute but finer resolution. Bigger patches = fewer tokens = faster but coarser. Where does sine prediction break?

### Experiment 4: Generalization

```python
print("\n=== EXPERIMENT: GENERALIZATION ===")
# Train ONLY on simple patterns (no sine), test on sine
simple_data = generate_data(2000, seed=0)
# Hack: overwrite with just trends, no oscillation
t = torch.linspace(0, 1, 512).unsqueeze(0).expand(2000, -1)
trend = (torch.rand(2000, 1) - 0.5) * 4
offset = (torch.rand(2000, 1) - 0.5) * 10
simple_data = offset + trend * t + torch.randn(2000, 512) * 0.1

m = NanoTST()
train_model(m, simple_data, epochs=50, lr=3e-4)
print("\nTest on UNSEEN sine wave:")
grammar_test(m, make_test_cases())
```
**Question:** Can a model trained without periodic data forecast sine waves? (Spoiler: no. This is why training data composition matters so much.)

---

## Step 10: Attention Visualization (15 min)

**Goal:** Peek inside the model to see WHAT it attends to.

The attention saved in `self.last_attn` shows you, for each patch, how much weight it puts on every previous patch. For a sine wave, you'd expect the model to attend to patches that are one period back.

```python
print("\n=== ATTENTION PATTERNS ===")
model.eval()
sine = make_test_cases()["sine"]
mu, sigma = model(sine)

# Look at last layer, first head
attn = model.blocks[-1].attn.last_attn[0, 0]   # (n_patches, n_patches)
print(f"Attention shape: {attn.shape}")
print("\nWhich patches does each patch attend to?")
for i in range(attn.size(0)):
    weights = attn[i, :i+1]  # only causal (up to current)
    top_k = weights.topk(min(3, len(weights)))
    top_patches = [(idx.item(), w.item()) for idx, w in zip(top_k.indices, top_k.values)]
    print(f"  Patch {i:2d} → {', '.join(f'patch {p}({w:.2f})' for p,w in top_patches)}")
```

**What to look for:**
- Does the model attend to periodic patches (patches that are one wavelength apart)?
- Does the first patch attend only to itself (it has no choice)?
- Do different heads attend to different things? Try other heads: `model.blocks[-1].attn.last_attn[0, 1]`

---

## Summary: What You Built and What You Learned

| Step | Component | Key Insight |
|------|-----------|-------------|
| 1 | Data + Patches | Patching reduces sequence length 32x, captures local structure |
| 2 | Linear baseline | No context = can't predict position-dependent patterns |
| 3 | Norm + Embedding | Normalization removes scale; embedding gives capacity |
| 4 | Self-attention | Context from all past patches; causal mask prevents cheating |
| 5 | Multi-head + FFN | Heads specialize; FFN processes; residuals enable depth |
| 6 | Gaussian head | Predict distributions, not points; model knows its uncertainty |
| 7 | Grammar test | Learning order: constant → trend → periodicity (easy → hard) |
| 8 | Forecasting | Autoregressive sampling; uncertainty grows with horizon |
| 9 | Experiments | Depth matters for periodicity; width has diminishing returns |
| 10 | Attention viz | The model discovers temporal structure in its attention patterns |

## What This Teaches You About Production Time Series Transformers

Now you can reason about why production models make their choices:

- **Causal patch norm** instead of InstanceNorm → handles non-stationarity without future leakage
- **RoPE** instead of learned positions → extrapolates to longer sequences than seen in training
- **SwiGLU** instead of GELU → better gradient flow, standard in modern transformers
- **Student-T mixture** instead of Gaussian → handles heavy tails (latency spikes, price jumps)
- **Factorized attention** (time + variate) → scales to thousands of variates efficiently
- **RMSNorm** instead of LayerNorm → faster, no mean subtraction needed

Each of these is an upgrade you can now implement because you understand what it replaces and why.
