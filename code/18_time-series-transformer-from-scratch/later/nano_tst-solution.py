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
    context_length: int = 512  # input timesteps
    patch_size: int = 32  # timesteps per patch
    d_model: int = 128  # embedding dimension
    n_heads: int = 4  # attention heads
    n_layers: int = 4  # transformer blocks
    dropout: float = 0.1

    @property
    def n_patches(self):
        return self.context_length // self.patch_size


# ============================================================
# INSTANCE NORMALIZATION
# ============================================================
class InstanceNorm(nn.Module):
    """Normalize input series to zero mean, unit variance.
    Stores stats for denormalization of output.
    (Simplified version of causal patch-based norm used in production models)"""

    def forward(self, x):
        # x: (batch, seq_len)
        self.mean = x.mean(dim=-1, keepdim=True)
        self.std = x.std(dim=-1, keepdim=True) + 1e-5
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean


# ============================================================
# PATCH EMBEDDING
# ============================================================
class PatchEmbedding(nn.Module):
    """Convert consecutive timesteps into patch tokens via linear projection.
    Like ViT patches but for 1D time series."""

    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.proj = nn.Linear(config.patch_size, config.d_model)

    def forward(self, x):
        B, L = x.shape
        x = x.view(B, L // self.patch_size, self.patch_size)
        return self.proj(x)


# ============================================================
# POSITIONAL ENCODING
# ============================================================
class PositionalEncoding(nn.Module):
    """Learned positional embeddings for patch positions.
    (Production models use RoPE + XPOS for better extrapolation)"""

    def __init__(self, config):
        super().__init__()
        self.pos_embed = nn.Embedding(config.n_patches, config.d_model)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.pos_embed(positions)


# ============================================================
# CAUSAL SELF-ATTENTION
# ============================================================
class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal mask.
    Each patch can only attend to itself and previous patches."""

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


# ============================================================
# FEED-FORWARD NETWORK
# ============================================================
class FeedForward(nn.Module):
    """Standard MLP with GELU activation.
    (Modern transformers use SwiGLU — see upgrade section in guide)"""

    def __init__(self, config):
        super().__init__()
        hidden = 4 * config.d_model
        self.fc1 = nn.Linear(config.d_model, hidden)
        self.fc2 = nn.Linear(hidden, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


# ============================================================
# TRANSFORMER BLOCK (Pre-Norm)
# ============================================================
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
# GAUSSIAN PREDICTION HEAD
# ============================================================
class GaussianHead(nn.Module):
    """Predict mu and sigma for next patch.
    (Production models use Student-T mixture — handles heavy tails better)"""

    def __init__(self, config):
        super().__init__()
        self.mu_proj = nn.Linear(config.d_model, config.patch_size)
        self.log_sigma_proj = nn.Linear(config.d_model, config.patch_size)

    def forward(self, x):
        mu = self.mu_proj(x)
        log_sigma = self.log_sigma_proj(x)
        sigma = torch.exp(log_sigma.clamp(-10, 10))
        return mu, sigma


# ============================================================
# FULL MODEL
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
        """x: (batch, seq_len) → mu, sigma: (batch, n_patches, patch_size)"""
        x = self.norm(x)
        x = self.patch_embed(x)
        x = self.pos_enc(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        mu, sigma = self.head(x)
        return mu, sigma

    def loss(self, x):
        """Next-patch prediction loss via Gaussian NLL."""
        mu, sigma = self.forward(x)

        B, L = x.shape
        P = self.config.patch_size
        x_normed = (x - self.norm.mean) / self.norm.std  # use cached stats
        patches = x_normed.view(B, L // P, P)

        # Shift: predict patches[1:] from representations[:-1]
        pred_mu = mu[:, :-1, :]
        pred_sigma = sigma[:, :-1, :]
        targets = patches[:, 1:, :]

        # Gaussian NLL
        nll = (
            0.5 * torch.log(2 * math.pi * pred_sigma**2)
            + 0.5 * ((targets - pred_mu) / pred_sigma) ** 2
        )
        return nll.mean()

    @torch.no_grad()
    def forecast(self, x, n_steps=64, n_samples=100):
        """Autoregressive forecasting with probabilistic samples."""
        self.eval()
        P = self.config.patch_size
        L = self.config.context_length

        all_samples = []
        for _ in range(n_samples):
            current = x.clone()
            generated = []

            steps_needed = (n_steps + P - 1) // P
            for _ in range(steps_needed):
                inp = current[:, -L:]
                mu, sigma = self.forward(inp)
                last_mu = mu[:, -1, :]
                last_sigma = sigma[:, -1, :]
                sample = torch.normal(last_mu, last_sigma)
                sample_denorm = self.norm.denormalize(sample)
                generated.append(sample_denorm)
                current = torch.cat([current, sample_denorm], dim=-1)

            forecast = torch.cat(generated, dim=-1)[:, :n_steps]
            all_samples.append(forecast)

        samples = torch.stack(all_samples, dim=0)
        return {
            "median": samples.median(dim=0).values,
            "mean": samples.mean(dim=0),
            "samples": samples,
            "q10": samples.quantile(0.1, dim=0),
            "q90": samples.quantile(0.9, dim=0),
        }


# ============================================================
# SYNTHETIC DATA
# ============================================================
def generate_synthetic_data(n_series=1000, length=512, seed=42):
    """Trend + seasonality + noise."""
    torch.manual_seed(seed)
    t = torch.linspace(0, 1, length).unsqueeze(0).expand(n_series, -1)

    freq = torch.randint(2, 20, (n_series, 1)).float()
    amplitude = torch.rand(n_series, 1) * 2 + 0.5
    trend = (torch.rand(n_series, 1) - 0.5) * 4
    offset = (torch.rand(n_series, 1) - 0.5) * 10

    series = offset + trend * t + amplitude * torch.sin(2 * math.pi * freq * t)
    noise = torch.randn_like(series) * 0.1
    return series + noise


# ============================================================
# TRAINING
# ============================================================
def train(epochs=50, batch_size=32, lr=3e-4):
    config = NanoTSTConfig()
    model = NanoTST(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    data = generate_synthetic_data(n_series=2000, length=config.context_length)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"NanoTST: {n_params:,} parameters")
    print(
        f"Config: {config.context_length} ctx, {config.patch_size} patch, "
        f"{config.d_model}d, {config.n_heads}h, {config.n_layers}L"
    )
    print(f"Patches per series: {config.n_patches}")
    print()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(data))
        total_loss = 0
        n_batches = 0

        for i in range(0, len(data) - batch_size, batch_size):
            batch = data[perm[i : i + batch_size]]
            loss = model.loss(batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / n_batches
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f}")

    return model


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    model = train()

    print("\n--- Forecasting test ---")
    test_data = generate_synthetic_data(n_series=1, length=512, seed=99)
    result = model.forecast(test_data, n_steps=64, n_samples=50)
    print(f"Input shape:    {test_data.shape}")
    print(f"Forecast shape: {result['median'].shape}")
    print(f"Median forecast (first 8): {result['median'][0, :8].tolist()}")
    print(f"Q10-Q90 range:  {(result['q90'] - result['q10']).mean().item():.4f}")
    print("\nDone! Try modifying the architecture and see what changes.")
