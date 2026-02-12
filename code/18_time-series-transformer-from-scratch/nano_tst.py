from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

PATCH_SIZE = 32  # number of time series points per "token"
CONTEXT_LEN = 512  # length of series, is context length the right name here?


def generate_data(n_series=1000, length=512, seed=77):
    """
    Generate synthetic time series data.
    """
    torch.manual_seed(seed)
    t = torch.linspace(0, 1, length).unsqueeze(0).expand(n_series, -1)
    freq = torch.randint(2, 20, (n_series, 1)).float()
    amplitude = torch.rand(n_series, 1) * 2 + 0.5
    trend = torch.rand(n_series, 1) - 0.5
    offset = (torch.rand(n_series, 1) - 0.5) * 10
    # Put it together to make a time series
    series = offset + trend * t + amplitude * torch.sin(2 * math.pi * freq * t)

    return series + torch.randn_like(series) * 0.1


# Step 1 - generate data
# data = generate_data()
# print(f"Generated: {data.shape}")

# one_series = data[0]
# patches = one_series.view(-1, PATCH_SIZE)


# Step 2 - Linear Baseline - one patch
class LinearBaseline(nn.Module):
    def __init__(self, patch_size=32):
        super().__init__()
        self.ps = patch_size
        self.proj = nn.Linear(patch_size, patch_size)  # 32x32 linear layer?

    def forward_and_loss(self, x):
        # x: (batch, seq_len)
        batch_size, seq_len = x.shape
        patches = x.view(batch_size, seq_len // self.ps, self.ps)
        # (batch, 16, 32)
        pred = self.proj(patches)
        # Next-patch prediction
        # Difference between the predict value from actual
        loss = ((pred[:, :-1] - patches[:, 1:]) ** 2).mean()
        # TODO - this just does MSE?
        return loss


def train_model(model, data, epochs=250, batch_size=32, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        perm = torch.randperm(len(data))  # TODO - what is this?
        total_loss, n = 0, 0
        for i in range(0, len(data) - batch_size, batch_size):
            # Pass in one batch at a time
            loss = model.forward_and_loss(data[perm[i : i + batch_size]])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d} | MSE: {total_loss/n:.6f}")


# Step 2 - Baseline with single linear layer
# print("\n === LINEAR BASELINE ===")
# data = generate_data(2000)
# baseline = LinearBaseline()
# train_model(baseline, data, epochs=1000)
# Around 500 epochs, saturates at MSE 0.99 - 1.00


# Step 3 - Normalize and Patch Embedding
# Normalize - so model doesnt have to learn scale
# Embed - the 32 dim patch is too small for learning
# complex features during attention


class InstanceNorm(nn.Module):
    """
    Normalize each series, store stats to denormalize.
    """

    def forward(self, x):
        self.mean = x.mean(dim=-1, keepdim=True)
        self.std = x.std(dim=-1, keepdim=True) + 1e-5  # TODO - why add this?
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean


class PatchEmbedding(nn.Module):
    """
    Reshape into patches, project into higher dimensions
    """

    def __init__(self, patch_size=32, d_model=128):
        super().__init__()
        self.ps = patch_size
        self.proj = nn.Linear(patch_size, d_model)
        # TODO - is my comment correct?
        # takes all 32 values in our patch, projects into higher dimension

    def forward(self, x):
        B, L = x.shape
        x = x.view(B, L // self.ps, self.ps)
        # TODO - what does view() do? why do we reshape?
        return self.proj(x)


# Step 3 - verify shapes
# data = generate_data(2000)
# norm = InstanceNorm()
# embed = PatchEmbedding()

# # Example
# sample = data[:2]
# normed = norm(sample)
# # Embedding layer - for each sample -> array of 128d vector for each patch
# embedded = embed(normed)

# print(f"\nRaw: {sample.shape} → Normed: {normed.shape} → Embedded: {embedded.shape}")
# print(f"Before norm — mean: {sample[0].mean():.2f}, std: {sample[0].std():.2f}")
# print(f"After norm  — mean: {normed[0].mean():.4f}, std: {normed[0].std():.4f}")


# Step 4 - Self attention
class SelfAttention(nn.Module):
    """
    Single-head self-attention with causal mask.
    """

    # TODO what does single-head mean? how do heads relate to the Transformer blocks
    def __init__(self, d_model=128):
        super().__init__()
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # TODO - why this shape? qkv is the 3 matrices for
        # query key and values, why is i 128 x 3*128?
        self.out_proj = nn.Linear(d_model, d_model)
        # TODO - what happens here in pytorch? are my layers
        # connected b.c of how I init in the module in the order
        # what are the rules for connection layers out == in of next?

    def forward(self, x):
        # TODO - every class has a forward() - thats a nn.Module thing?
        B, T, D = x.shape
        # (batch, n_patches, d_model)
        # each batch, has 128 embedding vector for every patch
        # each batch, has a number of patches, each patch has it's embedding vector
        qkv = self.qkv(x)
        # (batch, number of patches, 3*128d)
        # I see now we have the 3 vectors for each patch in each batch
        # patches are our tokens - each token has a q, k, v vector
        # TODO - do qkv have to be same size? are they equally as important?
        q, k, v = qkv.chunk(3, dim=-1)
        # TODO - what is chunk doing here?

        # Attention Scores - how much does a patch care about another patch?
        scale = math.sqrt(self.d_model)
        attn = (q @ k.transpose(-2, -1)) / scale

        # CAUSAL MASK: patch t can only see patches 0..t (never a future patch)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        # TODO - what is triu doing?
        # It gives the diagonal of 1s that we fill with inf?

        attn = F.softmax(attn, dim=-1)
        # TODO - Softmax makes rows sum to 1? why do we do that?
        self.last_attn = attn.detach()  # for visualization

        # Weighted combination of values
        out = attn @ v
        return self.out_proj(out)


# Now build the model
class AttentionModel(nn.Module):
    """
    Normalize -> Embed -> Attnetion -> Predict
    """

    def __init__(self, patch_size=32, d_model=128):
        super().__init__()
        self.ps = patch_size
        self.norm = InstanceNorm()
        self.embed = PatchEmbedding(patch_size, d_model)
        self.attn = SelfAttention(d_model)
        self.head = nn.Linear(d_model, patch_size)
        # TODO - what does this head layer do?

    def forward_and_loss(self, x):
        B, L = x.shape
        x_norm = self.norm(x)
        patches = x_norm.view(B, L // self.ps, self.ps)

        h = self.embed(x_norm)
        h = self.attn(h)
        pred = self.head(h)

        # Loss
        loss = ((pred[:, :-1] - patches[:, 1:]) ** 2).mean()
        return loss


# Step 4 attenoin model
# data = generate_data(n_series=2000)
# print("\n=== ATTENTION MODEL (single head, no FFN) ===")
# attn_model = AttentionModel(patch_size=16, d_model=128)
# n_params = sum(p.numel() for p in attn_model.parameters())
# print(f"Params: {n_params:,}")
# train_model(attn_model, data)


# Step 5 - multihead, FFN, residual
# multiple heads - each head can learn different patterns
# FFN - per position processing? # TODO - what is this?
# residuals?
# layer norm - stabilizes training


class MultiHeadCausalAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # TODO - output size spread out per head? is this necessary?
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        # TODO - why is this the output size?

    def forward(self, x):
        B, T, D = x.shape
        # Project and reshape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # TODO - what is permute(), whats happening here? why?
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        self.last_attn = attn.detach()  # for visual

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        # TODO - why the reshape, whats happening here
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        hidden = 4 * d_model
        # TODO - why hidden? what is this?
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        # oh its the hidden layer of our network

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))
        # TODO what is gelu?


class TransformerBlock(nn.Module):
    """
    LN - > Attn -> residual -> LN -> FFN -> residual
    # TODO - what are these and why this order?
    """

    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        # TODO - what is this?
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerModel(nn.Module):
    def __init__(self, patch_size=32, d_model=128, n_heads=4, n_layers=4):
        super().__init__()
        self.ps = patch_size
        self.norm = InstanceNorm()
        self.embed = PatchEmbedding(patch_size, d_model)
        self.pos = nn.Embedding(CONTEXT_LEN // patch_size, d_model)
        # TODO how does this learn positions?
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        # TODO -- what does layer norm do?
        self.head = nn.Linear(d_model, patch_size)

    def forward(self, x):
        B, L = x.shape
        x_norm = self.norm(x)
        h = self.embed(x_norm)
        h = h + self.pos(torch.arange(h.size(1), device=h.device))
        # TODO - how does this just "add" position info
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        return self.head(h)

    def forward_and_loss(self, x):
        pred = self.forward(x)
        B, L = x.shape
        x_norm = (x - self.norm.mean) / self.norm.std
        patches = x_norm.view(B, L // self.ps, self.ps)
        return ((pred[:, :-1] - patches[:, 1:]) ** 2).mean()


# Step 5
# data = generate_data(n_series=1000)
# print("\n=== FULL TRANSFORMER (4 layers, 4 heads, MSE) ===")
# transformer = TransformerModel()
# n_params = sum(p.numel() for p in transformer.parameters())
# print(f"Parameters: {n_params:,}")
# train_model(transformer, data, epochs=50, lr=3e-4)


# Step 6 - Gaussian head
class GaussianHead(nn.Module):
    def __init__(self, d_model=128, patch_size=32):
        super().__init__()
        self.mu_proj = nn.Linear(d_model, patch_size)
        self.log_sigma_proj = nn.Linear(d_model, patch_size)

    def forward(self, x):
        mu = self.mu_proj(x)
        log_sigma = self.log_sigma_proj(x)
        sigma = torch.exp(log_sigma.clamp(-10, 10))

        return mu, sigma


class NanoTST(nn.Module):
    def __init__(self, patch_size=32, d_model=128, n_heads=4, n_layers=4):
        super().__init__()
        self.ps = patch_size
        self.norm = InstanceNorm()
        self.embed = PatchEmbedding(patch_size, d_model)
        self.pos = nn.Embedding(CONTEXT_LEN // patch_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = GaussianHead(d_model, patch_size)

    def forward(self, x):
        x_norm = self.norm(x)
        h = self.embed(x_norm)
        h = h + self.pos(torch.arange(h.size(1), device=h.device))
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        return self.head(h)  # now returns (mu, sigma)

    def forward_and_loss(self, x):
        mu, sigma = self.forward(x)
        B, L = x.shape
        x_norm = (x - self.norm.mean) / self.norm.std
        patches = x_norm.view(B, L // self.ps, self.ps)

        pred_mu = mu[:, :-1]
        pred_sigma = sigma[:, :-1]
        targets = patches[:, 1:]

        # Gaussian NLL as loss metric
        nll = (
            0.5 * torch.log(2 * math.pi * pred_sigma**2)
            + 0.5 * ((targets - pred_mu) / pred_sigma) ** 2
        )
        return nll.mean()


# print("\n=== NanoTST (Gaussian head) ===")
# data = generate_data()
# model = NanoTST()
# n_params = sum(p.numel() for p in model.parameters())
# print(f"Parameters: {n_params:,}")
# train_model(model, data, epochs=50, lr=3e-4)

# Step 7 - Grammar eval
from ts_grammar_eval import train_with_grammar

print("\n=== TRAINING WITH GRAMMAR TEST ===")
data = generate_data(n_series=2000)
model = NanoTST()
train_with_grammar(model, data, epochs=100)


# Step 8 - auto regressive
# Now we see if model can build on its predictions to make something coherent
def forecast(model, x, n_steps=64, n_samples=50):
    """
    Generate future values by predicting one patch at a time.
    """
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
            sample = torch.normal(mu[:, -1], sigma[:, -1])
            # denormalize
            sample_real = model.norm.denormalize(sample)
            generated.append(sample_real)
            current = torch.cat([current, sample_real], dim=-1)

        all_samples.append(torch.cat(generated, dim=-1)[:, :n_steps])

    samples = torch.stack(all_samples)
    return {
        "median": samples.median(dim=0).values,
        "mean": samples.mean(dim=0),
        "q10": samples.quantile(0.1, dim=0),
        "q90": samples.quantile(0.9, dim=0),
    }


print("\n=== FORECASTING ===")
test_cases = make_test_cases()
for name, series in test_cases.items():
    result = forecast(model, series, n_steps=64, n_samples=30)
    spread = (result["q90"] - result["q10"]).mean().item()
    # Compare forecast to what the series "should" do
    print(
        f"{name:8s} | forecast first 4: {result['median'][0,:4].tolist()} | spread: {spread:.3f}"
    )
