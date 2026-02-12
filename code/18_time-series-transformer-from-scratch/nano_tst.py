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
print("\n === LINEAR BASELINE ===")
data = generate_data(2000)
baseline = LinearBaseline()
train_model(baseline, data, epochs=1000)
# Around 500 epochs, saturates at MSE 0.99 - 1.00
