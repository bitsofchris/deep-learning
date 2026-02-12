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


data = generate_data()
print(f"Generated: {data.shape}")

one_series = data[0]
patches = one_series.view(-1, PATCH_SIZE)
