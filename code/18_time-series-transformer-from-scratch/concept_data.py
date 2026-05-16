"""
Labeled synthetic time-series data for interpretability experiments.

The point of this module is to keep the data generating factors available
after sampling. If the model learns something useful, these labels let us ask
where trend, frequency, amplitude, noise, and jumps appear in the activations.
"""

from dataclasses import dataclass
import math

import torch


@dataclass
class ConceptBatch:
    series: torch.Tensor
    labels: dict[str, torch.Tensor]


def generate_labeled_data(n_series=1000, length=512, seed=77, jump_prob=0.25):
    """
    Generate sine + trend series and return the true latent factors.

    Labels:
      - freq: sine frequency
      - amplitude: sine amplitude
      - trend: linear slope coefficient
      - offset: vertical offset
      - noise_std: observation noise level
      - has_jump: whether a step jump was inserted
      - jump_pos: timestep where the jump starts (-1 if no jump)
      - jump_size: signed jump magnitude
    """
    generator = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 1, length).unsqueeze(0).expand(n_series, -1)

    freq = torch.randint(2, 20, (n_series, 1), generator=generator).float()
    amplitude = torch.rand(n_series, 1, generator=generator) * 2 + 0.5
    trend = torch.rand(n_series, 1, generator=generator) - 0.5
    offset = (torch.rand(n_series, 1, generator=generator) - 0.5) * 10
    noise_std = torch.rand(n_series, 1, generator=generator) * 0.18 + 0.02

    clean = offset + trend * t + amplitude * torch.sin(2 * math.pi * freq * t)

    has_jump = torch.rand(n_series, 1, generator=generator) < jump_prob
    jump_pos = torch.randint(length // 4, 3 * length // 4, (n_series, 1), generator=generator)
    jump_size = (torch.rand(n_series, 1, generator=generator) * 2 - 1) * amplitude
    jump_mask = torch.arange(length).unsqueeze(0) >= jump_pos
    jumps = has_jump.float() * jump_size * jump_mask.float()

    noise = torch.randn(n_series, length, generator=generator) * noise_std
    series = clean + jumps + noise

    labels = {
        "freq": freq.squeeze(-1),
        "amplitude": amplitude.squeeze(-1),
        "trend": trend.squeeze(-1),
        "offset": offset.squeeze(-1),
        "noise_std": noise_std.squeeze(-1),
        "has_jump": has_jump.float().squeeze(-1),
        "jump_pos": torch.where(has_jump.squeeze(-1), jump_pos.squeeze(-1), torch.full((n_series,), -1)),
        "jump_size": torch.where(has_jump.squeeze(-1), jump_size.squeeze(-1), torch.zeros(n_series)),
    }
    return ConceptBatch(series=series, labels=labels)


def make_concept_cases(length=512):
    """Canonical probes for attention, activation, and ablation plots."""
    t = torch.linspace(0, 1, length)
    return {
        "flat": torch.ones(1, length) * 3.0,
        "up_trend": torch.linspace(0, 5, length).unsqueeze(0),
        "down_trend": torch.linspace(5, 0, length).unsqueeze(0),
        "low_freq_sine": torch.sin(2 * math.pi * 3 * t).unsqueeze(0),
        "high_freq_sine": torch.sin(2 * math.pi * 12 * t).unsqueeze(0),
        "jump": (torch.sin(2 * math.pi * 5 * t) + (t > 0.55).float() * 2.0).unsqueeze(0),
        "noisy_sine": (
            torch.sin(2 * math.pi * 8 * t)
            + torch.randn(length, generator=torch.Generator().manual_seed(123)) * 0.25
        ).unsqueeze(0),
    }
