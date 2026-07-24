"""
Data loading for NanoTST — the module behind the dataloading videos.

Three ways to deliver the same synthetic series to the model:

1. SeriesDataset        — map-style: cheap random access over a precomputed tensor
2. SyntheticSeriesStream — iterable-style: an infinite stream, generated on demand
3. SeriesStreamNode      — torchdata.nodes BaseNode: the same infinite stream with
                           explicit state, so even an infinite stream can be
                           checkpointed and resumed mid-epoch

Classes live in a module (not the notebook) because macOS spawns DataLoader
workers by pickling — notebook-defined classes can't cross that boundary.
"""

import math

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

PATCH_SIZE = 32
CONTEXT_LEN = 512


def make_series(n_series: int, length: int, generator: torch.Generator) -> torch.Tensor:
    """Sine + trend + noise series, same recipe as nano_tst.generate_data."""
    t = torch.linspace(0, 1, length).unsqueeze(0).expand(n_series, -1)
    freq = torch.randint(2, 20, (n_series, 1), generator=generator).float()
    amplitude = torch.rand(n_series, 1, generator=generator) * 2 + 0.5
    trend = torch.rand(n_series, 1, generator=generator) - 0.5
    offset = (torch.rand(n_series, 1, generator=generator) - 0.5) * 10
    series = offset + trend * t + amplitude * torch.sin(2 * math.pi * freq * t)
    return series + torch.randn(n_series, length, generator=generator) * 0.1


# ---------------------------------------------------------------------------
# 1. Map-style: the random-access contract (__len__ + __getitem__)
# ---------------------------------------------------------------------------


class SeriesDataset(Dataset):
    """All series precomputed in memory; indexing is free, so the sampler owns order."""

    def __init__(self, n_series: int = 1000, length: int = CONTEXT_LEN, seed: int = 77):
        generator = torch.Generator().manual_seed(seed)
        self.series = make_series(n_series, length, generator)

    def __len__(self) -> int:
        return len(self.series)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.series[idx]


# ---------------------------------------------------------------------------
# 2. Iterable-style: the streaming contract (__iter__ only)
# ---------------------------------------------------------------------------


def series_at(seed: int, index: int, length: int = CONTEXT_LEN) -> torch.Tensor:
    """Deterministically generate series number `index` of stream `seed`."""
    generator = torch.Generator().manual_seed(seed * 1_000_003 + index)
    return make_series(1, length, generator)[0]


class SyntheticSeriesStream(IterableDataset):
    """An infinite stream: series are generated on demand, one at a time.

    There is no __len__ and no __getitem__ — sample i doesn't exist until
    the iterator produces it. With shard_per_worker=False every DataLoader
    worker runs this same iterator and delivers identical samples (the
    duplication demo). With shard_per_worker=True each worker takes an
    interleaved slice of the stream via get_worker_info().
    """

    def __init__(
        self, seed: int = 77, length: int = CONTEXT_LEN, shard_per_worker: bool = False
    ):
        self.seed = seed
        self.length = length
        self.shard_per_worker = shard_per_worker

    def __iter__(self):
        start, step = 0, 1
        info = get_worker_info()
        if self.shard_per_worker and info is not None:
            start, step = info.id, info.num_workers
        index = start
        while True:
            yield series_at(self.seed, index, self.length)
            index += step


class EpochSeededRandomSampler:
    """Shuffling whose randomness is reconstructible: perm = f(base_seed, epoch).

    A plain RandomSampler draws a fresh permutation every time it's iterated,
    so a resumed pipeline fast-forwards through the *wrong* order. Seeding by
    epoch makes re-iteration replay the same permutation — which is what makes
    exact mid-epoch resume possible (torchdata's SamplerWrapper calls
    set_epoch and fast-forwards on restore).
    """

    def __init__(self, data_source, base_seed: int = 0):
        self.data_source = data_source
        self.base_seed = base_seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.base_seed + self.epoch)
        yield from torch.randperm(len(self.data_source), generator=generator).tolist()

    def __len__(self) -> int:
        return len(self.data_source)


# ---------------------------------------------------------------------------
# 3. torchdata.nodes: the same stream as a BaseNode with explicit state
# ---------------------------------------------------------------------------

try:
    from torchdata.nodes import BaseNode

    class SeriesStreamNode(BaseNode[torch.Tensor]):
        """The infinite stream, rebuilt as a node.

        No generators, no hidden position: the entire state is one integer.
        That's what makes an *infinite* stream checkpointable — get_state()
        returns {"index": i}, reset(state) puts you back on sample i exactly.
        """

        def __init__(self, seed: int = 77, length: int = CONTEXT_LEN):
            super().__init__()
            self.seed = seed
            self.length = length
            self.index = 0

        def reset(self, initial_state: dict | None = None):
            super().reset(initial_state)
            self.index = initial_state["index"] if initial_state else 0

        def next(self) -> torch.Tensor:
            sample = series_at(self.seed, self.index, self.length)
            self.index += 1
            return sample

        def get_state(self) -> dict:
            return {"index": self.index}

except ImportError:  # torchdata not installed; map/iterable classes still usable
    pass


# ---------------------------------------------------------------------------
# Training on a loader (instead of hand-slicing a tensor)
# ---------------------------------------------------------------------------


def train_with_loader(
    model, loader, epochs: int = 3, lr: float = 3e-4, steps_per_epoch: int | None = None
) -> list[float]:
    """Same loop as nano_tst.train_model, but the loader owns delivery.

    steps_per_epoch caps iteration for infinite loaders.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch_losses = []
    for epoch in range(epochs):
        total_loss, n = 0.0, 0
        for step, batch in enumerate(loader):
            if steps_per_epoch is not None and step >= steps_per_epoch:
                break
            loss = model.forward_and_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
        epoch_losses.append(total_loss / max(n, 1))
        print(f"  Epoch {epoch + 1} | loss: {epoch_losses[-1]:.4f} ({n} steps)")
    return epoch_losses
