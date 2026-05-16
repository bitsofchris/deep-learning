"""
Small interpretability helpers for NanoTST.

This deliberately starts with simple, falsifiable tools:
  - activation capture
  - linear probes for known synthetic concepts
  - ablation tests for layer importance
  - PCA via torch SVD for visualization inputs
"""

from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass
class ProbeResult:
    layer: str
    concept: str
    score: float


class ActivationRecorder:
    """Forward-hook based activation cache for the NanoTST module layout."""

    def __init__(self, model):
        self.model = model
        self.cache = {}
        self.handles = []

    def _save(self, name):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                output = tuple(item.detach().cpu() for item in output)
            else:
                output = output.detach().cpu()
            self.cache[name] = output

        return hook

    def __enter__(self):
        self.handles.append(self.model.embed.register_forward_hook(self._save("embed")))
        for idx, block in enumerate(self.model.blocks):
            self.handles.append(block.norm1.register_forward_hook(self._save(f"blocks.{idx}.norm1")))
            self.handles.append(block.attn.register_forward_hook(self._save(f"blocks.{idx}.attn_out")))
            self.handles.append(block.norm2.register_forward_hook(self._save(f"blocks.{idx}.norm2")))
            self.handles.append(block.ffn.register_forward_hook(self._save(f"blocks.{idx}.ffn_out")))
            self.handles.append(block.register_forward_hook(self._save(f"blocks.{idx}.resid")))
        self.handles.append(self.model.final_norm.register_forward_hook(self._save("final_norm")))
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def attention_cache(self):
        """Return attention matrices saved by each block's attention module."""
        attn = {}
        for idx, block in enumerate(self.model.blocks):
            if hasattr(block.attn, "last_attn"):
                attn[f"blocks.{idx}.attn"] = block.attn.last_attn.detach().cpu()
        return attn


def capture_activations(model, series):
    """Run a forward pass and return activations, attentions, and prediction."""
    was_training = model.training
    model.eval()
    with torch.no_grad(), ActivationRecorder(model) as recorder:
        output = model(series)
        cache = dict(recorder.cache)
        attn = recorder.attention_cache()
    if was_training:
        model.train()
    return cache, attn, output


def summarize_tokens(activation, mode="last"):
    """
    Convert (batch, tokens, features) activations to one vector per series.

    Last-token summaries answer: what information is available where the model
    predicts the next patch? Mean summaries answer: what is represented globally?
    """
    if activation.ndim != 3:
        raise ValueError(f"expected rank-3 activation, got {activation.shape}")
    if mode == "last":
        return activation[:, -1, :]
    if mode == "mean":
        return activation.mean(dim=1)
    raise ValueError(f"unknown summary mode: {mode}")


def _standardize_with_train_stats(x_train, x_test, eps=1e-6):
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True) + eps
    return (x_train - mean) / std, (x_test - mean) / std


def fit_linear_probe(x_train, y_train, x_test, y_test, kind="regression", ridge=1e-3):
    """
    Closed-form linear probe.

    Regression score is R^2. Binary classification score is accuracy with a
    logistic-free least-squares classifier, which is enough for a first pass.
    """
    x_train, x_test = _standardize_with_train_stats(x_train.float(), x_test.float())
    ones_train = torch.ones(x_train.shape[0], 1)
    ones_test = torch.ones(x_test.shape[0], 1)
    x_train = torch.cat([x_train, ones_train], dim=1)
    x_test = torch.cat([x_test, ones_test], dim=1)

    y_train = y_train.float().reshape(-1, 1)
    y_test = y_test.float().reshape(-1, 1)
    eye = torch.eye(x_train.shape[1])
    weights = torch.linalg.solve(x_train.T @ x_train + ridge * eye, x_train.T @ y_train)
    pred = x_test @ weights

    if kind == "binary":
        return ((pred > 0.5).float() == y_test).float().mean().item()

    ss_res = ((y_test - pred) ** 2).sum()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum().clamp_min(1e-8)
    return (1 - ss_res / ss_tot).item()


def probe_concepts(model, series, labels, layer_names=None, train_frac=0.8, summary="last"):
    """Score how linearly decodable each concept is from each chosen layer."""
    cache, _attn, _output = capture_activations(model, series)
    if layer_names is None:
        layer_names = ["embed"] + [name for name in cache if name.endswith(".resid")] + ["final_norm"]

    n = series.shape[0]
    split = max(1, int(n * train_frac))
    results = []
    concepts = {
        "trend": "regression",
        "freq": "regression",
        "amplitude": "regression",
        "noise_std": "regression",
        "has_jump": "binary",
    }

    for layer in layer_names:
        x = summarize_tokens(cache[layer], mode=summary)
        for concept, kind in concepts.items():
            score = fit_linear_probe(
                x[:split],
                labels[concept][:split],
                x[split:],
                labels[concept][split:],
                kind=kind,
            )
            results.append(ProbeResult(layer=layer, concept=concept, score=score))
    return results


@contextmanager
def ablate_module(module):
    """Temporarily replace a module's output with zeros."""

    def hook(_module, _inputs, output):
        return torch.zeros_like(output)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def loss_on_series(model, series):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        loss = model.forward_and_loss(series).item()
    if was_training:
        model.train()
    return loss


def ablation_scores(model, cases):
    """
    Measure loss increase when ablating each block's attention or FFN output.

    Positive delta means the component helped that case. Near-zero delta means
    the component was not important for that behavior under this metric.
    """
    rows = []
    modules = []
    for idx, block in enumerate(model.blocks):
        modules.append((f"blocks.{idx}.attn", block.attn))
        modules.append((f"blocks.{idx}.ffn", block.ffn))

    for case_name, series in cases.items():
        base = loss_on_series(model, series)
        for module_name, module in modules:
            with ablate_module(module):
                ablated = loss_on_series(model, series)
            rows.append(
                {
                    "case": case_name,
                    "module": module_name,
                    "base_loss": base,
                    "ablated_loss": ablated,
                    "delta": ablated - base,
                }
            )
    return rows


def pca_2d(x):
    """Return a dependency-free 2D PCA projection."""
    x, _ = _standardize_with_train_stats(x.float(), x.float())
    _u, _s, vh = torch.linalg.svd(x, full_matrices=False)
    return x @ vh[:2].T
