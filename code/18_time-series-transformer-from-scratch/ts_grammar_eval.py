"""
Grammar evaluation for NanoTST.

The "grammar" of time series: flat → linear trend → sine wave.
Like a language model learning articles before grammar before idioms.

Watch the learning order during training:
  - Early: predictions are garbage for everything
  - Mid: flat is nailed first, line gets close, sine still bad
  - Late: sine improves, sigma reflects confidence (small for flat, larger for noisy)
"""

import torch
import math
import matplotlib.pyplot as plt
import os


def make_test_cases():
    """Canonical test patterns from simple to complex."""
    return {
        "flat": torch.ones(1, 512) * 3.0,
        "line": torch.linspace(0, 5, 512).unsqueeze(0),
        "sine": torch.sin(torch.linspace(0, 8 * math.pi, 512)).unsqueeze(0),
        "noisy": torch.sin(torch.linspace(0, 8 * math.pi, 512)).unsqueeze(0)
        + torch.randn(1, 512) * 0.2,
    }


def grammar_test(model, test_cases=None):
    """Print predictions vs reality for each pattern."""
    if test_cases is None:
        test_cases = make_test_cases()
    model.eval()
    for name, series in test_cases.items():
        mu, sigma = model(series)
        pred = mu[0, -1, :6].tolist()  # predicted next patch (first 6 values)
        conf = sigma[0, -1, :6].mean().item()  # average uncertainty
        print(
            f"  {name:8s} | pred: [{', '.join(f'{v:.2f}' for v in pred)}] "
            f"| sigma: {conf:.3f}"
        )
    model.train()


def plot_grammar(model, test_cases, epoch, loss, save_dir):
    """Save a 4-subplot figure showing target vs prediction for each pattern."""
    model.eval()
    patch_size = model.ps

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for idx, (name, series) in enumerate(test_cases.items()):
        ax = axes[idx]
        with torch.no_grad():
            mu, sigma = model(series)

        # Get normalized target (what the model actually sees)
        x_norm = (series - model.norm.mean) / model.norm.std
        patches = x_norm.view(1, -1, patch_size)
        n_patches = patches.shape[1]

        # Target: patches 1..N (what we want to predict)
        target = patches[0, 1:].reshape(-1).numpy()
        # Prediction: mu[0..N-1] (model's prediction for next patch)
        pred = mu[0, :-1].reshape(-1).numpy()
        # Uncertainty band
        sig = sigma[0, :-1].reshape(-1).numpy()

        # X-axis: starts at patch_size since we predict from patch 1 onward
        t = range(patch_size, patch_size + len(target))

        ax.plot(t, target, color="black", linewidth=1.5, label="target")
        ax.plot(t, pred, color="tab:blue", linewidth=1.5, label="predicted")
        ax.fill_between(
            t,
            pred - sig,
            pred + sig,
            alpha=0.2,
            color="tab:blue",
            label="±1σ",
        )
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("time step")
        ax.set_ylabel("normalized value")

    fig.suptitle(
        f"Epoch {epoch} | NLL Loss: {loss:.4f}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(save_dir, f"{epoch:03d}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")
    model.train()


def forecast(model, x, horizon_patches=4, n_samples=20):
    """
    Holdout forecast: chop last horizon_patches off the series as ground truth,
    autoregressively predict them from the context, compare to actual.
    """
    model.eval()
    P = model.ps
    holdout = horizon_patches * P
    context = x[:, :-holdout]
    all_samples = []

    with torch.no_grad():
        for _ in range(n_samples):
            current = context.clone()
            generated = []

            for _ in range(horizon_patches):
                mu, sigma = model(current)
                sample = torch.normal(mu[:, -1], sigma[:, -1])
                sample_real = model.norm.denormalize(sample)
                generated.append(sample_real)
                current = torch.cat([current, sample_real], dim=-1)

            all_samples.append(torch.cat(generated, dim=-1))

    samples = torch.stack(all_samples)
    model.train()
    return {
        "median": samples.median(dim=0).values,
        "mean": samples.mean(dim=0),
        "q10": samples.quantile(0.1, dim=0),
        "q90": samples.quantile(0.9, dim=0),
    }


def plot_forecast(model, save_dir, epoch=None, horizon_patches=4, n_samples=20):
    """Save 4-subplot figure: full actual series + forecast overlay on holdout."""
    test_cases = make_test_cases()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    P = model.ps
    holdout = horizon_patches * P

    for idx, (name, series) in enumerate(test_cases.items()):
        ax = axes[idx]
        result = forecast(model, series, horizon_patches, n_samples)

        total_len = series.shape[1]
        split = total_len - holdout

        # Full actual series in black
        actual = series[0].numpy()
        ax.plot(range(total_len), actual, color="black", linewidth=1.5, label="actual")

        # Forecast overlaid on the holdout region
        t_fc = range(split, total_len)
        median = result["median"][0].numpy()
        q10 = result["q10"][0].numpy()
        q90 = result["q90"][0].numpy()

        ax.plot(t_fc, median, color="tab:orange", linewidth=2, label="forecast")
        ax.fill_between(t_fc, q10, q90, alpha=0.25, color="tab:orange", label="q10–q90")

        # Dashed line at prediction boundary
        ax.axvline(split, color="gray", linestyle="--", alpha=0.7)

        mse = ((result["median"][0] - series[0, -holdout:]) ** 2).mean().item()
        ax.set_title(f"{name}  (MSE: {mse:.4f})", fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("time step")

    title = "Forecast vs Actual (holdout)"
    if epoch is not None:
        title = f"Epoch {epoch} | {title}"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fname = f"forecast_{epoch:03d}.png" if epoch else "forecast.png"
    path = os.path.join(save_dir, fname)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def get_save_dir(model):
    """Build save directory path from model hyperparams."""
    d_model = model.embed.proj.out_features
    n_heads = model.blocks[0].attn.n_heads
    n_layers = len(model.blocks)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder = f"d{d_model}_h{n_heads}_l{n_layers}_ps{model.ps}"
    save_dir = os.path.join(script_dir, "images", folder)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def train_with_grammar(model, data, epochs=100, batch_size=32, lr=3e-4, plot_every=1):
    """Train loop that runs grammar_test every plot_every epochs. Returns save_dir."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    test_cases = make_test_cases()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"NanoTST: {n_params:,} parameters\n")

    save_dir = get_save_dir(model)
    print(f"Saving plots to: {save_dir}\n")

    step = 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(data))
        total_loss, n = 0, 0

        for i in range(0, len(data) - batch_size, batch_size):
            loss = model.forward_and_loss(data[perm[i : i + batch_size]])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
            step += 1

        avg = total_loss / n
        if (epoch + 1) % plot_every == 0:
            print(f"Epoch {epoch + 1:3d} | Loss: {avg:.4f} | Step: {step}")
            grammar_test(model, test_cases)
            plot_grammar(model, test_cases, epoch + 1, avg, save_dir)
            plot_forecast(model, save_dir, epoch=epoch + 1)
            print()

    return save_dir
