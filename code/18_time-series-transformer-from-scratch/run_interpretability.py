"""
Train NanoTST briefly and run first-pass interpretability checks.

Run:
    /Users/chris/repos/deep-learning/.venv/bin/python run_interpretability.py
"""

import argparse
import csv
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(_SCRIPT_DIR, "images", ".mplconfig"), exist_ok=True)
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(_SCRIPT_DIR, "images", ".mplconfig"),
)
os.environ.setdefault("MPLBACKEND", "Agg")

import torch

from concept_data import generate_labeled_data, make_concept_cases
from interpretability import (
    ablation_scores,
    capture_activations,
    pca_2d,
    probe_concepts,
    summarize_tokens,
)
from nano_tst import NanoTST
from ts_grammar_eval import grammar_test


def train(model, data, epochs=20, batch_size=32, lr=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        perm = torch.randperm(len(data))
        total_loss, n = 0.0, 0
        for i in range(0, len(data) - batch_size, batch_size):
            loss = model.forward_and_loss(data[perm[i : i + batch_size]])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        print(f"epoch {epoch + 1:03d} | loss {total_loss / max(n, 1):.4f}")


def save_probe_scores(results, out_dir):
    path = os.path.join(out_dir, "probe_scores.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "concept", "score"])
        writer.writeheader()
        for result in results:
            writer.writerow(
                {"layer": result.layer, "concept": result.concept, "score": result.score}
            )
    print(f"saved {path}")


def save_ablation_scores(rows, out_dir):
    path = os.path.join(out_dir, "ablation_scores.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["case", "module", "base_loss", "ablated_loss", "delta"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {path}")


def plot_probe_scores(results, out_dir):
    import matplotlib.pyplot as plt

    concepts = sorted({result.concept for result in results})
    layers = list(dict.fromkeys(result.layer for result in results))
    x = range(len(layers))

    fig, axes = plt.subplots(len(concepts), 1, figsize=(10, 2.4 * len(concepts)), sharex=True)
    if len(concepts) == 1:
        axes = [axes]

    for ax, concept in zip(axes, concepts):
        scores = [
            next(result.score for result in results if result.layer == layer and result.concept == concept)
            for layer in layers
        ]
        ax.plot(x, scores, marker="o")
        ax.set_ylabel(concept)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.3)
        if concept == "has_jump":
            ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[-1].set_xticks(list(x))
    axes[-1].set_xticklabels(layers, rotation=35, ha="right")
    fig.suptitle("Linear probe score by layer")
    fig.tight_layout()
    path = os.path.join(out_dir, "probe_scores.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"saved {path}")


def plot_pca(model, concept_batch, out_dir):
    import matplotlib.pyplot as plt

    cache, _attn, _output = capture_activations(model, concept_batch.series)
    layers = ["embed", "final_norm"]
    color_values = concept_batch.labels["freq"]

    fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 5))
    if len(layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        x = summarize_tokens(cache[layer])
        xy = pca_2d(x)
        scatter = ax.scatter(xy[:, 0], xy[:, 1], c=color_values, cmap="viridis", s=16)
        ax.set_title(layer)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=axes, label="frequency")
    fig.suptitle("Activation PCA colored by true frequency")
    fig.tight_layout()
    path = os.path.join(out_dir, "activation_pca_frequency.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"saved {path}")


def plot_attention_cases(model, out_dir):
    import matplotlib.pyplot as plt

    cases = make_concept_cases()
    _cache, attn, _output = capture_activations(model, torch.cat(list(cases.values()), dim=0))
    case_names = list(cases)
    last_layer = sorted(attn)[-1]
    weights = attn[last_layer]
    head = 0

    fig, axes = plt.subplots(1, len(case_names), figsize=(3 * len(case_names), 3))
    for idx, (ax, name) in enumerate(zip(axes, case_names)):
        ax.imshow(weights[idx, head], cmap="magma", vmin=0, vmax=1)
        ax.set_title(name)
        ax.set_xlabel("source patch")
        ax.set_ylabel("dest patch")
    fig.suptitle(f"{last_layer}, head {head} attention")
    fig.tight_layout()
    path = os.path.join(out_dir, "attention_cases.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-series", type=int, default=1000)
    parser.add_argument("--eval-series", type=int, default=300)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--plots", action="store_true", help="also render PNG plots with matplotlib")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "interpretability")
    os.makedirs(out_dir, exist_ok=True)

    train_batch = generate_labeled_data(n_series=args.train_series, seed=77)
    eval_batch = generate_labeled_data(n_series=args.eval_series, seed=78)

    model = NanoTST(d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers)
    train(model, train_batch.series, epochs=args.epochs)

    print("\nGrammar checks:")
    grammar_test(model)

    print("\nProbing concepts:")
    probe_results = probe_concepts(model, eval_batch.series, eval_batch.labels)
    for result in probe_results:
        print(f"{result.layer:16s} | {result.concept:10s} | {result.score: .3f}")

    ablations = ablation_scores(model, make_concept_cases())
    save_probe_scores(probe_results, out_dir)
    save_ablation_scores(ablations, out_dir)
    if args.plots:
        plot_probe_scores(probe_results, out_dir)
        plot_pca(model, eval_batch, out_dir)
        plot_attention_cases(model, out_dir)
    else:
        print("skipped PNG plots; pass --plots to render them")


if __name__ == "__main__":
    main()
