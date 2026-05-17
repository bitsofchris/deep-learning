"""
Train NanoTST briefly and run first-pass interpretability checks.

Run:
    /Users/chris/repos/deep-learning/.venv/bin/python run_interpretability.py
"""

import argparse
import csv
from datetime import datetime
import html
import json
import os
import time

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


def iter_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        if len(batch) > 0:
            yield batch


def eval_loss(model, data, batch_size=64):
    was_training = model.training
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in iter_batches(data, batch_size):
            total_loss += model.forward_and_loss(batch).item()
            n += 1
    if was_training:
        model.train()
    return total_loss / max(n, 1)


def train(model, train_data, val_data, out_dir, epochs=20, batch_size=32, lr=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    best_val = float("inf")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    started = time.perf_counter()

    for epoch in range(epochs):
        epoch_started = time.perf_counter()
        perm = torch.randperm(len(train_data))
        total_loss, n = 0.0, 0
        for i in range(0, len(train_data), batch_size):
            idx = perm[i : i + batch_size]
            if len(idx) == 0:
                continue
            loss = model.forward_and_loss(train_data[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1

        train_loss = total_loss / max(n, 1)
        val_loss = eval_loss(model, val_data, batch_size=batch_size)
        epoch_seconds = time.perf_counter() - epoch_started
        elapsed_seconds = time.perf_counter() - started
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_seconds": epoch_seconds,
            "elapsed_seconds": elapsed_seconds,
        }
        history.append(row)
        print(
            f"epoch {epoch + 1:03d} | train {train_loss:.4f} | "
            f"val {val_loss:.4f} | {epoch_seconds:.1f}s"
        )

        latest_path = os.path.join(ckpt_dir, "latest.pt")
        torch.save(
            {"epoch": epoch + 1, "model_state": model.state_dict(), "history": history},
            latest_path,
        )
        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "history": history,
                },
                best_path,
            )

    return history


def save_history(history, out_dir):
    path = os.path.join(out_dir, "history.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "epoch_seconds",
                "elapsed_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(history)
    print(f"saved {path}")


def save_probe_scores(results, out_dir):
    path = os.path.join(out_dir, "probe_scores.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "concept", "score"])
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "layer": result.layer,
                    "concept": result.concept,
                    "score": result.score,
                }
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


def collect_prediction_rows(model):
    rows = []
    was_training = model.training
    model.eval()
    cases = make_concept_cases()

    with torch.no_grad():
        for case_name, series in cases.items():
            mu, sigma = model(series)
            x_norm = (series - model.norm.mean) / model.norm.std
            patches = x_norm.view(1, -1, model.ps)
            target = patches[0, 1:].reshape(-1)
            pred = mu[0, :-1].reshape(-1)
            sig = sigma[0, :-1].reshape(-1)
            for t, (actual, predicted, uncertainty) in enumerate(
                zip(target, pred, sig)
            ):
                rows.append(
                    {
                        "case": case_name,
                        "t": t + model.ps,
                        "actual_norm": actual.item(),
                        "pred_norm": predicted.item(),
                        "sigma": uncertainty.item(),
                    }
                )

    if was_training:
        model.train()
    return rows


def save_prediction_rows(model, out_dir):
    rows = collect_prediction_rows(model)
    path = os.path.join(out_dir, "predictions.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["case", "t", "actual_norm", "pred_norm", "sigma"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {path}")
    return rows


def _svg_line_chart(series_by_name, width=760, height=260):
    if not series_by_name:
        return "<p>No data.</p>"
    all_points = [point for series in series_by_name.values() for point in series]
    ys = [point[1] for point in all_points]
    xs = [point[0] for point in all_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_max = x_min + 1
    if y_min == y_max:
        y_max = y_min + 1
    pad = 36

    def scale_x(x):
        return pad + (x - x_min) / (x_max - x_min) * (width - 2 * pad)

    def scale_y(y):
        return height - pad - (y - y_min) / (y_max - y_min) * (height - 2 * pad)

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    paths = []
    legend = []
    for idx, (name, series) in enumerate(series_by_name.items()):
        color = colors[idx % len(colors)]
        points = " ".join(f"{scale_x(x):.1f},{scale_y(y):.1f}" for x, y in series)
        paths.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        legend.append(
            f'<span><i style="background:{color}"></i>{html.escape(name)}</span>'
        )

    return f"""
    <svg viewBox="0 0 {width} {height}" role="img">
      <rect x="0" y="0" width="{width}" height="{height}" fill="white"/>
      <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#aaa"/>
      <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#aaa"/>
      {''.join(paths)}
      <text x="{pad}" y="22" font-size="12" fill="#555">min {y_min:.3f} / max {y_max:.3f}</text>
    </svg>
    <div class="legend">{''.join(legend)}</div>
    """


def write_dashboard(history, prediction_rows, probe_results, ablations, args, out_dir):
    history_chart = _svg_line_chart(
        {
            "train_loss": [(row["epoch"], row["train_loss"]) for row in history],
            "val_loss": [(row["epoch"], row["val_loss"]) for row in history],
        }
    )
    total_seconds = history[-1]["elapsed_seconds"] if history else 0.0
    seconds_per_epoch = total_seconds / max(len(history), 1)

    pred_sections = []
    for case in sorted({row["case"] for row in prediction_rows}):
        rows = [row for row in prediction_rows if row["case"] == case]
        rows = rows[: min(len(rows), 160)]
        pred_sections.append(
            f"<h3>{html.escape(case)}</h3>"
            + _svg_line_chart(
                {
                    "actual_norm": [(row["t"], row["actual_norm"]) for row in rows],
                    "pred_norm": [(row["t"], row["pred_norm"]) for row in rows],
                },
                height=220,
            )
        )

    top_ablations = sorted(ablations, key=lambda row: row["delta"], reverse=True)[:12]
    ablation_rows = "\n".join(
        f"<tr><td>{html.escape(row['case'])}</td><td>{html.escape(row['module'])}</td>"
        f"<td>{row['delta']:.4f}</td></tr>"
        for row in top_ablations
    )
    probe_rows = "\n".join(
        f"<tr><td>{html.escape(result.layer)}</td><td>{html.escape(result.concept)}</td>"
        f"<td>{result.score:.3f}</td></tr>"
        for result in probe_results
    )

    payload = {
        "args": vars(args),
        "total_seconds": total_seconds,
        "seconds_per_epoch": seconds_per_epoch,
        "best_val_loss": min((row["val_loss"] for row in history), default=None),
    }
    with open(os.path.join(out_dir, "run_summary.json"), "w") as f:
        json.dump(payload, f, indent=2)

    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>NanoTST Training Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 32px; color: #111827; }}
    h1, h2, h3 {{ margin-bottom: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 12px; max-width: 900px; }}
    .metric {{ border: 1px solid #ddd; border-radius: 6px; padding: 12px; }}
    .metric strong {{ display: block; font-size: 22px; }}
    section {{ margin: 28px 0; max-width: 960px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; }}
    svg {{ width: 100%; max-width: 900px; border: 1px solid #e5e7eb; border-radius: 6px; }}
    .legend span {{ margin-right: 16px; font-size: 13px; }}
    .legend i {{ display: inline-block; width: 12px; height: 12px; margin-right: 5px; vertical-align: -1px; }}
    code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>NanoTST Training Report</h1>
  <div class="grid">
    <div class="metric">epochs<strong>{len(history)}</strong></div>
    <div class="metric">best val loss<strong>{payload['best_val_loss']:.4f}</strong></div>
    <div class="metric">total time<strong>{total_seconds:.1f}s</strong></div>
    <div class="metric">sec / epoch<strong>{seconds_per_epoch:.1f}s</strong></div>
  </div>
  <section>
    <h2>Loss Curves</h2>
    {history_chart}
    <p>Raw values: <code>history.csv</code>. Weights: <code>checkpoints/latest.pt</code> and <code>checkpoints/best.pt</code>.</p>
  </section>
  <section>
    <h2>Prediction Snapshots</h2>
    {''.join(pred_sections)}
    <p>Raw values: <code>predictions.csv</code>. These are normalized values because that is the space the model trains in.</p>
  </section>
  <section>
    <h2>Probe Scores</h2>
    <table><tr><th>Layer</th><th>Concept</th><th>Score</th></tr>{probe_rows}</table>
  </section>
  <section>
    <h2>Largest Ablation Deltas</h2>
    <table><tr><th>Case</th><th>Module</th><th>Loss Delta</th></tr>{ablation_rows}</table>
  </section>
</body>
</html>
"""
    path = os.path.join(out_dir, "report.html")
    with open(path, "w") as f:
        f.write(doc)
    print(f"saved {path}")


def plot_probe_scores(results, out_dir):
    import matplotlib.pyplot as plt

    concepts = sorted({result.concept for result in results})
    layers = list(dict.fromkeys(result.layer for result in results))
    x = range(len(layers))

    fig, axes = plt.subplots(
        len(concepts), 1, figsize=(10, 2.4 * len(concepts)), sharex=True
    )
    if len(concepts) == 1:
        axes = [axes]

    for ax, concept in zip(axes, concepts):
        scores = [
            next(
                result.score
                for result in results
                if result.layer == layer and result.concept == concept
            )
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
    _cache, attn, _output = capture_activations(
        model, torch.cat(list(cases.values()), dim=0)
    )
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--plots", action="store_true", help="also render PNG plots with matplotlib"
    )
    args = parser.parse_args()

    out_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "images", "interpretability"
    )
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, "runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "latest_run.txt"), "w") as f:
        f.write(out_dir + "\n")
    print(f"writing run artifacts to {out_dir}")

    train_batch = generate_labeled_data(n_series=args.train_series, seed=77)
    eval_batch = generate_labeled_data(n_series=args.eval_series, seed=78)

    model = NanoTST(d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers)
    history = train(
        model,
        train_batch.series,
        eval_batch.series,
        out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    save_history(history, out_dir)

    print("\nGrammar checks:")
    grammar_test(model)

    print("\nProbing concepts:")
    probe_results = probe_concepts(model, eval_batch.series, eval_batch.labels)
    for result in probe_results:
        print(f"{result.layer:16s} | {result.concept:10s} | {result.score: .3f}")

    ablations = ablation_scores(model, make_concept_cases())
    prediction_rows = save_prediction_rows(model, out_dir)
    save_probe_scores(probe_results, out_dir)
    save_ablation_scores(ablations, out_dir)
    write_dashboard(history, prediction_rows, probe_results, ablations, args, out_dir)
    if args.plots:
        plot_probe_scores(probe_results, out_dir)
        plot_pca(model, eval_batch, out_dir)
        plot_attention_cases(model, out_dir)
    else:
        print("skipped PNG plots; pass --plots to render them")


if __name__ == "__main__":
    main()
