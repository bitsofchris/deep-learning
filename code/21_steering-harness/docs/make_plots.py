"""Generate all blog-post plots from the harvested activations and run log.

Run from project root:
    python3 docs/make_plots.py

Writes PNGs to docs/plots/ and a README.md indexing them.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VEC_DIR = ROOT / "results" / "vectors"
RUNS_JSONL = ROOT / "results" / "runs.jsonl"
OUT = HERE / "plots"
OUT.mkdir(parents=True, exist_ok=True)

CONCEPTS = ["golden_gate", "golden_gate_v2", "pickles", "pirate"]
LAYERS = [6, 9, 12, 15, 18, 21, 24]

CONCEPT_COLORS = {
    "golden_gate":    "#1f77b4",  # blue
    "golden_gate_v2": "#17becf",  # cyan
    "pickles":        "#2ca02c",  # green
    "pirate":         "#d62728",  # red
}

KEYWORDS = {
    "pickles": re.compile(r"pickle|pickled|pickles|gherkin|kraut", re.I),
    "golden_gate": re.compile(r"golden gate", re.I),
    "golden_gate_v2": re.compile(r"golden gate", re.I),
    "pirate": re.compile(r"\barrr+|ahoy|matey|shiver|ye\b|scurvy|plunder|doubloon|yer\b|larf|swab", re.I),
}


def load_layer(concept: str, layer: int) -> dict:
    p = VEC_DIR / concept / f"layer_{layer:02d}.npz"
    return dict(np.load(p))


def load_runs() -> list[dict]:
    rows = []
    with open(RUNS_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ---------------- 1. Projection histograms ----------------
def plot_projection_hists():
    for concept in CONCEPTS:
        for layer in LAYERS:
            d = load_layer(concept, layer)
            unit = d["unit"]
            pos_proj = d["positive_acts"] @ unit
            neg_proj = d["negative_acts"] @ unit
            fig, ax = plt.subplots(figsize=(6, 4))
            bins = 20
            ax.hist(neg_proj, bins=bins, alpha=0.6, label="negative", color="C1")
            ax.hist(pos_proj, bins=bins, alpha=0.6, label="positive", color="C0")
            cos_pn = float(d["cos_pos_neg"])
            ax.set_title(f"{concept} — layer {layer} — cos(p,n)={cos_pn:.3f}")
            ax.set_xlabel("projection onto unit steering vector")
            ax.set_ylabel("count")
            ax.legend()
            fig.tight_layout()
            fig.savefig(OUT / f"proj_{concept}_layer{layer:02d}.png", dpi=130)
            plt.close(fig)


# ---------------- 2. 2D PCA per layer ----------------
def plot_pca():
    for concept in CONCEPTS:
        for layer in LAYERS:
            d = load_layer(concept, layer)
            X = np.vstack([d["positive_acts"], d["negative_acts"]])
            pca = PCA(n_components=2)
            Z = pca.fit_transform(X)
            n_pos = d["positive_acts"].shape[0]
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(Z[:n_pos, 0], Z[:n_pos, 1], c="C0", label="positive", s=40, alpha=0.8)
            ax.scatter(Z[n_pos:, 0], Z[n_pos:, 1], c="C1", label="negative", s=40, alpha=0.8)
            ev = pca.explained_variance_ratio_
            ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
            ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
            ax.set_title(f"{concept} — layer {layer} — PCA of last-token residual")
            ax.legend()
            fig.tight_layout()
            fig.savefig(OUT / f"pca_{concept}_layer{layer:02d}.png", dpi=130)
            plt.close(fig)


# ---------------- 3. ||diff|| across layers ----------------
def plot_diff_norm():
    for concept in CONCEPTS:
        norms = []
        for layer in LAYERS:
            d = load_layer(concept, layer)
            norms.append(float(np.linalg.norm(d["diff"])))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(LAYERS, norms, marker="o")
        ax.set_xlabel("layer")
        ax.set_ylabel("||diff||")
        ax.set_title(f"{concept} — norm of mean-diff vector across layers")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"diff_norm_{concept}.png", dpi=130)
        plt.close(fig)


# ---------------- 4. cos(pos, neg) overlaid ----------------
def plot_cos_pn():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for concept in CONCEPTS:
        vals = []
        for layer in LAYERS:
            d = load_layer(concept, layer)
            vals.append(float(d["cos_pos_neg"]))
        ax.plot(LAYERS, vals, marker="o", label=concept)
    ax.set_xlabel("layer")
    ax.set_ylabel("cos(positive_mean, negative_mean)")
    ax.set_title("Contrastive signal strength — closer to 1.0 = weaker")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "cos_pn_vs_layer.png", dpi=130)
    plt.close(fig)


# ---------------- 5. Unit-vector rotation heatmap ----------------
def plot_unit_rotation():
    for concept in CONCEPTS:
        units = np.stack([load_layer(concept, L)["unit"] for L in LAYERS])
        M = units @ units.T
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(LAYERS)))
        ax.set_yticks(range(len(LAYERS)))
        ax.set_xticklabels(LAYERS)
        ax.set_yticklabels(LAYERS)
        ax.set_xlabel("layer")
        ax.set_ylabel("layer")
        for i in range(len(LAYERS)):
            for j in range(len(LAYERS)):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if abs(M[i,j]) < 0.6 else "white")
        ax.set_title(f"{concept} — cos(unit_i, unit_j) across layers")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(OUT / f"unit_rotation_{concept}.png", dpi=130)
        plt.close(fig)


# ---------------- 6. Emergence grid heatmap ----------------
def plot_emergence(runs: list[dict]):
    alphas = [2, 4, 6, 8, 10]
    for concept in CONCEPTS:
        pat = KEYWORDS[concept]
        grid = np.full((len(LAYERS), len(alphas)), np.nan)
        for r in runs:
            if r.get("concept") != concept:
                continue
            if not r.get("extra", {}).get("grid"):
                continue
            L, a = r["layer"], r["alpha"]
            if L in LAYERS and a in alphas:
                i, j = LAYERS.index(L), alphas.index(a)
                grid[i, j] = len(pat.findall(r.get("output", "")))
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(grid, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels(alphas)
        ax.set_yticks(range(len(LAYERS)))
        ax.set_yticklabels(LAYERS)
        ax.set_xlabel("alpha")
        ax.set_ylabel("layer")
        for i in range(len(LAYERS)):
            for j in range(len(alphas)):
                v = grid[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{int(v)}", ha="center", va="center",
                            color="white" if v < np.nanmax(grid) * 0.5 else "black",
                            fontsize=9)
        ax.set_title(f"{concept} — keyword hits per (layer × α)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(OUT / f"emergence_{concept}.png", dpi=130)
        plt.close(fig)


# ---------------- 7. Residual norm across layers ----------------
def plot_resid_norm():
    fig, ax = plt.subplots(figsize=(6, 4))
    for concept in CONCEPTS:
        vals = [float(load_layer(concept, L)["typical_norm"]) for L in LAYERS]
        ax.plot(LAYERS, vals, marker="o", label=concept)
    ax.set_xlabel("layer")
    ax.set_ylabel("typical ||resid||")
    ax.set_title("Residual stream norm across layers (property of the model)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "resid_norm_vs_layer.png", dpi=130)
    plt.close(fig)


# ---------------- 8. ALL concepts together with mean-diff arrows ----------------
def _scatter_all_concepts(ax, coords_by_concept, title):
    from matplotlib.lines import Line2D
    for concept, (pos2d, neg2d) in coords_by_concept.items():
        c = CONCEPT_COLORS[concept]
        ax.scatter(pos2d[:, 0], pos2d[:, 1], c=c, s=22, marker="o",
                   alpha=0.8, edgecolors="white", linewidths=0.3)
        ax.scatter(neg2d[:, 0], neg2d[:, 1], c=c, s=22, marker="x", alpha=0.6)
        pm, nm = pos2d.mean(0), neg2d.mean(0)
        ax.annotate("", xy=pm, xytext=nm,
                    arrowprops=dict(arrowstyle="->", color=c, lw=2.4, alpha=0.95,
                                    shrinkA=0, shrinkB=0))
        ax.plot(*pm, marker="*", color=c, markersize=16,
                markeredgecolor="black", markeredgewidth=0.7)
        ax.plot(*nm, marker="s", color=c, markersize=9,
                markeredgecolor="black", markeredgewidth=0.5)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])


def _reduce(stacked: np.ndarray, method: str):
    if method == "pca":
        return PCA(n_components=2).fit_transform(stacked)
    elif method == "umap":
        import umap
        return umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.2,
                         metric="cosine", random_state=0).fit_transform(stacked)
    raise ValueError(method)


def _build_coords_all_concepts(layer: int, method: str):
    parts = []
    sizes = []
    for concept in CONCEPTS:
        d = load_layer(concept, layer)
        parts.append(d["positive_acts"])
        parts.append(d["negative_acts"])
        sizes.append((d["positive_acts"].shape[0], d["negative_acts"].shape[0]))
    stacked = np.vstack(parts)
    coords = _reduce(stacked, method)
    out = {}
    idx = 0
    for concept, (n_pos, n_neg) in zip(CONCEPTS, sizes):
        out[concept] = (coords[idx:idx + n_pos], coords[idx + n_pos:idx + n_pos + n_neg])
        idx += n_pos + n_neg
    return out


def _legend_handles():
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=CONCEPT_COLORS[c], markersize=10, label=c)
               for c in CONCEPTS]
    handles += [
        Line2D([0], [0], marker="*", color="black", linestyle="",
               markersize=13, markerfacecolor="grey", label="positive centroid"),
        Line2D([0], [0], marker="s", color="black", linestyle="",
               markersize=9, markerfacecolor="grey", label="negative centroid"),
        Line2D([0], [0], marker="o", color="grey", linestyle="", markersize=7, label="+ sentence"),
        Line2D([0], [0], marker="x", color="grey", linestyle="", markersize=7, label="− sentence"),
    ]
    return handles


def plot_all_concepts_at_layer(layer: int, method: str):
    coords = _build_coords_all_concepts(layer, method)
    fig, ax = plt.subplots(figsize=(9, 8))
    _scatter_all_concepts(
        ax, coords,
        f"{method.upper()} of 240 sentence activations at layer {layer}\n"
        f"arrows = mean-diff steering vectors (negative centroid → positive centroid)"
    )
    ax.legend(handles=_legend_handles(), loc="best", fontsize=9, framealpha=0.92)
    fig.tight_layout()
    path = OUT / f"{method}_all_concepts_layer{layer:02d}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path.name}")


def plot_emergence_vs_depth(method: str):
    layers_grid = [6, 12, 18, 24]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, layer in zip(axes.flat, layers_grid):
        coords = _build_coords_all_concepts(layer, method)
        _scatter_all_concepts(ax, coords, f"layer {layer}")
    axes.flat[0].legend(handles=_legend_handles(), loc="best", fontsize=8, framealpha=0.92)
    fig.suptitle(
        f"{method.upper()} of sentence activations — concept clusters emerging with depth",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = OUT / f"emergence_vs_depth_{method}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path.name}")


# ---------------- README ----------------
def write_readme():
    lines = ["# Steering-harness plots\n",
             "Generated by `docs/make_plots.py` from `results/`.\n"]

    def section(title, body):
        lines.append(f"## {title}\n\n{body}\n")

    def img(path, caption):
        return f"![{caption}](./{path})\n\n*{caption}*\n"

    section("1. Projection histograms (per concept × layer)",
            "Last-token residuals projected onto the mean-diff unit vector. "
            "Separation = linearly extractable concept.\n\n" +
            "\n".join(img(f"proj_{c}_layer{L:02d}.png", f"{c} — layer {L}")
                      for c in CONCEPTS for L in LAYERS))

    section("2. 2D PCA of last-token residuals",
            "PCA on the 60 stacked activations (30 positive + 30 negative) per layer.\n\n" +
            "\n".join(img(f"pca_{c}_layer{L:02d}.png", f"{c} — layer {L}")
                      for c in CONCEPTS for L in LAYERS))

    section("3. Norm of diff vector across layers",
            "\n".join(img(f"diff_norm_{c}.png", f"{c} — ||diff|| vs layer")
                      for c in CONCEPTS))

    section("4. cos(positive_mean, negative_mean) across layers",
            img("cos_pn_vs_layer.png",
                "All concepts overlaid — closer to 1.0 means more-shared background"))

    section("5. Layer-to-layer unit vector rotation",
            "Cosine between the steering unit at layer i and layer j.\n\n" +
            "\n".join(img(f"unit_rotation_{c}.png", f"{c} — unit rotation heatmap")
                      for c in CONCEPTS))

    section("6. Emergence heatmap (layer × α)",
            "Keyword hit count per cell from runs.jsonl.\n\n" +
            "\n".join(img(f"emergence_{c}.png", f"{c} — keyword hits")
                      for c in CONCEPTS))

    section("7. Typical residual norm across layers",
            img("resid_norm_vs_layer.png",
                "Why α needs to scale with layer depth"))

    (OUT / "README.md").write_text("".join(lines))


def main():
    runs = load_runs()
    print(f"loaded {len(runs)} runs")
    print("--- the big-picture plots (all concepts in one space + arrows) ---")
    plot_all_concepts_at_layer(21, "pca")
    plot_all_concepts_at_layer(21, "umap")
    plot_emergence_vs_depth("pca")
    plot_emergence_vs_depth("umap")
    print("--- per-concept diagnostics ---")
    plot_projection_hists(); print("projection hists done")
    plot_pca(); print("pca done")
    plot_diff_norm(); print("diff norm done")
    plot_cos_pn(); print("cos pn done")
    plot_unit_rotation(); print("unit rotation done")
    plot_emergence(runs); print("emergence done")
    plot_resid_norm(); print("resid norm done")
    write_readme(); print("readme done")


if __name__ == "__main__":
    main()
