# Prompt for a separate Claude: generate plots for the steering-harness experiments

Paste everything below into a fresh Claude Code session running in the `code/21_steering-harness/` directory.

---

## Context

I ran a series of activation-steering experiments on Gemma-2-2B using mean-difference contrastive pairs. Three concepts:

- **golden_gate** (v1): "Golden Gate Bridge" sentences vs. "bridges elsewhere in the world" sentences
- **golden_gate_v2**: same concept, minimal pairs — identical syntax, only difference is "the Golden Gate Bridge" vs. "the bridge"
- **pickles**: pickle-specific sentences vs. diverse other foods

For each concept I:

1. Harvested last-token residual-stream activations at layers 6, 9, 12, 15, 18, 21, 24 for 30 positive and 30 negative sentences
2. Computed mean-diff steering vectors at each layer (`unit = (pos_mean - neg_mean) / ||pos_mean - neg_mean||`)
3. Injected `α × 0.1 × typical_resid_norm × unit` into the residual stream during generation at that layer
4. Ran a 6×5 grid (layers × α ∈ {2, 4, 6, 8, 10}) with a fixed prompt, logging the completion for each cell

**Key finding**: pickles produced a clean obsession mode at deep layers (layer 21-24 → full "Pickles Pickles Pickles" loop). Golden Gate v1 drifted to "coastal California" rather than the bridge itself. Golden Gate v2 (minimal pairs) was the targeted fix for that.

## Where the data is

```
results/
├── runs.jsonl                    # append-only log, one JSON object per (concept, layer, alpha) cell
├── runs.md                       # same data as a markdown table
└── vectors/
    ├── golden_gate/
    │   ├── sentences.json        # {"positive": [...], "negative": [...]}
    │   ├── layer_06.npz
    │   ├── layer_09.npz
    │   ├── layer_12.npz
    │   ├── layer_15.npz
    │   ├── layer_18.npz
    │   ├── layer_21.npz
    │   └── layer_24.npz
    ├── golden_gate_v2/
    │   └── [same structure]
    └── pickles/
        └── [same structure]
```

### Schema of each `layer_NN.npz`

```python
import numpy as np
d = np.load("results/vectors/pickles/layer_21.npz")
d["positive_acts"]   # shape (30, 2304)  — residual at last token, per positive sentence
d["negative_acts"]   # shape (30, 2304)  — same for negatives
d["positive_mean"]   # shape (2304,)     — mean of positive_acts
d["negative_mean"]   # shape (2304,)     — mean of negative_acts
d["diff"]            # shape (2304,)     — positive_mean - negative_mean
d["unit"]            # shape (2304,)     — diff / ||diff|| (the steering vector)
d["typical_norm"]    # scalar            — avg ||resid|| at this layer/position
d["cos_pos_neg"]     # scalar            — cosine(positive_mean, negative_mean)
```

### Schema of each line in `runs.jsonl`

```json
{
  "timestamp": "2026-04-23T12:45:05",
  "concept": "pickles",
  "model": "google/gemma-2-2b",
  "layer": 21,
  "alpha": 4.0,
  "prompt": "My favorite food in the whole world is",
  "output": " pickled kraut…but if you don't know what they are...",
  "resid_norm": 446.4,
  "cos_pos_neg": 0.971,
  "pos_count": 30,
  "neg_count": 30,
  "pos_examples": ["Pickles are my favorite snack...", ...],
  "neg_examples": ["Pizza is my favorite meal...", ...],
  "pos_strategy": "last",
  "extra": {"alpha_scale": 0.1, "seed": 0, "grid": true}
}
```

The layer=-1, alpha=0 rows are unsteered baselines for each concept.

## What to plot

Produce each plot as a separate PNG under `docs/plots/`. Use matplotlib. Don't worry about styling — I'll handle that in a second pass. One chart per figure, readable axis labels, no fancy theming. Use the data as-is.

### 1. Per-concept: projection histograms by layer

For each concept, one figure per layer (so 7 figures × 3 concepts = 21 PNGs, named `proj_<concept>_layer<NN>.png`):

- Project every positive activation onto that layer's `unit` vector → scalar per positive
- Project every negative activation onto `unit` → scalar per negative
- Overlay two histograms on the same axes. Title: `{concept} — layer {N} — cos(p,n)={...}`
- Expect: for pickles at layers 18-24 the histograms should be visibly separated; for golden_gate v1 they should overlap heavily

### 2. Per-concept: 2D PCA of activations by layer

For each concept, one figure per layer (`pca_<concept>_layer<NN>.png`):

- Stack `positive_acts` + `negative_acts` = (60, 2304)
- Fit `sklearn.decomposition.PCA(n_components=2)`, transform
- Scatter: positives one color, negatives another
- Title includes concept + layer

### 3. Per-concept: norm of diff vector across layers (line chart)

One figure per concept (`diff_norm_<concept>.png`):
- x-axis: layer number (6, 9, 12, 15, 18, 21, 24)
- y-axis: `||diff||` (L2 norm)
- Where the concept is "strongest" in the residual stream

### 4. All concepts overlaid: cos(pos_mean, neg_mean) across layers

Single figure `cos_pn_vs_layer.png`:
- x: layer. y: `cos_pos_neg`
- One line per concept, labeled
- Caption hint: closer to 1.0 = positive/negative groups have more similar mean activations (weaker contrastive signal)

### 5. Per-concept: layer-to-layer unit vector rotation heatmap

One figure per concept (`unit_rotation_<concept>.png`):
- 7×7 matrix where cell (i,j) = `cos(unit_at_layer_i, unit_at_layer_j)`
- Imshow with a colormap, annotate cells with the value
- Tells us whether the steering direction is "the same direction that gets sharper" vs. "different directions at different depths"

### 6. Grid heatmap: concept emergence across (layer × α)

One figure per concept (`emergence_<concept>.png`). Read from `runs.jsonl`:
- Rows: layer (6, 9, 12, 15, 18, 21, 24 — use whatever is in the data)
- Cols: alpha (2, 4, 6, 8, 10)
- Cell value: keyword hit count. For pickles, count matches of `pickle|pickled|pickles|gherkin|kraut` in the output (case-insensitive). For golden_gate and golden_gate_v2, count matches of `golden gate|Golden Gate`. Use a heatmap color scheme.
- Alternative: use simple binary (present/absent) if counts are noisy

### 7. Residual norm across layers (shared)

Single figure `resid_norm_vs_layer.png`:
- x: layer. y: `typical_norm` from the npz files. One line per concept (they'll be nearly identical — this is a property of the model, not the pairs)
- Caption hint: explains why deeper layers need lower α

## Output

- Write all plots to `docs/plots/`
- Also write a single `docs/plots/README.md` that embeds each plot with a one-sentence caption, in the order above
- Keep code in `docs/make_plots.py`, runnable as `python3 docs/make_plots.py` from the project root

Ask before adding any new dependency beyond numpy, matplotlib, scikit-learn.
