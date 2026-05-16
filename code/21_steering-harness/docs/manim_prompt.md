# Prompt for a Manim-focused Claude: make animations for the steering-harness blog post

Paste the rest of this file into a fresh Claude Code session running in the `code/21_steering-harness/` directory. Target: Manim Community Edition (`manim` on PyPI). Output MP4s (or GIFs if preferred) under `docs/animations/`.

---

## Project context (short)

Activation-steering experiments on Gemma-2-2B (26 layers, d_model=2304). For each concept we harvested last-token residual activations at layers {6, 9, 12, 15, 18, 21, 24}, computed a mean-difference unit vector per layer, and injected `α × 0.1 × typical_norm × unit` into the residual stream at generation time. Full writeup: `docs/blog_draft.md`.

Concepts:
- **pickles** — cleanly linear, produces obsession mode at deep layers
- **golden_gate** (v1) — drifts to "coastal California," not the bridge itself
- **golden_gate_v2** — minimal-pairs redo, sharper
- **pirate** — used in the composition finale

## Full paths to the data

Absolute paths so you can read them directly:

```
/Users/chris/repos/deep-learning/code/21_steering-harness/
├── results/
│   ├── runs.jsonl                                    # one JSON per (concept, layer, α) cell + baselines
│   ├── runs.md                                       # same data, markdown table
│   └── vectors/
│       ├── pickles/
│       │   ├── sentences.json                        # {"positive": [...30...], "negative": [...30...]}
│       │   ├── layer_06.npz
│       │   ├── layer_09.npz
│       │   ├── layer_12.npz
│       │   ├── layer_15.npz
│       │   ├── layer_18.npz
│       │   ├── layer_21.npz
│       │   └── layer_24.npz
│       ├── golden_gate/      (same 7 layers + sentences.json)
│       ├── golden_gate_v2/   (same)
│       └── pirate/           (same)
├── docs/
│   ├── blog_draft.md                                 # narrative + findings
│   ├── plotting_prompt.md                            # static-plot spec (already executed)
│   ├── make_plots.py                                 # static plots — already run
│   ├── plots/                                        # 53 PNGs (PCA, histograms, heatmaps)
│   └── animations/                                   # <-- WRITE MP4s HERE
└── data/
    ├── golden_gate_pairs.py
    ├── golden_gate_v2_pairs.py
    ├── food_pairs.py
    └── pirate_pairs.py
```

### Schema of each `layer_NN.npz`

```python
import numpy as np
d = np.load("results/vectors/pickles/layer_21.npz")
d["positive_acts"]   # (30, 2304)
d["negative_acts"]   # (30, 2304)
d["positive_mean"]   # (2304,)
d["negative_mean"]   # (2304,)
d["diff"]            # (2304,) = positive_mean - negative_mean
d["unit"]            # (2304,) = diff / ||diff||
d["typical_norm"]    # scalar, avg ||resid|| at this layer
d["cos_pos_neg"]     # scalar
```

### Schema of each row in `runs.jsonl`

```json
{
  "concept": "pickles",
  "layer": 21,
  "alpha": 4.0,
  "prompt": "My favorite food in the whole world is",
  "output": " pickled kraut...",
  "resid_norm": 446.4,
  "cos_pos_neg": 0.971,
  "extra": {"grid": true, "seed": 0, "alpha_scale": 0.1}
}
```

Baselines have `layer=-1, alpha=0`.

## What already exists (don't redo)

Static plots in `docs/plots/` already cover PCA scatter, projection histograms, diff-norm curves, cosine-vs-layer, rotation heatmaps, and emergence heatmaps. Your job is the *motion* — the animations those can't convey.

---

## Animations to produce

One MP4 per section. Reasonable target: 720p–1080p, 8–20 s each unless noted. Use real data where possible; where a value isn't in the data (e.g. token-level probabilities), synthesize plausible monotonic curves labeled clearly as "schematic."

Use matplotlib output from static plots as the *thumbnail* style where it helps (viewer should feel these are the same concepts). Preferred palette: C0 (blue) = positive/target, C1 (orange) = negative, neutral grey for baseline/axes.

### 1. `steering_vector_addition.mp4` — "what even is steering?" (15 s)

The conceptual opener. Show, in 2D for clarity:
- A cartoon residual stream vector `h` at layer 18 (arrow from origin).
- The unit steering vector `u` appears alongside, labeled "mean(pickle) − mean(other)".
- Slider for α (0 → 10). As α grows, a scaled copy `α · 0.1 · ||h|| · u` fades in.
- Vector sum `h + α·…·u` appears, visibly rotating from original direction toward `u`.
- Caption at bottom changes with α: "α=0: the beach" → "α=4: pickled kraut" → "α=10: Pickles Pickles Pickles" (pull captions from `runs.jsonl` for concept="pickles", layer=21).

**Why Manim wins:** showing literal vector addition with a morphing tip is its native strength.

### 2. `alpha_sweep_morph.mp4` — the money clip (12 s)

Single fixed prompt `"My favorite food in the whole world is"`, layer 21, α sweeping 0 → 10.
- Big typewriter-style text area shows the generated completion for each α, pulled from `runs.jsonl` where `concept=pickles, layer=21, grid=true`.
- Below it, an α dial (0–10) turning smoothly. Snap the visible text to the *nearest* logged α (0, 2, 4, 6, 8, 10) so every phrase is real.
- Subtle residual-norm bar on the side showing the injection magnitude growing.
- End frame freezes on α=10 "Pickles Pickles Pickles…" with the word "Pickles" repeating and fading out.

**Why Manim wins:** the drift from coherent → obsessive is the most visceral part of the story.

### 3. `pca_with_steering_arrow.mp4` — geometry → behavior (15 s)

- Start by loading `results/vectors/pickles/layer_21.npz`, stacking positive_acts + negative_acts (60 × 2304), PCA to 2D.
- Animate the two scatter clouds appearing (blue positives, orange negatives).
- Draw `positive_mean` and `negative_mean` as filled circles, then an arrow from neg_mean → pos_mean labeled `diff`.
- Normalize arrow to `unit` and re-draw.
- Then: animate a "generated token" dot starting at the negative centroid. As α increases (shown by a small gauge), the dot slides along the `unit` direction. At key α values, pop up the actual generated phrase beside the dot.
- Compare shot: same animation for `golden_gate` layer 21 — clouds *overlap*, dot barely moves relative to cloud variance. Title "linearly extractable vs. not."

### 4. `concept_composition.mp4` — three directions, one output (18 s)

The finale's centerpiece.
- Simplified 3D axes (or 3 stacked horizontal tracks, one per concept).
- Layer-stack diagram on the left: 26 boxes; highlight layer 12 (pirate), 21 (pickles), 24 (GG v2) with colored arrows.
- Unit vectors for each concept animate into those layers.
- Right side: typewriter text area. Start with baseline for prompt `"The best adventure I can imagine is"` (look up in `runs.jsonl` layer=-1, alpha=0).
- Turn on pirate (only) → text retypes with pirate-flavored output.
- Turn on pickles (+ pirate) → retype with the invented "pickle-a-de-do" style.
- Turn on GG v2 → full composed output: "Golden Dreadken," "California Gilbertos," etc. (pull from the blog or `runs.jsonl` compose rows if present).
- End frame: all three toggles lit, composed output fully on screen.

**Why Manim wins:** the *addition* of independent knobs is literally vector math — perfect for animated composition.

### 5. `layer_sweep_landscape.mp4` — where does the concept live? (12 s)

For concept=`golden_gate` (v1): sweep layer from 6 → 24 at fixed α=4. Pull outputs from `runs.jsonl`.
- Left panel: vertical layer stack, current layer highlighted.
- Center panel: text fades in/out for each layer.
- Right panel: a tiny thumbnail of the PCA scatter at that layer (just a simplified shape, or a mini-screenshot from `docs/plots/pca_golden_gate_layerXX.png`).
- Caption walks through the blog narrative: "layer 6 → Kalahari Desert (noise)", "layer 15 → physics equations", "layer 18 → Santa Cruz surfing", "layer 24 → bridge bridge bridge."

### 6. `cos_pn_diagnostic.mp4` — why the signal is hidden (8 s, optional)

Animate two vectors at cos=0.99 (nearly parallel), then subtract to reveal the tiny perpendicular `diff`. Zoom into that sliver and label it "the steering direction." Punchline: "everything important lives in a 1% perturbation."

**Why Manim wins:** this is exactly the kind of "zoom into a tiny angle between near-parallel vectors" that is painful in matplotlib.

### 7. `obsession_emergence.mp4` — the (layer × α) grid lighting up (10 s, optional)

Take the emergence heatmap (`docs/plots/emergence_pickles.png`) and animate cells lighting up in order of α sweep, with a small text popup at each cell showing the actual output snippet. Ends with the brightest cell (layer 24, α=4 region) flashing and the full obsession text overlaying.

---

## Technical notes for the Manim agent

- **Environment:** `/Users/chris/repos/deep-learning/.venv` is the project venv. Manim isn't installed yet — ask before adding it (`manim` from PyPI, community edition). Expect large dep footprint (LaTeX, ffmpeg). Confirm with the user first.
- **Text handling:** long generation outputs don't fit on screen. Use a monospace font, wrap to ~50 chars, and animate with `AddTextLetterByLetter` or `Write` so the "the model is generating" feel is there.
- **Don't re-run the model.** All outputs are already in `runs.jsonl`. Parse it, don't call the GPU.
- **Don't redo PCA in Manim.** Compute in NumPy/sklearn first, pass 2D points to Manim. Same for unit-vector math.
- **Keep animations short.** Blog readers scrub — 8–20 s each, looping cleanly.
- **Filename convention:** `docs/animations/NN_<slug>.mp4` (01_steering_vector_addition, 02_alpha_sweep_morph, etc.) so they sort in narrative order.
- **Write a `docs/animations/README.md`** mirroring `docs/plots/README.md` — each animation embedded with a one-sentence caption in the order above.
- **Ask before adding any dep** beyond `manim`, `numpy`, `matplotlib`, `scikit-learn`.

## Priority if you only get to some

1. `alpha_sweep_morph.mp4` — single most shareable clip
2. `steering_vector_addition.mp4` — the conceptual opener
3. `concept_composition.mp4` — the finale
4. `pca_with_steering_arrow.mp4` — geometry → behavior bridge
5. `layer_sweep_landscape.mp4`
6. the two optional ones

Deliver in that order.
