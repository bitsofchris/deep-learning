# PCA Explorer — walkthrough

Track 2 of [plan.md](../plan.md): *discover what axes of variation exist in text you haven't explicitly labeled for.*

## What we built

**Backend** (`backend/server.py`)
- `compute_pca(matrix, k)` — fits sklearn PCA, caches `(scores, explained_variance_ratio)` to `.pca_cache/` keyed by matrix hash + k.
- `POST /api/pca` — takes a filter (or explicit ids) + truncation `dims` + `n_components`, returns per-item scores on every PC plus the explained-variance ratio.

**Frontend** (`frontend/src/components/PcaExplorer.tsx`)
- Scope / Dims / Components controls
- Scree list: one row per PC, bar sized by relative explained variance, click to make active
- Ranked list: every item sorted by its score on the active PC, with a diverging bar (green positive / red negative) and optional tag-pill overlay

---

## How to use it

1. `./start.sh`, open http://localhost:5173, click **PCA**.
2. Make sure the filter bar includes the slice you want (e.g. all 200 labeled sentences).
3. Click **Compute PCA**.
4. Click **PC1** in the scree list. Look at the top 10 and bottom 10 in the ranked list.
5. Toggle a tag pill (`gender`, `class`, `length`) — do the pills cluster at the extremes?
6. If yes, you've just recovered that concept *without telling the algorithm about it*.
7. Repeat for PC2, PC3, …

---

## What each part is showing you, and why

### The scree list
Each PC is a direction in 3072-d embedding space, ranked by how much variance of the dataset lies along it. The percentage is how much of the spread is captured by that single axis.

**Why useful:** tells you where signal ends and noise begins. If PC1 = 18% and PC50 = 0.2%, the first few PCs are where the structure lives. Clicking through them top-down is an ordered tour of "most-important axes first."

### The ranked list (per PC)
Every item projected onto the chosen PC, sorted by score. Positive and negative extremes are the two poles of that axis; items near 0 are neutral on this dimension.

**Why useful:** PCs are just vectors of numbers — they only become *interpretable* when you read what lives at the extremes. If the top of PC1 is all queens and princesses and the bottom is all farmers and blacksmiths, that axis is "royal ↔ common." You just named it.

### Tag-pill overlay
Colors each row by a known tag value (e.g. gender). Gives you an instant visual check: does the unsupervised axis line up with a label you already have?

**Why useful:** it's the sanity check. On the labeled 200, if PC1 colored by `class` shows royal on one end and common on the other, PCA recovered a human concept from raw geometry — proof the method works before you point it at unlabeled data. If the tags scatter randomly, the PC is capturing something *else* — still real, just not that tag. That's when it gets interesting.

### Matryoshka (Dims) toggle
Recomputes PCA after truncating each embedding to the first N dimensions. OpenAI's `text-embedding-3-large` is trained so earlier dims carry coarser semantic structure.

**Why useful:** tells you how robust an axis is. If PC1's meaning holds at 32 dims, it's a dominant, cheap-to-compute concept. If it only appears at 3072 dims, it's a fine-grained distinction that needs full resolution.

### Scope: All vs. Selected
"All" fits PCA on the whole filtered slice. "Selected" fits on just the items you checked in the Data tab.

**Why useful:** this is the "topic stratification" lever from the plan. Running PCA on the whole corpus finds the *biggest* variance — which is usually topic. Running it on one topic slice (e.g. only health-related notes) finds *within-topic* axes, which is where the interesting style/tone/stance structure lives.

---

## The core idea, once more

> Meaning lives on directions in high-dimensional space, not on individual dimensions.

PCA is the simplest tool that finds those directions without being told what to look for. The ranked list turns each direction into a human-readable story by showing you its two poles.

That's the whole trick. Everything else — within-topic PCA, naming axes, projecting new text onto a named axis as a lightweight classifier — is the same move applied in different contexts.

---

## What's next

- **Name-the-axis persistence** — save a PC with a human label, project new items onto it later (bridges into Track 1's directions machinery)
- **Within-topic PCA on a vault slice** — point the tool at an Obsidian slice instead of the synthetic 200
- **Compare PCs across slices** — do the same axes appear in 2024 notes vs. 2025 notes?
