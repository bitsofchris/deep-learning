# Phase 1 — Supervised Direction Finding

A recap of Track 1 from [plan.md](plan.md): what we built, what it teaches, and how to walk someone through it.

---

## What we built

A new **Directions** tab in the embedding explorer that turns the "gender experiment" from the plan into a working microscope.

**Backend** — 7 endpoints in [backend/server.py](backend/server.py), plus a `directions` SQLite table:

| Endpoint | Purpose |
|---|---|
| `POST /directions/create` | `v = mean(A) − mean(B)`, unit-normalised, persisted |
| `POST /directions/create-random` | random unit vector (control) |
| `GET /directions` | list + pairwise cosine matrix |
| `DELETE /directions/{id}` | remove |
| `POST /directions/{id}/project` | scalar `e·v̂` for every cached embedding (filter- and Matryoshka-aware) |
| `POST /directions/auc-scan` | per-dim AUC separating two tag values |
| `POST /directions/matryoshka` | rebuild the direction at 32/64/…/3072 dims, report 1D AUC |

**Frontend** — [frontend/src/components/DirectionsPanel.tsx](frontend/src/components/DirectionsPanel.tsx):

- Create-form (tag · A − B) and random-direction button
- Directions list with active-row highlight + delete
- Pairwise cosine matrix between stored directions
- For the active direction:
  - **Projection histogram** — stacked by label
  - **Per-dim AUC curve** — with a "distributed representation" TLDR
  - **Matryoshka sweep** — AUC vs. truncation dim
- Dim selector to re-project at 32/64/128/256/512/1024/3072

---

## Core ideas

### Distributed representation — the ML insight
Meaning isn't stored in a single dimension; it's stored on a **direction** that recruits thousands of dimensions, each contributing a small, coherent piece. This is arguably *the* central insight of modern deep learning: superposed, overlapping directions give exponential representational capacity, graceful degradation, and natural compositionality.

### Finding a concept direction
```
v = mean(class_A embeddings) − mean(class_B embeddings)
v̂ = v / ‖v‖                    # unit-normalise
```
Everything uncorrelated with the A/B label averages out to zero. The concept signal adds up coherently. No pairing needed — `mean(aᵢ − bᵢ) = mean(aᵢ) − mean(bᵢ)`.

### Reading the histogram (the money visual)
`score = e · v̂` — one scalar per embedding. Since OpenAI embeddings are unit-norm (`‖e‖ ≈ 1`), this equals `cos(angle from v̂)`. Positive → aligned with A-ness, negative → aligned with B-ness, zero → orthogonal (no signal along this axis). Two clean humps = the model had a coherent internal representation of the concept all along.

### Every direction is a contrast
A direction isn't "gender" — it's "**male-ness vs. female-ness**." A continuum with two ends, not a category. You never find "the X direction," you find "the X-vs-Y direction." Picking the negative side is the craft.

### Embedding-space ≠ activation-space
The directions in this tool live in a frozen encoder's output space (what text *looks like* to a model). They aren't injectable — you can't "Golden Gate Claude" with them. That requires activation-space directions inside an open-weights model (Track 3). Same mean-diff math, different plumbing.

---

## Demo walkthrough

Starts at `Directions` tab with the labeled 200-sentence dataset loaded.

### 1. Build the gender direction
Create: `name=gender`, `tag=gender`, `A=female`, `B=male` → **Create**.
- Histogram: two clean humps, female right, male left, minimal overlap.
- *"This is one number per sentence, from 3072 dims. It's the projection onto a direction we found with one subtraction."*

### 2. Show the negative result (AUC scan)
Scroll to the per-dim AUC curve.
- Noisy band around 0.5; best single dim peaks ~0.8.
- *"No single coordinate encodes gender. If we only had individual dims, we'd be stuck."*
- Read the blue callout: distributed representation, meaning-as-direction.

### 3. Contrast with the full direction
Back up to the histogram.
- *"Same 200 sentences, same embeddings, but instead of looking at any one dim we dot-product with the full direction. One scalar, clean separation. That gap between the AUC curve and this histogram **is** distributed representation."*

### 4. Build a second direction and check independence
Create: `name=class`, `tag=class`, `A=royal`, `B=common`.
- Scroll to the cosine matrix. Gender ↔ class should sit near 0.
- *"Two concepts the model represents on near-orthogonal axes. You could remove one surgically without disturbing the other."*

### 5. Random control
Create a random direction.
- Projection histogram: single overlapping blob.
- Cosine matrix: near-zero against everything.
- *"A random arrow in a 3072-d space is orthogonal to basically every real direction. Which is how we know the learned directions are doing actual work."*

### 6. Matryoshka — where does it break?
Select the gender direction, flip the dim selector down: 3072 → 1024 → 256 → 128 → 64 → 32.
- Watch the histogram humps blur. Watch the Matryoshka AUC curve descend.
- *"The direction is robust for a while — even 256d still separates cleanly — then collapses. That tipping point is where the truncation has destroyed the geometry the concept lived on."*

### 7. Project untagged content
Clear the filter bar so all 411 cached items project.
- Labeled points form the two humps; untagged items (grey `(none)`) pile up near zero with some spread.
- *"Any text we've embedded can be scored on this axis — including your Obsidian notes. The tool generalises."*

---

## What I internalised

- **Distributed representation**, made tangible. Not a cat neuron; a cat direction.
- **Mean-diff** as the trivial-math hammer for finding a direction given any contrast.
- **Dot product with a unit vector** as the geometric operation that collapses a 3072-d question to a 1-d answer.
- **`‖e‖`**, **`v̂`**, `e · v̂ = cos(θ)` — the vector-math vocabulary for talking about concepts as directions.
- **Directions are relative** — the choice of the B-side determines what you isolate.
- **Embedding space ≠ activation space** — this tool builds intuition; Track 3 is where you actually steer a model.
- **Matryoshka** isn't just size reduction; it's a resolution knob on concept geometry, and different concepts break at different resolutions.

---

## Deferred to next pass

- **Ablation toggle on UMAP** — subtract `(e · v̂) v̂` from every embedding before projecting, verify that other tag structure survives. Needs a `direction_id` param on `POST /umap` and a toggle on the UMAP tab.
- **Dalio-style contrast** — load two text piles via the loader, build a custom direction, project personal notes.
- **Track 3** — move this same mean-diff math into a local model's residual stream and actually steer generation.
