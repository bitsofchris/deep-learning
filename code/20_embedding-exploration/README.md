# Embedding Explorer

A hands-on tool for visualizing and comparing text embeddings from OpenAI's `text-embedding-3-large` model. Paste text, get embeddings, see how meanings relate.

Built as a microscope for developing intuition about embeddings — not a production tool. Extended with a labeled-dataset mode for structured demos (e.g. the gender × class sentences demo).

## Launch

```bash
cd code/20_embedding-exploration
./start.sh
```

Opens at **http://localhost:5173**. Ctrl-C to stop both processes.

**Requirements:** Python 3.11+, Node 18+, `OPENAI_API_KEY` in `keys.env` at the repo root.

On first run `start.sh` installs Python deps (`fastapi uvicorn numpy openai umap-learn`) and runs `npm install` automatically.

---

## Two modes in one tool

### Freeform mode
Type any text, embed it, analyze it. No tags required. This is the original exploration microscope — use it to build intuition.

### Labeled-dataset mode
Items carry optional tags (`gender`, `class`, `length`). Once any items have tags, a **filter bar** appears at the top — you can filter the cache and UMAP by any combination of tag values. This is what powers structured demos where you want to see clusters emerge by category.

Both modes share one SQLite DB (`backend/embeddings.db`). Tagged and untagged items live side-by-side — untagged items simply have NULL tag values.

---

## Ingesting a labeled dataset

The loader takes a JSON file of tagged sentences, batches them to OpenAI, and writes straight into `embeddings.db` with tags populated. No UI involvement — just run it, then start the tool.

**JSON format** (save as e.g. `synthetic_sentences.json` in this folder):

```json
[
  {"text": "The queen addressed her court", "gender": "female", "class": "royal", "length": "short"},
  {"text": "A farmer tilled his field at dawn",  "gender": "male",   "class": "common", "length": "short"},
  ...
]
```

**Run the loader:**

```bash
cd backend
export $(grep -v '^#' ../../../keys.env | xargs)
python3 load_demo.py ../synthetic_sentences.json
```

- Batches 100 sentences per OpenAI request
- Skips any text that's already in the DB (safe to re-run)
- Prints progress per batch and a final count

Then `./start.sh` — the filter bar appears automatically and the items are available in every view.

---

## Views

### Data (home tab)
Text input, embed button, and a scrollable items list. Below the input: **tag pickers** (`gender`, `class`, `length`). Set them before embedding to apply tags. Above the list: **Select all** (respects the active filter) · **Clear**. Each row has:
- Checkbox — select for analysis views
- Tag pills — colored by value
- ✕ — delete

### Fingerprint
**Select 1+ texts.** Line chart — x = dimension index (0–3071, sampled every 12th), y = raw value. One trace per selected text. See the shape of a vector.

### Diff
**Select exactly 2.** Three panels: 1D heatmap strip showing |A−B| intensity per dim, full difference curve, and top-15 most discriminating dimensions table.

### Matrix
**Select 2+.** N×N cosine similarity grid with colored cells, plus all pairs ranked by similarity.

### Matryoshka
**Select exactly 2.** Cosine similarity at 256 → 512 → 1024 → 2048 → 3072 dimensions. Elbow annotation marks biggest drop.

### UMAP
**Uses: all cached items (filtered by active filter bar).**

Three controls:
- **Compute UMAP** — runs the projection (cached to disk after first run)
- **Dims** — `32 | 128 | 256 | 512 | 1024 | 3072` — Matryoshka truncation. Lower dims = coarser semantic structure. Switching dims auto-recomputes.
- **Color by** — `none | gender | class | length` — color points by any tag column. Legend shows value counts.

Selected items (checked in Cache) are ringed in white. Hover any point for full text + tag values.

This is the main demo view. Flow:
1. Load labeled data → filter all on → compute UMAP at 3072d → color by gender → see two clouds
2. Drop to 32d → see if coarse gender structure still holds
3. Color by class → same points, different story
4. Color by length → if meaning dominates length, clusters shouldn't separate

---

## How it works

```
keys.env → start.sh → backend (FastAPI :8000) + frontend (Vite :5173)
                              ↓
                    embeddings.db (SQLite)
                    .umap_cache/  (pickle files)
```

- Backend handles all OpenAI API calls. Key never touches the browser.
- Embeddings stored as raw `float32` blobs. Subsequent requests for the same text are instant cache hits.
- UMAP projections cached to `backend/.umap_cache/` keyed by a SHA-1 of the matrix bytes — only recomputes when the set of points or truncation dim changes.
- Tag columns (`gender`, `class`, `length`) added to the schema automatically on first run (ALTER TABLE). Existing data is preserved.

---

## Files

```
20_embedding-exploration/
├── start.sh                      # single launch command
├── backend/
│   ├── server.py                 # FastAPI — 9 endpoints
│   ├── load_demo.py              # bulk-embed a labeled JSON file
│   ├── embeddings.db             # single shared cache
│   └── .umap_cache/              # pickle files keyed by matrix hash
└── frontend/
    └── src/
        ├── App.tsx               # tabs, filter bar, global state
        ├── api.ts                # typed API client
        ├── types.ts              # shared interfaces
        └── components/
            ├── CachePanel.tsx    # input, tag pickers, cache list, clone
            ├── FilterBar.tsx     # tag value chips at top
            ├── FingerprintChart.tsx
            ├── DiffHeatmap.tsx
            ├── SimilarityMatrix.tsx
            ├── MatryoshkaCurve.tsx
            ├── UmapScatter.tsx   # dim toggle + color-by-tag
            └── ArithmeticSandbox.tsx  # hidden — modern sentence embeddings
                                       # don't do clean A−B+C arithmetic
```

UMAP caching helpers (`_matrix_hash`, `_truncate_normalize`, `_normalize_coords`, `compute_umap`) reused from [`openaugi/experiments/knowledge-explorer`](../../../openaugi/experiments/knowledge-explorer).

---

## Schema

Single table, `embeddings`:

```sql
CREATE TABLE embeddings (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  text       TEXT    UNIQUE NOT NULL,
  embedding  BLOB    NOT NULL,          -- float32[3072]
  gender     TEXT,                       -- nullable tag
  class      TEXT,                       -- nullable tag
  length     TEXT,                       -- nullable tag
  created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);
```

Adding a new tag column: add it to `TAG_COLUMNS` in `backend/server.py` and to `backend/load_demo.py`. The migration in `init_db()` will `ALTER TABLE ADD COLUMN` on next start.
