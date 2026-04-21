"""
Embedding Explorer backend.

Single SQLite DB (embeddings.db) with optional tag columns:
  gender, class, length — nullable. Freeform items have NULL tags,
  seeded demo items have them populated.

Endpoints:
  POST   /api/embed                       — embed text (cache-first via SQLite)
  POST   /api/cache/list                  — list cached items, with optional filters
  GET    /api/tag-values                  — distinct values per tag column
  DELETE /api/cache/{id}                  — remove from cache
  POST   /api/vectors                     — raw float32 vectors for given IDs
  POST   /api/umap                        — 2D UMAP projection (dims arg for Matryoshka)
  POST   /api/pca                         — PCA scores + explained variance over a slice
  POST   /api/similarity                  — N×N cosine similarity matrix
  POST   /api/matryoshka                  — cosine sim at [256,512,1024,2048,3072] dims
  POST   /api/arithmetic                  — A − B + C, nearest neighbors in cache
  POST   /api/directions/create           — mean(A) − mean(B) unit direction for a tag
  POST   /api/directions/create-random    — random unit direction (control)
  GET    /api/directions                  — list all directions + pairwise cosines
  DELETE /api/directions/{id}             — delete a direction
  POST   /api/directions/{id}/project     — scalar e·v̂ for every cached embedding
  POST   /api/directions/auc-scan         — per-dim AUC between two tag values
  POST   /api/directions/matryoshka       — separation of a direction across dims

UMAP caching helpers reused from:
  openaugi/experiments/knowledge-explorer/backend/server.py
"""

import argparse
import hashlib
import logging
import os
import pickle
import sqlite3
import threading
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

HERE = Path(__file__).parent
DB_PATH = HERE / "embeddings.db"
UMAP_CACHE_DIR = HERE / ".umap_cache"
PCA_CACHE_DIR = HERE / ".pca_cache"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MATRYOSHKA_DIMS = [256, 512, 1024, 2048, 3072]
TAG_COLUMNS = ["gender", "class", "length", "domain"]

_umap_lock = threading.Lock()

app = FastAPI(title="Embedding Explorer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── DB ─────────────────────────────────────────────────────────────────────────


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            text       TEXT    UNIQUE NOT NULL,
            embedding  BLOB    NOT NULL,
            gender     TEXT,
            class      TEXT,
            length     TEXT,
            created_at TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """
    )
    # Migrate: add tag columns if DB pre-dates them
    existing_cols = {
        r[1] for r in conn.execute("PRAGMA table_info(embeddings)").fetchall()
    }
    for col in TAG_COLUMNS:
        if col not in existing_cols:
            log.info("Adding %s column to embeddings", col)
            conn.execute(f"ALTER TABLE embeddings ADD COLUMN {col} TEXT")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS directions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT    UNIQUE NOT NULL,
            tag        TEXT,                         -- NULL for random-control directions
            value_a    TEXT,                         -- positive side (scores > 0)
            value_b    TEXT,                         -- negative side
            vector     BLOB    NOT NULL,             -- float32[3072], unit-normalised
            kind       TEXT    NOT NULL,             -- 'mean_diff' | 'random'
            n_a        INTEGER,
            n_b        INTEGER,
            created_at TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """
    )
    conn.commit()
    conn.close()


def open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── Filters ────────────────────────────────────────────────────────────────────


def _build_filter_sql(filters: dict | None) -> tuple[str, list]:
    """Build WHERE clause from {tag_col: [values,...]} dict. Safe: cols are whitelisted."""
    if not filters:
        return "", []
    clauses, params = [], []
    for col, values in filters.items():
        if col not in TAG_COLUMNS or not values:
            continue
        placeholders = ",".join("?" * len(values))
        clauses.append(f"{col} IN ({placeholders})")
        params.extend(values)
    if not clauses:
        return "", []
    return " WHERE " + " AND ".join(clauses), params


def _row_to_item(r: sqlite3.Row) -> dict:
    return {
        "id": r["id"],
        "text": r["text"],
        "created_at": r["created_at"],
        "tags": {c: r[c] for c in TAG_COLUMNS if r[c] is not None},
    }


# ── OpenAI ─────────────────────────────────────────────────────────────────────


def fetch_embedding(text: str) -> list[float]:
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OPENAI_API_KEY not set — check keys.env")
    try:
        import openai  # type: ignore
    except ImportError:
        raise HTTPException(
            500, "openai package not installed: pip install openai"
        ) from None
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        resp = client.embeddings.create(model="text-embedding-3-large", input=text)
        return resp.data[0].embedding
    except openai.AuthenticationError as e:
        raise HTTPException(
            401, f"OpenAI authentication failed — check your API key: {e}"
        ) from e
    except openai.RateLimitError as e:
        raise HTTPException(429, f"OpenAI rate limit: {e}") from e
    except openai.APIError as e:
        raise HTTPException(502, f"OpenAI API error: {e}") from e


# ── Math helpers ────────────────────────────────────────────────────────────────
# _matrix_hash, _truncate_normalize, _normalize_coords reused verbatim from
# openaugi/experiments/knowledge-explorer/backend/server.py


def _matrix_hash(matrix: np.ndarray) -> str:
    return hashlib.sha1(matrix.tobytes()).hexdigest()[:16]  # noqa: S324


def _truncate_normalize(matrix: np.ndarray, dims: int) -> np.ndarray:
    d = min(dims, matrix.shape[1])
    m = matrix[:, :d].copy()
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    coords = coords.copy()
    for axis in range(2):
        col = coords[:, axis]
        span = col.max() - col.min()
        if span > 0:
            coords[:, axis] = (col - col.min()) / span * 2 - 1
    return coords


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    an = a / (np.linalg.norm(a) + 1e-10)
    bn = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(an, bn))


# ── Cluster-quality metrics (per tag column) ──────────────────────────────────


def _knn_purity(X: np.ndarray, labels: list[str], k: int) -> float:
    """For each point, fraction of its k nearest neighbours that share its label.
    X is assumed L2-normalised → dot product = cosine similarity.
    """
    n = len(X)
    if n <= k + 1:
        return float("nan")
    sim = X @ X.T
    # Fill diagonal with -inf so self is never in top-k
    np.fill_diagonal(sim, -np.inf)
    top = np.argpartition(-sim, k, axis=1)[:, :k]
    scores = np.fromiter(
        (sum(1 for j in top[i] if labels[j] == labels[i]) / k for i in range(n)),
        dtype=float,
        count=n,
    )
    return float(scores.mean())


def _silhouette_cosine(X: np.ndarray, labels: list[str]) -> float:
    try:
        from sklearn.metrics import silhouette_score  # type: ignore
    except ImportError:
        return float("nan")
    try:
        return float(silhouette_score(X, labels, metric="cosine"))
    except ValueError:
        return float("nan")


def _compute_tag_metrics(items: list[dict], matrix: np.ndarray) -> dict:
    """Compute kNN purity + silhouette per tag column with ≥2 distinct values.

    Uses the truncated embedding space (`matrix`), not the UMAP projection.
    Only items that actually have a tag value for the column are included.
    """
    result: dict[str, dict] = {}
    for col in TAG_COLUMNS:
        indices = [i for i, it in enumerate(items) if it["tags"].get(col)]
        if len(indices) < 6:
            continue
        labels = [items[i]["tags"][col] for i in indices]
        n_classes = len(set(labels))
        if n_classes < 2:
            continue
        X = matrix[indices]
        k = min(5, len(indices) - 1)
        knn_p = _knn_purity(X, labels, k=k)
        sil = _silhouette_cosine(X, labels)
        # Chance baseline for kNN purity = 1/n_classes (balanced) — included so
        # the frontend can show "lift over chance"
        baseline = 1.0 / n_classes
        result[col] = {
            "knn_purity": round(knn_p, 4),
            "knn_k": k,
            "silhouette": round(sil, 4),
            "chance_baseline": round(baseline, 4),
            "n_classes": n_classes,
            "n_points": len(indices),
        }
    return result


# ── UMAP (cached, serialized) ───────────────────────────────────────────────────
# compute_umap reused verbatim from
# openaugi/experiments/knowledge-explorer/backend/server.py


def _run_umap(matrix: np.ndarray) -> np.ndarray:
    try:
        import umap  # type: ignore
    except ImportError:
        raise HTTPException(
            500, "umap-learn not installed: pip install umap-learn"
        ) from None
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, max(2, len(matrix) - 1)),
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        low_memory=True,
    )
    return reducer.fit_transform(matrix)


def compute_pca(matrix: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA, return (scores [n, k], explained_variance_ratio [k]). Cached to disk."""
    try:
        from sklearn.decomposition import PCA  # type: ignore
    except ImportError:
        raise HTTPException(
            500, "scikit-learn not installed: pip install scikit-learn"
        ) from None
    PCA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    k = min(n_components, matrix.shape[0] - 1, matrix.shape[1])
    cache_key = _matrix_hash(matrix)
    cache_file = PCA_CACHE_DIR / f"pca_{cache_key}_n{len(matrix)}_k{k}.pkl"
    if cache_file.exists():
        log.info("PCA cache hit — %s", cache_file.name)
        with cache_file.open("rb") as f:
            return pickle.load(f)  # noqa: S301
    log.info("PCA computing %d×%d, k=%d…", *matrix.shape, k)
    pca = PCA(n_components=k, random_state=42)
    scores = pca.fit_transform(matrix)
    evr = pca.explained_variance_ratio_
    with cache_file.open("wb") as f:
        pickle.dump((scores, evr), f)
    log.info("PCA cached → %s", cache_file.name)
    return scores, evr


def compute_umap(matrix: np.ndarray) -> np.ndarray:
    UMAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = _matrix_hash(matrix)
    cache_file = UMAP_CACHE_DIR / f"umap_{cache_key}_{len(matrix)}.pkl"

    if cache_file.exists():
        log.info("UMAP cache hit — %s", cache_file.name)
        with cache_file.open("rb") as f:
            return pickle.load(f)  # noqa: S301

    with _umap_lock:
        if cache_file.exists():
            with cache_file.open("rb") as f:
                return pickle.load(f)  # noqa: S301
        log.info("UMAP computing %d×%d…", *matrix.shape)
        coords = _run_umap(matrix)
        with cache_file.open("wb") as f:
            pickle.dump(coords, f)
        log.info("UMAP cached → %s", cache_file.name)
        return coords


# ── DB helpers ──────────────────────────────────────────────────────────────────


def _load_by_ids(conn: sqlite3.Connection, ids: list[int]) -> list[dict]:
    cols = "id, text, embedding, created_at, " + ", ".join(TAG_COLUMNS)
    if ids:
        ph = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT {cols} FROM embeddings WHERE id IN ({ph}) ORDER BY id",
            ids,
        ).fetchall()
    else:
        rows = conn.execute(f"SELECT {cols} FROM embeddings ORDER BY id").fetchall()
    return [
        {
            "id": r["id"],
            "text": r["text"],
            "embedding": np.frombuffer(r["embedding"], dtype=np.float32).copy(),
            "tags": {c: r[c] for c in TAG_COLUMNS if r[c] is not None},
        }
        for r in rows
    ]


def _load_filtered(conn: sqlite3.Connection, filters: dict | None) -> list[dict]:
    """Load embeddings matching tag filters."""
    cols = "id, text, embedding, created_at, " + ", ".join(TAG_COLUMNS)
    where_sql, params = _build_filter_sql(filters)
    rows = conn.execute(
        f"SELECT {cols} FROM embeddings{where_sql} ORDER BY id",
        params,
    ).fetchall()
    return [
        {
            "id": r["id"],
            "text": r["text"],
            "embedding": np.frombuffer(r["embedding"], dtype=np.float32).copy(),
            "tags": {c: r[c] for c in TAG_COLUMNS if r[c] is not None},
        }
        for r in rows
    ]


# ── Request models ──────────────────────────────────────────────────────────────


class EmbedRequest(BaseModel):
    text: str
    tags: dict[str, str] | None = None


class CacheListRequest(BaseModel):
    filters: dict[str, list[str]] | None = None


class VectorsRequest(BaseModel):
    ids: list[int]


class UmapRequest(BaseModel):
    ids: list[int] | None = None
    filters: dict[str, list[str]] | None = None
    dims: int = 3072


class SimilarityRequest(BaseModel):
    ids: list[int]


class MatryoshkaRequest(BaseModel):
    id_a: int
    id_b: int


class PcaRequest(BaseModel):
    ids: list[int] | None = None
    filters: dict[str, list[str]] | None = None
    dims: int = 3072
    n_components: int = 10


class ArithmeticRequest(BaseModel):
    id_a: int
    id_b: int
    id_c: int


class CreateDirectionRequest(BaseModel):
    name: str
    tag: str
    value_a: str  # positive side (scores > 0 mean "more like A")
    value_b: str


class CreateRandomDirectionRequest(BaseModel):
    name: str
    seed: int | None = None


class ProjectDirectionRequest(BaseModel):
    filters: dict[str, list[str]] | None = None
    dims: int = 3072


class AucScanRequest(BaseModel):
    tag: str
    value_a: str
    value_b: str


class DirectionMatryoshkaRequest(BaseModel):
    tag: str
    value_a: str
    value_b: str
    dims: list[int] | None = None


# ── Endpoints ───────────────────────────────────────────────────────────────────


@app.get("/api/health")
def health():
    return {"status": "ok", "tag_columns": TAG_COLUMNS}


@app.post("/api/embed")
def embed(req: EmbedRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "text cannot be empty")
    tags = req.tags or {}
    conn = open_db()
    try:
        row = conn.execute(
            "SELECT id FROM embeddings WHERE text = ?", (text,)
        ).fetchone()
        if row:
            # Update tags if provided (allows re-tagging via clone flow)
            if tags:
                set_clauses = [f"{c} = ?" for c in TAG_COLUMNS if c in tags]
                if set_clauses:
                    values = [tags[c] for c in TAG_COLUMNS if c in tags]
                    conn.execute(
                        f"UPDATE embeddings SET {', '.join(set_clauses)} WHERE id = ?",
                        [*values, row["id"]],
                    )
                    conn.commit()
            return {"id": row["id"], "text": text, "cached": True}

        vec = fetch_embedding(text)
        arr = np.array(vec, dtype=np.float32)
        col_names = ["text", "embedding"] + [c for c in TAG_COLUMNS if c in tags]
        col_values = [text, arr.tobytes()] + [tags[c] for c in TAG_COLUMNS if c in tags]
        placeholders = ",".join("?" * len(col_values))
        conn.execute(
            f"INSERT INTO embeddings ({', '.join(col_names)}) VALUES ({placeholders})",
            col_values,
        )
        conn.commit()
        row = conn.execute(
            "SELECT id FROM embeddings WHERE text = ?", (text,)
        ).fetchone()
        return {"id": row["id"], "text": text, "cached": False}
    finally:
        conn.close()


@app.post("/api/cache/list")
def list_cache(req: CacheListRequest):
    cols = "id, text, created_at, " + ", ".join(TAG_COLUMNS)
    where_sql, params = _build_filter_sql(req.filters)
    conn = open_db()
    try:
        rows = conn.execute(
            f"SELECT {cols} FROM embeddings{where_sql} ORDER BY created_at DESC",
            params,
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        return {
            "items": [_row_to_item(r) for r in rows],
            "total": total,
        }
    finally:
        conn.close()


@app.get("/api/tag-values")
def get_tag_values():
    """Distinct values per tag column (non-null only)."""
    conn = open_db()
    try:
        result = {}
        for col in TAG_COLUMNS:
            rows = conn.execute(
                f"SELECT DISTINCT {col} FROM embeddings WHERE {col} IS NOT NULL ORDER BY {col}"
            ).fetchall()
            result[col] = [r[col] for r in rows]
        return {"tag_columns": TAG_COLUMNS, "tag_values": result}
    finally:
        conn.close()


@app.delete("/api/cache/{item_id}")
def delete_cache(item_id: int):
    conn = open_db()
    try:
        conn.execute("DELETE FROM embeddings WHERE id = ?", (item_id,))
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


@app.post("/api/vectors")
def get_vectors(req: VectorsRequest):
    conn = open_db()
    try:
        items = _load_by_ids(conn, req.ids)
    finally:
        conn.close()
    return {
        "items": [
            {
                "id": it["id"],
                "text": it["text"],
                "embedding": it["embedding"].tolist(),
                "tags": it["tags"],
            }
            for it in items
        ]
    }


@app.post("/api/umap")
def umap_endpoint(req: UmapRequest):
    """2D UMAP projection. Supports Matryoshka via `dims` param."""
    conn = open_db()
    try:
        if req.ids:
            items = _load_by_ids(conn, req.ids)
        else:
            items = _load_filtered(conn, req.filters)
    finally:
        conn.close()
    if len(items) < 3:
        raise HTTPException(400, f"Need ≥3 embeddings for UMAP, have {len(items)}")
    matrix = _truncate_normalize(np.stack([it["embedding"] for it in items]), req.dims)
    coords = _normalize_coords(compute_umap(matrix))
    metrics = _compute_tag_metrics(items, matrix)
    return {
        "points": [
            {
                "id": items[i]["id"],
                "text": items[i]["text"],
                "tags": items[i]["tags"],
                "x": round(float(coords[i, 0]), 4),
                "y": round(float(coords[i, 1]), 4),
            }
            for i in range(len(items))
        ],
        "dims": req.dims,
        "count": len(items),
        "metrics": metrics,
    }


@app.post("/api/similarity")
def similarity(req: SimilarityRequest):
    if len(req.ids) < 2:
        raise HTTPException(400, "Need ≥2 IDs")
    conn = open_db()
    try:
        items = _load_by_ids(conn, req.ids)
    finally:
        conn.close()
    n = len(items)
    matrix = [
        [
            round(_cosine_sim(items[i]["embedding"], items[j]["embedding"]), 4)
            for j in range(n)
        ]
        for i in range(n)
    ]
    return {"texts": [it["text"] for it in items], "matrix": matrix}


@app.post("/api/matryoshka")
def matryoshka(req: MatryoshkaRequest):
    conn = open_db()
    try:
        items = _load_by_ids(conn, [req.id_a, req.id_b])
    finally:
        conn.close()
    if len(items) < 2:
        raise HTTPException(400, "One or both IDs not found")
    a = items[0]["embedding"].reshape(1, -1)
    b = items[1]["embedding"].reshape(1, -1)
    sims = []
    for d in MATRYOSHKA_DIMS:
        at = _truncate_normalize(a, d)[0]
        bt = _truncate_normalize(b, d)[0]
        sims.append(round(_cosine_sim(at, bt), 4))
    return {
        "text_a": items[0]["text"],
        "text_b": items[1]["text"],
        "dims": MATRYOSHKA_DIMS,
        "similarities": sims,
    }


@app.post("/api/pca")
def pca_endpoint(req: PcaRequest):
    """PCA over filtered slice. Returns per-item scores on top-K PCs + explained variance."""
    conn = open_db()
    try:
        items = (
            _load_by_ids(conn, req.ids)
            if req.ids
            else _load_filtered(conn, req.filters)
        )
    finally:
        conn.close()
    if len(items) < 3:
        raise HTTPException(400, f"Need ≥3 embeddings for PCA, have {len(items)}")
    if req.n_components < 1:
        raise HTTPException(400, "n_components must be ≥1")
    matrix = _truncate_normalize(np.stack([it["embedding"] for it in items]), req.dims)
    scores, evr = compute_pca(matrix, req.n_components)
    return {
        "dims": req.dims,
        "n_components": int(scores.shape[1]),
        "count": len(items),
        "explained_variance_ratio": [round(float(v), 6) for v in evr],
        "items": [
            {
                "id": items[i]["id"],
                "text": items[i]["text"],
                "tags": items[i]["tags"],
                "scores": [
                    round(float(scores[i, j]), 6) for j in range(scores.shape[1])
                ],
            }
            for i in range(len(items))
        ],
    }


@app.post("/api/arithmetic")
def arithmetic(req: ArithmeticRequest):
    conn = open_db()
    try:
        inputs = _load_by_ids(conn, [req.id_a, req.id_b, req.id_c])
        all_items = _load_by_ids(conn, [])
    finally:
        conn.close()
    if len(inputs) < 3:
        raise HTTPException(400, "One or more IDs not found")
    a, b, c = inputs[0]["embedding"], inputs[1]["embedding"], inputs[2]["embedding"]
    result = a - b + c
    result_norm = result / (np.linalg.norm(result) + 1e-10)
    input_ids = {req.id_a, req.id_b, req.id_c}
    neighbors = sorted(
        [
            {
                "id": it["id"],
                "text": it["text"],
                "similarity": round(_cosine_sim(result_norm, it["embedding"]), 4),
            }
            for it in all_items
            if it["id"] not in input_ids
        ],
        key=lambda x: -x["similarity"],
    )
    return {
        "text_a": inputs[0]["text"],
        "text_b": inputs[1]["text"],
        "text_c": inputs[2]["text"],
        "nearest": neighbors[:5],
    }


# ── Directions ────────────────────────────────────────────────────────────────


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-10:
        raise HTTPException(
            400, "Direction has near-zero norm — the two class means are identical"
        )
    return v / n


def _load_tag_vectors(
    conn: sqlite3.Connection, tag: str, values: list[str]
) -> dict[str, np.ndarray]:
    """Return {value: (n, 3072)} for each requested value of a tag column."""
    if tag not in TAG_COLUMNS:
        raise HTTPException(400, f"Unknown tag column: {tag}")
    ph = ",".join("?" * len(values))
    rows = conn.execute(
        f"SELECT {tag} AS v, embedding FROM embeddings WHERE {tag} IN ({ph})",
        values,
    ).fetchall()
    out: dict[str, list[np.ndarray]] = {v: [] for v in values}
    for r in rows:
        out[r["v"]].append(np.frombuffer(r["embedding"], dtype=np.float32).copy())
    return {
        v: (np.stack(arr) if arr else np.zeros((0, 3072), np.float32))
        for v, arr in out.items()
    }


def _direction_row_to_dict(r: sqlite3.Row) -> dict:
    return {
        "id": r["id"],
        "name": r["name"],
        "tag": r["tag"],
        "value_a": r["value_a"],
        "value_b": r["value_b"],
        "kind": r["kind"],
        "n_a": r["n_a"],
        "n_b": r["n_b"],
        "created_at": r["created_at"],
    }


def _load_direction_vec(
    conn: sqlite3.Connection, direction_id: int
) -> tuple[dict, np.ndarray]:
    r = conn.execute(
        "SELECT * FROM directions WHERE id = ?", (direction_id,)
    ).fetchone()
    if not r:
        raise HTTPException(404, f"Direction {direction_id} not found")
    vec = np.frombuffer(r["vector"], dtype=np.float32).copy()
    return _direction_row_to_dict(r), vec


def _per_dim_auc(X: np.ndarray, pos_mask: np.ndarray) -> np.ndarray:
    """Vectorised Mann-Whitney U → AUC per column.

    AUC=1 means dim perfectly separates (A > B), AUC=0 means the opposite.
    Frontend can display |auc - 0.5| for ranking regardless of sign.
    """
    n = X.shape[0]
    n_pos = int(pos_mask.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        raise HTTPException(400, "Both classes need at least one member")
    ranks = np.argsort(np.argsort(X, axis=0), axis=0) + 1
    rank_sum_pos = ranks[pos_mask].sum(axis=0)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


@app.post("/api/directions/create")
def create_direction(req: CreateDirectionRequest):
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "name cannot be empty")
    if req.value_a == req.value_b:
        raise HTTPException(400, "value_a and value_b must differ")
    conn = open_db()
    try:
        grouped = _load_tag_vectors(conn, req.tag, [req.value_a, req.value_b])
        a_mat = grouped[req.value_a]
        b_mat = grouped[req.value_b]
        if len(a_mat) == 0 or len(b_mat) == 0:
            raise HTTPException(
                400,
                f"No embeddings for {req.tag}={req.value_a} ({len(a_mat)}) "
                f"or {req.tag}={req.value_b} ({len(b_mat)})",
            )
        v = a_mat.mean(axis=0) - b_mat.mean(axis=0)
        v_hat = _unit(v).astype(np.float32)
        try:
            conn.execute(
                """
                INSERT INTO directions
                  (name, tag, value_a, value_b, vector, kind, n_a, n_b)
                VALUES (?, ?, ?, ?, ?, 'mean_diff', ?, ?)
                """,
                (
                    name,
                    req.tag,
                    req.value_a,
                    req.value_b,
                    v_hat.tobytes(),
                    len(a_mat),
                    len(b_mat),
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            raise HTTPException(409, f"Direction name already exists: {name}") from None
        row = conn.execute(
            "SELECT * FROM directions WHERE name = ?", (name,)
        ).fetchone()
        return _direction_row_to_dict(row)
    finally:
        conn.close()


@app.post("/api/directions/create-random")
def create_random_direction(req: CreateRandomDirectionRequest):
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "name cannot be empty")
    rng = np.random.default_rng(req.seed)
    v = rng.standard_normal(3072).astype(np.float32)
    v_hat = _unit(v).astype(np.float32)
    conn = open_db()
    try:
        try:
            conn.execute(
                """
                INSERT INTO directions
                  (name, tag, value_a, value_b, vector, kind, n_a, n_b)
                VALUES (?, NULL, NULL, NULL, ?, 'random', NULL, NULL)
                """,
                (name, v_hat.tobytes()),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            raise HTTPException(409, f"Direction name already exists: {name}") from None
        row = conn.execute(
            "SELECT * FROM directions WHERE name = ?", (name,)
        ).fetchone()
        return _direction_row_to_dict(row)
    finally:
        conn.close()


@app.get("/api/directions")
def list_directions():
    conn = open_db()
    try:
        rows = conn.execute(
            "SELECT * FROM directions ORDER BY created_at DESC"
        ).fetchall()
    finally:
        conn.close()
    dirs = [_direction_row_to_dict(r) for r in rows]
    vectors = [np.frombuffer(r["vector"], dtype=np.float32) for r in rows]
    n = len(vectors)
    cosines = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            cosines[i][j] = round(float(np.dot(vectors[i], vectors[j])), 4)
    return {"directions": dirs, "cosines": cosines}


@app.delete("/api/directions/{direction_id}")
def delete_direction(direction_id: int):
    conn = open_db()
    try:
        conn.execute("DELETE FROM directions WHERE id = ?", (direction_id,))
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


@app.post("/api/directions/{direction_id}/project")
def project_direction(direction_id: int, req: ProjectDirectionRequest):
    """Project every cached embedding (under filters) onto a stored direction.

    If `dims < 3072`, both the embeddings and the direction are Matryoshka-truncated
    and re-normalised before the dot product. Lets the frontend preview how the
    direction holds up at lower resolutions without creating a new one.
    """
    conn = open_db()
    try:
        d, v = _load_direction_vec(conn, direction_id)
        items = _load_filtered(conn, req.filters)
    finally:
        conn.close()
    if not items:
        return {"direction": d, "dims": int(req.dims), "points": []}
    matrix = np.stack([it["embedding"] for it in items])
    if req.dims < matrix.shape[1]:
        matrix = _truncate_normalize(matrix, req.dims)
        v = v[: req.dims]
        v = v / (np.linalg.norm(v) + 1e-10)
    scores = matrix @ v
    return {
        "direction": d,
        "dims": int(req.dims),
        "points": [
            {
                "id": items[i]["id"],
                "text": items[i]["text"],
                "tags": items[i]["tags"],
                "score": round(float(scores[i]), 6),
            }
            for i in range(len(items))
        ],
    }


@app.post("/api/directions/auc-scan")
def auc_scan(req: AucScanRequest):
    """Per-dimension AUC separating value_a (positive) vs value_b (negative)."""
    conn = open_db()
    try:
        grouped = _load_tag_vectors(conn, req.tag, [req.value_a, req.value_b])
    finally:
        conn.close()
    a_mat = grouped[req.value_a]
    b_mat = grouped[req.value_b]
    if len(a_mat) == 0 or len(b_mat) == 0:
        raise HTTPException(400, "Both classes need at least one member")
    X = np.vstack([a_mat, b_mat])
    pos = np.concatenate([np.ones(len(a_mat), bool), np.zeros(len(b_mat), bool)])
    auc = _per_dim_auc(X, pos)
    abs_signal = np.abs(auc - 0.5)
    order = np.argsort(-abs_signal)
    top = [
        {
            "dim": int(i),
            "auc": round(float(auc[i]), 4),
            "signal": round(float(abs_signal[i]), 4),
        }
        for i in order[:20]
    ]
    return {
        "tag": req.tag,
        "value_a": req.value_a,
        "value_b": req.value_b,
        "n_a": int(len(a_mat)),
        "n_b": int(len(b_mat)),
        "auc": [round(float(x), 4) for x in auc],
        "top": top,
        "max_abs_signal": round(float(abs_signal.max()), 4),
    }


@app.post("/api/directions/matryoshka")
def direction_matryoshka(req: DirectionMatryoshkaRequest):
    """Build the mean-diff direction at several Matryoshka dims and measure
    how cleanly it still separates (1D AUC of e·v̂ on the training classes).
    """
    dims = req.dims or [32, 64, 128, 256, 512, 1024, 2048, 3072]
    conn = open_db()
    try:
        grouped = _load_tag_vectors(conn, req.tag, [req.value_a, req.value_b])
    finally:
        conn.close()
    a_mat = grouped[req.value_a]
    b_mat = grouped[req.value_b]
    if len(a_mat) == 0 or len(b_mat) == 0:
        raise HTTPException(400, "Both classes need at least one member")
    full_dim = a_mat.shape[1]
    out = []
    for d in dims:
        d_eff = min(d, full_dim)
        at = _truncate_normalize(a_mat, d_eff)
        bt = _truncate_normalize(b_mat, d_eff)
        v = at.mean(axis=0) - bt.mean(axis=0)
        norm = float(np.linalg.norm(v))
        if norm < 1e-10:
            out.append({"dim": d_eff, "auc": 0.5, "mean_gap": 0.0})
            continue
        v_hat = v / norm
        scores_a = at @ v_hat
        scores_b = bt @ v_hat
        X = np.concatenate([scores_a, scores_b])[:, None]
        pos = np.concatenate(
            [np.ones(len(scores_a), bool), np.zeros(len(scores_b), bool)]
        )
        auc = float(_per_dim_auc(X, pos)[0])
        out.append(
            {
                "dim": d_eff,
                "auc": round(auc, 4),
                "mean_gap": round(float(scores_a.mean() - scores_b.mean()), 4),
            }
        )
    return {
        "tag": req.tag,
        "value_a": req.value_a,
        "value_b": req.value_b,
        "n_a": int(len(a_mat)),
        "n_b": int(len(b_mat)),
        "points": out,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    init_db()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
