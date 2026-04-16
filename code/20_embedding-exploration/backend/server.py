"""
Embedding Explorer backend.

Single SQLite DB (embeddings.db) with optional tag columns:
  gender, class, length — nullable. Freeform items have NULL tags,
  seeded demo items have them populated.

Endpoints:
  POST   /api/embed               — embed text (cache-first via SQLite)
  POST   /api/cache/list          — list cached items, with optional filters
  GET    /api/tag-values          — distinct values per tag column
  DELETE /api/cache/{id}          — remove from cache
  POST   /api/vectors             — raw float32 vectors for given IDs
  POST   /api/umap                — 2D UMAP projection (dims arg for Matryoshka)
  POST   /api/similarity          — N×N cosine similarity matrix
  POST   /api/matryoshka          — cosine sim at [256,512,1024,2048,3072] dims
  POST   /api/arithmetic          — A − B + C, nearest neighbors in cache

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
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MATRYOSHKA_DIMS = [256, 512, 1024, 2048, 3072]
TAG_COLUMNS = ["gender", "class", "length"]

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


class ArithmeticRequest(BaseModel):
    id_a: int
    id_b: int
    id_c: int


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    init_db()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
