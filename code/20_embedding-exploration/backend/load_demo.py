"""
Bulk-embed a JSON file of labeled sentences into the demo database.

Usage:
  export OPENAI_API_KEY=...
  python3 load_demo.py ../demo_dataset.json

JSON format:
  [{"text": "...", "gender": "male", "class": "royal", "length": "short"}, ...]
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DB_PATH = HERE / "embeddings.db"
TAG_COLUMNS = ["gender", "class", "length"]
BATCH_SIZE = 100


def init_db(conn: sqlite3.Connection) -> None:
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
    existing_cols = {
        r[1] for r in conn.execute("PRAGMA table_info(embeddings)").fetchall()
    }
    for col in TAG_COLUMNS:
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE embeddings ADD COLUMN {col} TEXT")
    conn.commit()


def embed_batch(texts: list[str], api_key: str) -> list[list[float]]:
    import openai

    client = openai.OpenAI(api_key=api_key)
    resp = client.embeddings.create(model="text-embedding-3-large", input=texts)
    return [d.embedding for d in resp.data]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="Path to labeled dataset JSON")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    data = json.loads(Path(args.json_path).read_text())
    print(f"Loaded {len(data)} items from {args.json_path}")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    init_db(conn)

    existing = {
        r["text"] for r in conn.execute("SELECT text FROM embeddings").fetchall()
    }
    to_embed = [d for d in data if d["text"] not in existing]
    skipped = len(data) - len(to_embed)
    if skipped:
        print(f"Skipping {skipped} already-embedded items")

    if not to_embed:
        print("Nothing to do.")
        conn.close()
        return

    total_inserted = 0
    for i in range(0, len(to_embed), args.batch_size):
        batch = to_embed[i : i + args.batch_size]
        texts = [item["text"] for item in batch]
        t0 = time.time()
        print(
            f"Embedding batch {i // args.batch_size + 1} " f"({len(batch)} items)…",
            end=" ",
            flush=True,
        )
        try:
            embeddings = embed_batch(texts, api_key)
        except Exception as e:
            print(f"\nFAILED on batch at index {i}: {e}")
            sys.exit(1)

        for item, vec in zip(batch, embeddings, strict=True):
            arr = np.array(vec, dtype=np.float32)
            col_names = ["text", "embedding"] + [c for c in TAG_COLUMNS if item.get(c)]
            col_values = [item["text"], arr.tobytes()] + [
                item[c] for c in TAG_COLUMNS if item.get(c)
            ]
            placeholders = ",".join("?" * len(col_values))
            conn.execute(
                f"INSERT OR IGNORE INTO embeddings ({', '.join(col_names)}) VALUES ({placeholders})",
                col_values,
            )
            total_inserted += 1
        conn.commit()
        print(f"{time.time() - t0:.1f}s")

    total = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    print(f"\nDone. Inserted {total_inserted} new items. DB total: {total}")
    conn.close()


if __name__ == "__main__":
    main()
