"""Merge experiment results from remote DBs into the local DB.

When running experiments in parallel on separate RunPod pods, each pod
writes to its own copy of the DB. This script merges the experiment,
forecast, and evaluation rows from one or more remote DBs into the local one.

Only merges results tables (experiments, forecasts, evaluations).
Does NOT touch raw_ohlcv, features, or cv_folds — those are identical
across all copies since they were uploaded from the same local DB.

Usage:
    python -m eval.merge_results results/pod1.db results/pod2.db
    python -m eval.merge_results results/*.db
"""

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Tables to merge and their primary key columns (for conflict resolution)
MERGE_TABLES = {
    "experiments": "experiment_id",
    "forecasts": "(experiment_id, ticker, forecast_date, target_date)",
    "evaluations": "(experiment_id, ticker, cv_fold)",
}


def merge_db(local_db_path: Path, remote_db_path: Path):
    """Merge results from a remote DB into the local DB."""
    if not remote_db_path.exists():
        logger.error("Remote DB not found: %s", remote_db_path)
        return

    logger.info("Merging from %s", remote_db_path)

    # Record local counts before merge for safety reporting
    local_conn = sqlite3.connect(str(local_db_path))
    before_counts = {}
    for table in MERGE_TABLES:
        before_counts[table] = local_conn.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]

    # Attach the remote DB (use DELETE journal mode to avoid WAL lock conflicts)
    local_conn.execute("ATTACH DATABASE ? AS remote", (str(remote_db_path),))

    total_merged = 0

    for table in MERGE_TABLES:
        remote_count = local_conn.execute(
            f"SELECT COUNT(*) FROM remote.{table}"
        ).fetchone()[0]
        if remote_count == 0:
            logger.info("  %s: 0 remote rows (skip)", table)
            continue

        # INSERT OR IGNORE — only add rows with new primary keys, never overwrite existing
        local_conn.execute(
            f"INSERT OR IGNORE INTO {table} SELECT * FROM remote.{table}"
        )

        after_count = local_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        new_rows = after_count - before_counts[table]
        logger.info(
            "  %s: +%d new rows (was %d, now %d, %d remote total)",
            table,
            new_rows,
            before_counts[table],
            after_count,
            remote_count,
        )
        total_merged += new_rows

    local_conn.commit()
    local_conn.execute("DETACH DATABASE remote")
    local_conn.close()

    logger.info("  Total: %d new rows merged (no existing rows modified)", total_merged)


def main():
    parser = argparse.ArgumentParser(
        description="Merge experiment results from remote DBs"
    )
    parser.add_argument("dbs", nargs="+", type=Path, help="Remote DB files to merge")
    parser.add_argument(
        "--local-db",
        type=Path,
        default=PROJECT_ROOT / "market" / "commodities.db",
        help="Local DB to merge into",
    )
    args = parser.parse_args()

    logger.info("Local DB: %s", args.local_db)

    for remote_path in args.dbs:
        merge_db(args.local_db, remote_path)

    # Print summary of what's in the local DB now
    conn = sqlite3.connect(str(args.local_db))
    n_exp = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    n_fc = conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
    n_ev = conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
    conn.close()

    logger.info(
        "=== Local DB now has: %d experiments, %d forecasts, %d evaluations ===",
        n_exp,
        n_fc,
        n_ev,
    )


if __name__ == "__main__":
    main()
