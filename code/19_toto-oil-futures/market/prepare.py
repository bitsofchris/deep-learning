"""Data preparation — compute features, generate CV folds, build Toto datasets.

Reads raw OHLCV from SQLite, computes base features (returns, log_returns, etc.),
generates purged k-fold CV boundaries with embargo, and provides a function to
build Toto-compatible HuggingFace Dataset dicts for any fold.

Usage:
    python -m data.prepare                # compute features + CV folds
    python -m data.prepare --info         # print summary of what's in the DB
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .db import (
    get_db,
    get_cv_folds,
    get_feature_data,
    insert_cv_folds,
    insert_features,
    query_df,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------


def compute_features(conn) -> pd.DataFrame:
    """Compute base features from raw OHLCV data.

    For each ticker: returns, log_returns, rolling z-score normalized close.
    """
    raw = query_df(conn, "SELECT * FROM raw_ohlcv ORDER BY ticker, date")

    if raw.empty:
        logger.warning("No raw data in DB. Run download first.")
        return pd.DataFrame()

    frames = []
    for ticker, group in raw.groupby("ticker"):
        g = group.sort_values("date").copy()

        g["returns"] = g["close"].pct_change()
        g["log_returns"] = np.log(g["close"] / g["close"].shift(1))

        # Volatility-adjusted returns (de Prado): returns / rolling_std(returns)
        # 20-day rolling vol is standard (~1 trading month)
        rolling_vol = g["returns"].rolling(window=20, min_periods=10).std()
        g["vol_adj_returns"] = g["returns"] / rolling_vol.replace(0, np.nan)

        # Rolling z-score normalization (60-day window)
        rolling_mean = g["close"].rolling(window=60, min_periods=10).mean()
        rolling_std = g["close"].rolling(window=60, min_periods=10).std()
        g["normalized"] = (g["close"] - rolling_mean) / rolling_std.replace(0, np.nan)

        g["ticker"] = ticker
        frames.append(
            g[
                [
                    "ticker",
                    "date",
                    "close",
                    "returns",
                    "log_returns",
                    "vol_adj_returns",
                    "normalized",
                    "volume",
                ]
            ].dropna(subset=["returns"])
        )

    result = pd.concat(frames, ignore_index=True)
    logger.info(
        "Computed features for %d tickers (%d rows)",
        result["ticker"].nunique(),
        len(result),
    )
    return result


# ---------------------------------------------------------------------------
# Purged K-Fold CV
# ---------------------------------------------------------------------------


def generate_purged_kfold(
    conn,
    n_folds: int = 3,
    embargo_pct: float = 0.01,
    purge_days: int = 5,
) -> tuple[str, list[dict]]:
    """Generate purged k-fold CV boundaries (de Prado style).

    Folds are TIME-BASED — all tickers share the same temporal boundaries.
    Purge removes training data whose label window overlaps test boundaries.
    Embargo adds a buffer after each test fold to handle autocorrelation.

    Returns:
        (split_id, list of fold dicts for DB insertion)
    """
    # Get the full date range from features (all tickers share this timeline)
    dates_df = query_df(conn, "SELECT DISTINCT date FROM features ORDER BY date")
    dates = dates_df["date"].tolist()
    n_dates = len(dates)

    if n_dates == 0:
        raise ValueError("No feature data. Run compute_features first.")

    embargo_size = max(1, int(n_dates * embargo_pct))

    # Create a deterministic split_id from the config
    config_hash = hashlib.md5(
        json.dumps(
            {
                "n_folds": n_folds,
                "embargo_pct": embargo_pct,
                "purge_days": purge_days,
                "n_dates": n_dates,
            }
        ).encode()
    ).hexdigest()[:12]
    split_id = f"pkf_{n_folds}f_{config_hash}"

    # Divide dates into n_folds contiguous groups
    fold_size = n_dates // n_folds
    fold_records = []

    logger.info(
        "Generating %d-fold purged CV: %d dates, embargo=%d days, purge=%d days",
        n_folds,
        n_dates,
        embargo_size,
        purge_days,
    )

    for fold_idx in range(n_folds):
        # Test fold boundaries
        test_start_idx = fold_idx * fold_size
        test_end_idx = (
            (fold_idx + 1) * fold_size - 1 if fold_idx < n_folds - 1 else n_dates - 1
        )
        test_start = dates[test_start_idx]
        test_end = dates[test_end_idx]

        # Embargo after test fold
        embargo_end_idx = min(test_end_idx + embargo_size, n_dates - 1)
        embargo_start = dates[test_end_idx + 1] if test_end_idx + 1 < n_dates else None
        embargo_end = dates[embargo_end_idx] if embargo_start else None

        # Test fold record
        fold_records.append(
            {
                "split_id": split_id,
                "fold_number": fold_idx,
                "split": "test",
                "start_date": test_start,
                "end_date": test_end,
                "n_days": test_end_idx - test_start_idx + 1,
                "embargo_start": embargo_start,
                "embargo_end": embargo_end,
            }
        )

        # Train = everything NOT in (test + purge + embargo)
        # Purge: remove purge_days before test_start and after test_end
        purge_start_idx = max(0, test_start_idx - purge_days)
        purge_end_idx = min(n_dates - 1, test_end_idx + purge_days)
        excluded_end_idx = min(n_dates - 1, max(purge_end_idx, embargo_end_idx))

        # Train segments: before excluded zone and after excluded zone
        train_dates = []
        if purge_start_idx > 0:
            train_dates.extend(dates[:purge_start_idx])
        if excluded_end_idx + 1 < n_dates:
            train_dates.extend(dates[excluded_end_idx + 1 :])

        if train_dates:
            fold_records.append(
                {
                    "split_id": split_id,
                    "fold_number": fold_idx,
                    "split": "train",
                    "start_date": train_dates[0],
                    "end_date": train_dates[-1],
                    "n_days": len(train_dates),
                    "embargo_start": None,
                    "embargo_end": None,
                }
            )

        logger.info(
            "  Fold %d: test=[%s, %s] (%d days), embargo=[%s, %s], train=%d days",
            fold_idx,
            test_start,
            test_end,
            test_end_idx - test_start_idx + 1,
            embargo_start or "N/A",
            embargo_end or "N/A",
            len(train_dates),
        )

    return split_id, fold_records


# ---------------------------------------------------------------------------
# Toto dataset builder
# ---------------------------------------------------------------------------


def get_fold_dates(conn, split_id: str, fold_number: int) -> dict[str, tuple[str, str]]:
    """Get train/test date ranges for a specific fold.

    Returns dict with 'train' and 'test' keys, each mapping to (start_date, end_date).
    Also returns excluded dates (purge + embargo zones) for proper filtering.
    """
    folds = get_cv_folds(conn, split_id)
    result = {}
    for f in folds:
        if f["fold_number"] == fold_number:
            result[f["split"]] = (f["start_date"], f["end_date"])
    return result


def build_toto_dataset(
    conn,
    tickers: list[str],
    start_date: str,
    end_date: str,
    target_col: str = "returns",
) -> list[dict]:
    """Build a list of series dicts from DB data, ready for HF Dataset conversion.

    Each dict represents one time series (one ticker) with:
        - timestamp: list of date strings
        - target: list of float values (the target column)
        - volume: list of float values (potential exogenous variable)

    This is the intermediate format before conversion to HF Dataset / Toto's
    custom_dataset dict. The actual HF conversion happens in model/finetune.py
    since it depends on Toto being installed.
    """
    df = get_feature_data(conn, tickers, start_date, end_date, target_col)

    if df.empty:
        logger.warning(
            "No feature data for tickers=%s in [%s, %s]", tickers, start_date, end_date
        )
        return []

    series_list = []
    for ticker, group in df.groupby("ticker"):
        g = group.sort_values("date")
        series_list.append(
            {
                "ticker": ticker,
                "timestamp": g["date"].tolist(),
                target_col: g[target_col].tolist(),
                "volume": g["volume"].tolist(),
                "close": g["close"].tolist(),
            }
        )
        logger.info(
            "  %s: %d points (%s to %s)",
            ticker,
            len(g),
            g["date"].iloc[0],
            g["date"].iloc[-1],
        )

    return series_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def prepare(config: dict | None = None):
    """Full preparation pipeline: compute features → generate CV folds."""
    if config is None:
        config = load_config()

    cv_cfg = config["cv"]

    with get_db() as conn:
        # Step 1: Compute features
        logger.info("=== Computing features ===")
        features_df = compute_features(conn)
        if features_df.empty:
            return
        insert_features(conn, features_df)

        # Step 2: Generate CV folds
        logger.info("=== Generating CV folds ===")
        split_id, fold_records = generate_purged_kfold(
            conn,
            n_folds=cv_cfg["n_folds"],
            embargo_pct=cv_cfg["embargo_pct"],
            purge_days=cv_cfg["purge_days"],
        )
        insert_cv_folds(conn, fold_records)

        # Step 3: Summary
        logger.info("=== Preparation Summary ===")
        ticker_count = conn.execute(
            "SELECT COUNT(DISTINCT ticker) FROM features"
        ).fetchone()[0]
        row_count = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
        fold_count = conn.execute(
            "SELECT COUNT(DISTINCT fold_number) FROM cv_folds WHERE split_id = ?",
            (split_id,),
        ).fetchone()[0]
        logger.info("  Features: %d rows for %d tickers", row_count, ticker_count)
        logger.info("  CV folds: %d folds (split_id=%s)", fold_count, split_id)
        logger.info("  Target column: %s", config["data"]["target"])

        return split_id


def print_info():
    """Print summary of what's in the DB."""
    with get_db() as conn:
        # Raw data
        raw = query_df(
            conn,
            "SELECT ticker, MIN(date) as first, MAX(date) as last, COUNT(*) as n FROM raw_ohlcv GROUP BY ticker ORDER BY ticker",
        )
        print("\n=== Raw OHLCV ===")
        if raw.empty:
            print("  (empty — run download first)")
        else:
            for _, row in raw.iterrows():
                print(
                    f"  {row['ticker']:8s} {row['first']} to {row['last']}  ({row['n']} rows)"
                )

        # Features
        feat = query_df(
            conn,
            "SELECT ticker, MIN(date) as first, MAX(date) as last, COUNT(*) as n FROM features GROUP BY ticker ORDER BY ticker",
        )
        print("\n=== Features ===")
        if feat.empty:
            print("  (empty — run prepare first)")
        else:
            for _, row in feat.iterrows():
                print(
                    f"  {row['ticker']:8s} {row['first']} to {row['last']}  ({row['n']} rows)"
                )

        # CV folds
        folds = query_df(
            conn,
            "SELECT split_id, fold_number, split, start_date, end_date, n_days FROM cv_folds ORDER BY split_id, fold_number, split",
        )
        print("\n=== CV Folds ===")
        if folds.empty:
            print("  (empty — run prepare first)")
        else:
            for _, row in folds.iterrows():
                print(
                    f"  [{row['split_id']}] fold {row['fold_number']} {row['split']:5s}: "
                    f"{row['start_date']} to {row['end_date']} ({row['n_days']} days)"
                )


def main():
    parser = argparse.ArgumentParser(description="Prepare features and CV folds")
    parser.add_argument("--info", action="store_true", help="Print DB summary")
    args = parser.parse_args()

    if args.info:
        print_info()
    else:
        prepare()


if __name__ == "__main__":
    main()
