"""SQLite database module — single source of truth for all project data.

All tables, connections, and query helpers live here. Every other module
reads from and writes to the DB through this interface.
"""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parent / "commodities.db"

SCHEMA = """
-- Raw OHLCV data from any source
CREATE TABLE IF NOT EXISTS raw_ohlcv (
    ticker      TEXT NOT NULL,
    date        TEXT NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      REAL,
    source      TEXT NOT NULL,
    fetched_at  TEXT NOT NULL,
    PRIMARY KEY (ticker, date, source)
);

-- Base computed features
CREATE TABLE IF NOT EXISTS features (
    ticker          TEXT NOT NULL,
    date            TEXT NOT NULL,
    close           REAL,
    returns         REAL,
    log_returns     REAL,
    vol_adj_returns REAL,
    normalized      REAL,
    volume          REAL,
    PRIMARY KEY (ticker, date)
);

-- Extensible derived features (EAV pattern)
CREATE TABLE IF NOT EXISTS derived_features (
    ticker          TEXT NOT NULL,
    date            TEXT NOT NULL,
    feature_name    TEXT NOT NULL,
    value           REAL,
    computed_at     TEXT NOT NULL,
    PRIMARY KEY (ticker, date, feature_name)
);

-- CV fold definitions (time-based, all tickers share boundaries)
CREATE TABLE IF NOT EXISTS cv_folds (
    split_id        TEXT NOT NULL,
    fold_number     INTEGER NOT NULL,
    split           TEXT NOT NULL,
    start_date      TEXT NOT NULL,
    end_date        TEXT NOT NULL,
    n_days          INTEGER,
    embargo_start   TEXT,
    embargo_end     TEXT,
    PRIMARY KEY (split_id, fold_number, split)
);

-- Experiment tracking
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id   TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    config_json     TEXT NOT NULL,
    cv_fold         INTEGER,
    status          TEXT NOT NULL,
    checkpoint_path TEXT,
    train_loss      REAL,
    val_loss        REAL,
    notes           TEXT
);

-- Forecast outputs
CREATE TABLE IF NOT EXISTS forecasts (
    experiment_id   TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    forecast_date   TEXT NOT NULL,
    target_date     TEXT NOT NULL,
    horizon         INTEGER NOT NULL,
    predicted       REAL,
    predicted_q05   REAL,
    predicted_q95   REAL,
    actual          REAL,
    PRIMARY KEY (experiment_id, ticker, forecast_date, target_date)
);

-- Evaluation metrics
CREATE TABLE IF NOT EXISTS evaluations (
    experiment_id       TEXT NOT NULL,
    ticker              TEXT NOT NULL,
    cv_fold             INTEGER NOT NULL,
    mae                 REAL,
    rmse                REAL,
    directional_acc     REAL,
    sharpe              REAL,
    n_predictions       INTEGER,
    PRIMARY KEY (experiment_id, ticker, cv_fold)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_raw_ohlcv_ticker ON raw_ohlcv(ticker);
CREATE INDEX IF NOT EXISTS idx_features_ticker ON features(ticker);
CREATE INDEX IF NOT EXISTS idx_features_date ON features(date);
CREATE INDEX IF NOT EXISTS idx_derived_features_name ON derived_features(feature_name);
CREATE INDEX IF NOT EXISTS idx_forecasts_experiment ON forecasts(experiment_id);
"""


def get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Get a SQLite connection with WAL mode for concurrent reads."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db(db_path: Path = DEFAULT_DB_PATH):
    """Context manager for DB connections. Auto-commits on success, rolls back on error."""
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path = DEFAULT_DB_PATH):
    """Create all tables if they don't exist."""
    with get_db(db_path) as conn:
        conn.executescript(SCHEMA)
    logger.info("Database initialized at %s", db_path)


def insert_ohlcv(conn: sqlite3.Connection, df: pd.DataFrame, source: str):
    """Insert OHLCV data. Skips rows that already exist (idempotent)."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            row.ticker,
            row.date,
            row.open,
            row.high,
            row.low,
            row.close,
            row.volume,
            source,
            now,
        )
        for row in df.itertuples(index=False)
    ]
    conn.executemany(
        """INSERT OR IGNORE INTO raw_ohlcv
           (ticker, date, open, high, low, close, volume, source, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    logger.info(
        "Inserted %d rows from source '%s' (duplicates skipped)", len(rows), source
    )


def insert_features(conn: sqlite3.Connection, df: pd.DataFrame):
    """Insert computed features. Replaces existing rows."""
    rows = [
        (
            row.ticker,
            row.date,
            row.close,
            row.returns,
            row.log_returns,
            row.vol_adj_returns,
            row.normalized,
            row.volume,
        )
        for row in df.itertuples(index=False)
    ]
    conn.executemany(
        """INSERT OR REPLACE INTO features
           (ticker, date, close, returns, log_returns, vol_adj_returns, normalized, volume)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    logger.info("Inserted %d feature rows", len(rows))


def insert_cv_folds(conn: sqlite3.Connection, folds: list[dict]):
    """Insert CV fold definitions. Replaces existing for same split_id."""
    for f in folds:
        conn.execute(
            """INSERT OR REPLACE INTO cv_folds
               (split_id, fold_number, split, start_date, end_date, n_days, embargo_start, embargo_end)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                f["split_id"],
                f["fold_number"],
                f["split"],
                f["start_date"],
                f["end_date"],
                f["n_days"],
                f.get("embargo_start"),
                f.get("embargo_end"),
            ),
        )
    logger.info("Inserted %d CV fold records", len(folds))


def query_df(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> pd.DataFrame:
    """Run a query and return results as a DataFrame."""
    return pd.read_sql_query(sql, conn, params=params)


def get_tickers(conn: sqlite3.Connection) -> list[str]:
    """Get all tickers that have raw data."""
    rows = conn.execute(
        "SELECT DISTINCT ticker FROM raw_ohlcv ORDER BY ticker"
    ).fetchall()
    return [r[0] for r in rows]


def get_feature_data(
    conn: sqlite3.Connection,
    tickers: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    target_col: str = "returns",
) -> pd.DataFrame:
    """Get feature data for specified tickers and date range."""
    placeholders = ",".join("?" * len(tickers))
    sql = f"SELECT * FROM features WHERE ticker IN ({placeholders})"
    params = list(tickers)

    if start_date:
        sql += " AND date >= ?"
        params.append(start_date)
    if end_date:
        sql += " AND date <= ?"
        params.append(end_date)

    sql += " ORDER BY ticker, date"
    return pd.read_sql_query(sql, conn, params=params)


def get_cv_folds(conn: sqlite3.Connection, split_id: str) -> list[dict]:
    """Get CV fold definitions for a given split config."""
    rows = conn.execute(
        "SELECT * FROM cv_folds WHERE split_id = ? ORDER BY fold_number, split",
        (split_id,),
    ).fetchall()
    return [dict(r) for r in rows]
