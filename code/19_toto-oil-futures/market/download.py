"""Download orchestrator — reads config, calls data sources, writes to SQLite.

Usage:
    python -m data.download                    # uses config.yaml defaults
    python -m data.download --tickers CL=F GC=F  # override tickers
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from .db import get_db, init_db, insert_ohlcv
from .sources.yfinance_source import YFinanceSource

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Registry of available data sources
SOURCES = {
    "yfinance": YFinanceSource,
}


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def download(config: dict | None = None, tickers: list[str] | None = None):
    """Download OHLCV data for configured tickers and store in SQLite."""
    if config is None:
        config = load_config()

    data_cfg = config["data"]
    source_name = data_cfg["source"]
    start_date = data_cfg["start_date"]

    # Combine train + eval tickers (deduplicated)
    if tickers is None:
        tickers = list(
            dict.fromkeys(data_cfg["train_tickers"] + data_cfg["eval_tickers"])
        )

    logger.info("=== Download: %d tickers from %s ===", len(tickers), source_name)

    # Initialize DB
    init_db()

    # Get the data source
    source_cls = SOURCES.get(source_name)
    if source_cls is None:
        raise ValueError(
            f"Unknown source: {source_name}. Available: {list(SOURCES.keys())}"
        )
    source = source_cls()

    # Fetch data
    df = source.fetch(tickers, start_date)

    if df.empty:
        logger.warning("No data returned!")
        return

    # Write to DB
    with get_db() as conn:
        insert_ohlcv(conn, df, source=source_name)

    # Print summary
    logger.info("=== Download Summary ===")
    for ticker in sorted(df["ticker"].unique()):
        t_df = df[df["ticker"] == ticker]
        logger.info(
            "  %-8s %s to %s  (%d rows)",
            ticker,
            t_df["date"].min(),
            t_df["date"].max(),
            len(t_df),
        )
    logger.info("Total: %d rows for %d tickers", len(df), df["ticker"].nunique())


def main():
    parser = argparse.ArgumentParser(description="Download commodities data")
    parser.add_argument("--tickers", nargs="+", help="Override tickers from config")
    parser.add_argument("--start-date", help="Override start date")
    args = parser.parse_args()

    config = load_config()
    if args.start_date:
        config["data"]["start_date"] = args.start_date

    download(config, tickers=args.tickers)


if __name__ == "__main__":
    main()
