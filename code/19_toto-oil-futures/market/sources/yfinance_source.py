"""yfinance data source — downloads OHLCV from Yahoo Finance."""

import logging

import pandas as pd
import yfinance as yf

from .base import DataSource

logger = logging.getLogger(__name__)


class YFinanceSource(DataSource):
    def fetch(
        self, tickers: list[str], start_date: str, end_date: str | None = None
    ) -> pd.DataFrame:
        logger.info(
            "Downloading %d tickers from yfinance (%s to %s)",
            len(tickers),
            start_date,
            end_date or "latest",
        )

        frames = []
        failed = []

        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                kwargs = {"start": start_date}
                if end_date:
                    kwargs["end"] = end_date

                hist = t.history(**kwargs)

                if hist.empty:
                    logger.warning("No data returned for %s", ticker)
                    failed.append(ticker)
                    continue

                df = hist.reset_index()
                df = df.rename(
                    columns={
                        "Date": "date",
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    }
                )
                df["ticker"] = ticker
                # Normalize date to ISO string (strip timezone if present)
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                df = df[["ticker", "date", "open", "high", "low", "close", "volume"]]

                frames.append(df)
                logger.info(
                    "  %s: %d rows (%s to %s)",
                    ticker,
                    len(df),
                    df["date"].iloc[0],
                    df["date"].iloc[-1],
                )

            except Exception as e:
                logger.error("Failed to download %s: %s", ticker, e)
                failed.append(ticker)

        if failed:
            logger.warning("Failed tickers: %s", failed)

        if not frames:
            return pd.DataFrame(
                columns=["ticker", "date", "open", "high", "low", "close", "volume"]
            )

        result = pd.concat(frames, ignore_index=True)
        logger.info("Total: %d rows for %d tickers", len(result), len(frames))
        return result
