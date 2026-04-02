"""Abstract base class for data sources.

Every data source (yfinance, FRED, CSV, etc.) implements this interface.
The download orchestrator calls fetch() and gets back a standardized DataFrame.
"""

from abc import ABC, abstractmethod

import pandas as pd


class DataSource(ABC):
    """Interface for pluggable data sources.

    Implementations must return a DataFrame with columns:
        ticker, date, open, high, low, close, volume

    - date should be ISO 8601 string (YYYY-MM-DD)
    - numeric columns should be float (NaN for missing)
    """

    @abstractmethod
    def fetch(
        self, tickers: list[str], start_date: str, end_date: str | None = None
    ) -> pd.DataFrame:
        """Download OHLCV data for the given tickers and date range.

        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), None for latest available

        Returns:
            DataFrame with columns: ticker, date, open, high, low, close, volume
        """
        ...
