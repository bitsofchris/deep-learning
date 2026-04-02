"""Abstract base class for derived feature generators.

Each feature generator computes one or more derived features from the base
feature data in SQLite. New features are registered in registry.py and
stored in the derived_features table (EAV pattern).

To add a new feature:
  1. Create a new file in data/features/ (e.g., volatility.py)
  2. Subclass FeatureGenerator
  3. Register it in registry.py
  4. Add the feature name to config.yaml under exogenous_features
"""

from abc import ABC, abstractmethod

import pandas as pd


class FeatureGenerator(ABC):
    """Interface for derived feature computation.

    Implementations compute one named feature from base feature data.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this feature (used in DB and config)."""
        ...

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute the feature from base feature data.

        Args:
            df: DataFrame with columns: ticker, date, close, returns,
                log_returns, normalized, volume. Sorted by date.
                Contains data for a SINGLE ticker.

        Returns:
            Series of float values, same index as df.
        """
        ...
