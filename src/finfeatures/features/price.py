"""
Price-based features.

These are the fundamental first-order transforms of raw OHLCV data.
Everything else typically depends on one of these.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finfeatures.core.base import Columns, Feature


class Returns(Feature):
    """Simple (arithmetic) close-to-close returns: (P_t / P_{t-1}) - 1."""

    name = "returns"
    required_cols = [Columns.CLOSE]
    description = "Simple close-to-close returns"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[Columns.RETURN] = df[Columns.CLOSE].pct_change()
        return out


class LogReturns(Feature):
    """Log returns: ln(P_t / P_{t-1})."""

    name = "log_returns"
    required_cols = [Columns.CLOSE]
    description = "Log (continuously compounded) close-to-close returns"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[Columns.LOG_RETURN] = np.log(df[Columns.CLOSE] / df[Columns.CLOSE].shift(1))
        return out


class PriceRange(Feature):
    """
    Intraday range features:
      - high_low_range:      (high - low) / close
      - open_close_range:    (close - open) / open
      - overnight_gap:       (open_t - close_{t-1}) / close_{t-1}
    """

    name = "price_range"
    required_cols = [Columns.OPEN, Columns.HIGH, Columns.LOW, Columns.CLOSE]
    description = "Intraday and overnight range metrics"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["high_low_range"] = (df[Columns.HIGH] - df[Columns.LOW]) / df[Columns.CLOSE]
        out["open_close_range"] = (df[Columns.CLOSE] - df[Columns.OPEN]) / df[Columns.OPEN]
        out["overnight_gap"] = (df[Columns.OPEN] - df[Columns.CLOSE].shift(1)) / df[
            Columns.CLOSE
        ].shift(1)
        return out


class TypicalPrice(Feature):
    """
    Typical price: (H + L + C) / 3.
    Used as the basis for VWAP-like indicators.
    """

    name = "typical_price"
    required_cols = [Columns.HIGH, Columns.LOW, Columns.CLOSE]
    description = "Typical price (H + L + C) / 3"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["typical_price"] = (df[Columns.HIGH] + df[Columns.LOW] + df[Columns.CLOSE]) / 3.0
        return out


class CumulativeReturn(Feature):
    """
    Cumulative return from the first row: P_t / P_0 - 1.
    Useful for visualisation and drawdown computation.
    """

    name = "cumulative_return"
    required_cols = [Columns.CLOSE]
    description = "Cumulative return from the start of the series"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["cumulative_return"] = df[Columns.CLOSE] / df[Columns.CLOSE].iloc[0] - 1
        return out


class PriceRelativeToHigh(Feature):
    """
    Rolling distance from the rolling n-period high/low.
      - pct_from_high_N:  (close - rolling_high) / rolling_high
      - pct_from_low_N:   (close - rolling_low)  / rolling_low
    """

    name = "price_relative_to_high_low"
    required_cols = [Columns.CLOSE]
    description = "Rolling distance from rolling high and low"

    def __init__(self, window: int = 52) -> None:
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        w = self.window
        rolling_high = df[Columns.CLOSE].rolling(w).max()
        rolling_low = df[Columns.CLOSE].rolling(w).min()
        out[f"pct_from_high_{w}"] = (df[Columns.CLOSE] - rolling_high) / rolling_high
        out[f"pct_from_low_{w}"] = (df[Columns.CLOSE] - rolling_low) / rolling_low
        return out
