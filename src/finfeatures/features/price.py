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


class LogTransform(Feature):
    """
    Log-transformed OHLCV columns.

    Produces log_open, log_high, log_low, log_close, log_volume.
    Many regime-detection features operate in log-price space since
    log returns are additive and more naturally normally distributed.
    """

    name = "log_transform"
    required_cols = [Columns.OPEN, Columns.HIGH, Columns.LOW, Columns.CLOSE, Columns.VOLUME]
    description = "Log-transformed OHLCV columns"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["log_open"] = np.log(df[Columns.OPEN])
        out["log_high"] = np.log(df[Columns.HIGH])
        out["log_low"] = np.log(df[Columns.LOW])
        out["log_close"] = np.log(df[Columns.CLOSE])
        out["log_volume"] = np.log1p(df[Columns.VOLUME])
        return out


class CandleShape(Feature):
    """
    Candle shape features in log-price space.

    Requires LogTransform to have run first.

      - body:        log_close - log_open  (positive = bullish)
      - range:       log_high - log_low    (total candle extent)
      - upper_wick:  log_high - log_close
      - lower_wick:  log_close - log_low
      - CLV:         (log_close - log_low) / (log_high - log_low)  (Close Location Value)
    """

    name = "candle_shape"
    required_cols = ["log_open", "log_high", "log_low", "log_close"]
    description = "Candle body, wicks, and close location value in log space"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        log_o = df["log_open"]
        log_h = df["log_high"]
        log_l = df["log_low"]
        log_c = df["log_close"]

        out["body"] = log_c - log_o
        hl_range = log_h - log_l
        out["range"] = hl_range
        out["upper_wick"] = log_h - log_c
        out["lower_wick"] = log_c - log_l
        out["CLV"] = (log_c - log_l) / hl_range.replace(0, np.nan)
        return out


class CrossDay(Feature):
    """
    Cross-day relationship features in log-price space.

    Measures today's close/open relative to yesterday's high/low.
    Requires LogTransform to have run first.
    """

    name = "cross_day"
    required_cols = ["log_open", "log_high", "log_low", "log_close"]
    description = "Cross-day high/low relationships in log space"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        log_o = df["log_open"]
        log_h = df["log_high"]
        log_l = df["log_low"]
        log_c = df["log_close"]

        out["overnight"] = log_o - log_c.shift(1)
        out["C_minus_yH"] = log_c - log_h.shift(1)
        out["C_minus_yL"] = log_c - log_l.shift(1)
        out["O_minus_yH"] = log_o - log_h.shift(1)
        out["O_minus_yL"] = log_o - log_l.shift(1)
        return out


class ShapeDynamics(Feature):
    """
    First differences of candle shape features and log volume.

    Captures how the microstructure of price bars is changing day-to-day.
    Requires LogTransform and CandleShape to have run first.
    """

    name = "shape_dynamics"
    required_cols = ["body", "range", "upper_wick", "lower_wick", "CLV", "log_volume"]
    description = "First differences of candle shape features"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["d_body"] = df["body"].diff()
        out["d_range"] = df["range"].diff()
        out["d_upper_wick"] = df["upper_wick"].diff()
        out["d_lower_wick"] = df["lower_wick"].diff()
        out["d_CLV"] = df["CLV"].diff()
        out["d_log_vol"] = df["log_volume"].diff()
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
