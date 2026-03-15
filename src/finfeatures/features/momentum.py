"""
Momentum features.

RSI, Rate of Change, Stochastic Oscillator, Williams %R, and CCI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finfeatures.core.base import Columns, Feature


class RSI(Feature):
    """
    Relative Strength Index (Wilder, 1978).
    RSI ranges 0-100.  Overbought > 70, oversold < 30.
    """

    name = "rsi"
    required_cols = [Columns.CLOSE]
    description = "Relative Strength Index"

    def __init__(self, window: int = 14) -> None:
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        delta = df[Columns.CLOSE].diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / self.window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / self.window, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        out[f"rsi_{self.window}"] = 100 - (100 / (1 + rs))
        return out


class RateOfChange(Feature):
    """
    Rate of Change: (P_t / P_{t-n}) - 1.
    Measures momentum over n periods.
    """

    name = "rate_of_change"
    required_cols = [Columns.CLOSE]
    description = "N-period Rate of Change (momentum)"

    def __init__(self, window: int = 10) -> None:
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[f"roc_{self.window}"] = (
            df[Columns.CLOSE] / df[Columns.CLOSE].shift(self.window) - 1
        )
        return out


class StochasticOscillator(Feature):
    """
    Stochastic Oscillator (%K and %D).
    %K = (Close - LowestLow) / (HighestHigh - LowestLow) * 100
    %D = SMA(%K, d_window)
    """

    name = "stochastic"
    required_cols = [Columns.HIGH, Columns.LOW, Columns.CLOSE]
    description = "Stochastic Oscillator %K and %D"

    def __init__(self, k_window: int = 14, d_window: int = 3) -> None:
        self.k_window = k_window
        self.d_window = d_window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        low_min  = df[Columns.LOW].rolling(self.k_window).min()
        high_max = df[Columns.HIGH].rolling(self.k_window).max()
        pct_k = 100 * (df[Columns.CLOSE] - low_min) / (high_max - low_min)
        out[f"stoch_k_{self.k_window}"] = pct_k
        out[f"stoch_d_{self.k_window}"] = pct_k.rolling(self.d_window).mean()
        return out


class WilliamsR(Feature):
    """
    Williams %R.
    %R = (HighestHigh - Close) / (HighestHigh - LowestLow) * -100
    Ranges -100 to 0.  Overbought > -20, oversold < -80.
    """

    name = "williams_r"
    required_cols = [Columns.HIGH, Columns.LOW, Columns.CLOSE]
    description = "Williams %R oscillator"

    def __init__(self, window: int = 14) -> None:
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        high_max = df[Columns.HIGH].rolling(self.window).max()
        low_min  = df[Columns.LOW].rolling(self.window).min()
        out[f"williams_r_{self.window}"] = (
            -100 * (high_max - df[Columns.CLOSE]) / (high_max - low_min)
        )
        return out


class CommodityChannelIndex(Feature):
    """
    Commodity Channel Index (Lambert, 1980).
    CCI = (TypicalPrice - SMA(TP)) / (0.015 * MeanDeviation)
    """

    name = "cci"
    required_cols = [Columns.HIGH, Columns.LOW, Columns.CLOSE]
    description = "Commodity Channel Index"

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        tp  = (df[Columns.HIGH] + df[Columns.LOW] + df[Columns.CLOSE]) / 3
        sma = tp.rolling(self.window).mean()
        mad = tp.rolling(self.window).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        out[f"cci_{self.window}"] = (tp - sma) / (0.015 * mad)
        return out


class MomentumScore(Feature):
    """
    Composite normalised momentum score — average of standardised
    multi-period returns (1m, 3m, 6m, 12m).  Used in cross-sectional
    factor models and regime filters.
    """

    name = "momentum_score"
    required_cols = [Columns.CLOSE]
    description = "Multi-period normalised momentum factor score"

    def __init__(self, periods: list[int] | None = None) -> None:
        self.periods = periods or [21, 63, 126, 252]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        scores = []
        for p in self.periods:
            ret = df[Columns.CLOSE].pct_change(p)
            col = f"mom_ret_{p}"
            out[col] = ret
            # Z-score within a rolling 252-day window
            z = (ret - ret.rolling(252).mean()) / ret.rolling(252).std()
            z_col = f"mom_zscore_{p}"
            out[z_col] = z
            scores.append(z)
        if scores:
            out["momentum_composite"] = pd.concat(scores, axis=1).mean(axis=1)
        return out
