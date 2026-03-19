"""
Trend-following features.

Includes moving averages (SMA, EMA), MACD, and trend strength indicators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finfeatures.core.base import Columns, Feature, _validate_window, safe_divide


class SimpleMovingAverage(Feature):
    """
    Simple Moving Averages over multiple windows.
    Also computes close/SMA ratio (price relative to MA).
    """

    name = "sma"
    required_cols = [Columns.CLOSE]
    description = "Simple Moving Averages + price/MA ratio"

    def __init__(self, windows: list[int] | None = None) -> None:
        self.windows = windows or [10, 20, 50, 200]
        for w in self.windows:
            _validate_window(w, "windows element")

    @property
    def min_periods(self) -> int:
        return max(self.windows)

    @property
    def output_cols(self) -> list[str]:
        cols: list[str] = []
        for w in self.windows:
            cols.extend([f"sma_{w}", f"close_sma_{w}_ratio"])
        return cols

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for w in self.windows:
            ma = df[Columns.CLOSE].rolling(w).mean()
            out[f"sma_{w}"] = ma
            out[f"close_sma_{w}_ratio"] = safe_divide(df[Columns.CLOSE], ma) - 1
        return out


class ExponentialMovingAverage(Feature):
    """
    Exponential Moving Averages over multiple windows.
    """

    name = "ema"
    required_cols = [Columns.CLOSE]
    description = "Exponential Moving Averages"

    def __init__(self, windows: list[int] | None = None) -> None:
        self.windows = windows or [12, 26]
        for w in self.windows:
            _validate_window(w, "windows element")

    @property
    def min_periods(self) -> int:
        return max(self.windows)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for w in self.windows:
            out[f"ema_{w}"] = df[Columns.CLOSE].ewm(span=w, adjust=False).mean()
        return out


class MACD(Feature):
    """
    Moving Average Convergence/Divergence.
    Components:
      - macd_line:   EMA(fast) - EMA(slow)
      - signal_line: EMA(macd_line, signal_span)
      - histogram:   macd_line - signal_line
    """

    name = "macd"
    required_cols = [Columns.CLOSE]
    description = "MACD line, signal line, and histogram"

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> None:
        _validate_window(fast, "fast")
        _validate_window(slow, "slow")
        _validate_window(signal, "signal")
        if fast >= slow:
            raise ValueError(f"'fast' must be < 'slow', got fast={fast}, slow={slow}")
        self.fast = fast
        self.slow = slow
        self.signal = signal

    @property
    def min_periods(self) -> int:
        return self.slow + self.signal

    @property
    def output_cols(self) -> list[str]:
        return [
            "macd_line",
            "macd_signal",
            "macd_hist",
            "macd_line_pct",
            "macd_signal_pct",
            "macd_hist_pct",
        ]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        ema_fast = df[Columns.CLOSE].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df[Columns.CLOSE].ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
        out["macd_line"] = macd_line
        out["macd_signal"] = signal_line
        out["macd_hist"] = macd_line - signal_line
        # Normalise by price for cross-asset comparability
        out["macd_line_pct"] = safe_divide(macd_line, df[Columns.CLOSE])
        out["macd_signal_pct"] = safe_divide(signal_line, df[Columns.CLOSE])
        out["macd_hist_pct"] = safe_divide(out["macd_hist"], df[Columns.CLOSE])
        return out


class TrendStrength(Feature):
    """
    ADX-based trend strength indicator (Wilder, 1978).

    Outputs:
      - adx_N:    Average Directional Index (trend strength, 0-100)
      - di_plus:  +DI (uptrend strength)
      - di_minus: -DI (downtrend strength)
    """

    name = "trend_strength"
    required_cols = [Columns.HIGH, Columns.LOW, Columns.CLOSE]
    description = "ADX trend strength indicator"

    def __init__(self, window: int = 14) -> None:
        _validate_window(window)
        self.window = window

    @property
    def min_periods(self) -> int:
        return 2 * self.window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        w = self.window
        high, low, close = df[Columns.HIGH], df[Columns.LOW], df[Columns.CLOSE]
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        # True Range
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)

        # Directional movement
        up_move = high - prev_high
        down_move = prev_low - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        atr_s = pd.Series(tr).ewm(span=w, adjust=False).mean()
        plus_dm_s = pd.Series(plus_dm).ewm(span=w, adjust=False).mean()
        minus_dm_s = pd.Series(minus_dm).ewm(span=w, adjust=False).mean()

        di_plus = safe_divide(100 * plus_dm_s, atr_s)
        di_minus = safe_divide(100 * minus_dm_s, atr_s)
        dx = safe_divide(100 * (di_plus - di_minus).abs(), di_plus + di_minus)
        adx = dx.ewm(span=w, adjust=False).mean()

        out[f"adx_{w}"] = adx.values
        out["di_plus"] = di_plus.values
        out["di_minus"] = di_minus.values
        return out


class MACrossover(Feature):
    """
    Boolean / float MA crossover signal.
    ma_cross_fast_slow > 0 means fast MA is above slow MA.
    """

    name = "ma_crossover"
    description = "Fast/slow moving average crossover signal"

    def __init__(self, fast: int = 50, slow: int = 200) -> None:
        _validate_window(fast, "fast")
        _validate_window(slow, "slow")
        self.fast = fast
        self.slow = slow
        self.required_cols = [f"sma_{self.fast}", f"sma_{self.slow}"]

    @property
    def min_periods(self) -> int:
        return self.slow

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        fast_col = f"sma_{self.fast}"
        slow_col = f"sma_{self.slow}"
        spread = df[fast_col] - df[slow_col]
        out[f"ma_cross_{self.fast}_{self.slow}"] = safe_divide(spread, df[Columns.CLOSE])
        out[f"ma_cross_sign_{self.fast}_{self.slow}"] = np.sign(spread)
        return out
