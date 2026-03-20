"""
Trend-following features.

Includes moving averages (SMA, EMA), MACD, and trend strength indicators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finfeatures.core._compat import HAS_TALIB, _f64, talib
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
        close = df[Columns.CLOSE]
        for w in self.windows:
            if HAS_TALIB and w >= 2:
                ma = pd.Series(talib.SMA(_f64(close), timeperiod=w), index=close.index)
            else:
                ma = close.rolling(w).mean()
            out[f"sma_{w}"] = ma
            out[f"close_sma_{w}_ratio"] = safe_divide(close, ma) - 1
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
        close = df[Columns.CLOSE]
        for w in self.windows:
            if HAS_TALIB:
                out[f"ema_{w}"] = talib.EMA(_f64(close), timeperiod=w)
            else:
                out[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()
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
        close = df[Columns.CLOSE]
        if HAS_TALIB:
            macd_vals, signal_vals, hist_vals = talib.MACD(
                _f64(close),
                fastperiod=self.fast,
                slowperiod=self.slow,
                signalperiod=self.signal,
            )
            macd_line = pd.Series(macd_vals, index=close.index)
            signal_line = pd.Series(signal_vals, index=close.index)
            out["macd_line"] = macd_line
            out["macd_signal"] = signal_line
            out["macd_hist"] = pd.Series(hist_vals, index=close.index)
        else:
            ema_fast = close.ewm(span=self.fast, adjust=False).mean()
            ema_slow = close.ewm(span=self.slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
            out["macd_line"] = macd_line
            out["macd_signal"] = signal_line
            out["macd_hist"] = macd_line - signal_line
        # Normalise by price for cross-asset comparability
        out["macd_line_pct"] = safe_divide(out["macd_line"], close)
        out["macd_signal_pct"] = safe_divide(out["macd_signal"], close)
        out["macd_hist_pct"] = safe_divide(out["macd_hist"], close)
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
        if HAS_TALIB:
            h, lo, c = _f64(high), _f64(low), _f64(close)
            out[f"adx_{w}"] = talib.ADX(h, lo, c, timeperiod=w)
            out["di_plus"] = talib.PLUS_DI(h, lo, c, timeperiod=w)
            out["di_minus"] = talib.MINUS_DI(h, lo, c, timeperiod=w)
        else:
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


class KAMA(Feature):
    """
    Kaufman's Adaptive Moving Average.
    Adjusts smoothing based on the efficiency ratio (direction / volatility).
    """

    name = "kama"
    required_cols = [Columns.CLOSE]
    description = "Kaufman Adaptive Moving Average"

    def __init__(self, window: int = 10, fast_period: int = 2, slow_period: int = 30) -> None:
        _validate_window(window)
        _validate_window(fast_period, "fast_period")
        _validate_window(slow_period, "slow_period")
        self.window = window
        self.fast_period = fast_period
        self.slow_period = slow_period

    @property
    def min_periods(self) -> int:
        return self.window

    @property
    def output_cols(self) -> list[str]:
        return [f"kama_{self.window}"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        close = df[Columns.CLOSE]
        if HAS_TALIB:
            out[f"kama_{self.window}"] = talib.KAMA(_f64(close), timeperiod=self.window)
        else:
            fast_sc = 2.0 / (self.fast_period + 1)
            slow_sc = 2.0 / (self.slow_period + 1)
            vals = close.values.astype(float)
            n = len(vals)
            kama = np.full(n, np.nan)
            w = self.window
            if n > w:
                kama[w - 1] = vals[w - 1]
                for i in range(w, n):
                    direction = abs(vals[i] - vals[i - w])
                    volatility = np.sum(np.abs(np.diff(vals[i - w : i + 1])))
                    er = 0.0 if volatility == 0 else direction / volatility
                    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
                    kama[i] = kama[i - 1] + sc * (vals[i] - kama[i - 1])
            out[f"kama_{self.window}"] = kama
        return out


class ParabolicSAR(Feature):
    """
    Parabolic Stop and Reverse (Wilder, 1978).
    Tracks price with an accelerating trailing stop.
    """

    name = "parabolic_sar"
    required_cols = [Columns.HIGH, Columns.LOW]
    description = "Parabolic SAR"

    def __init__(self, acceleration: float = 0.02, maximum: float = 0.2) -> None:
        if acceleration <= 0:
            raise ValueError(f"'acceleration' must be > 0, got {acceleration!r}")
        if maximum <= 0:
            raise ValueError(f"'maximum' must be > 0, got {maximum!r}")
        self.acceleration = acceleration
        self.maximum = maximum

    @property
    def min_periods(self) -> int:
        return 2

    @property
    def output_cols(self) -> list[str]:
        return ["sar"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        high, low = df[Columns.HIGH], df[Columns.LOW]
        if HAS_TALIB:
            out["sar"] = talib.SAR(
                _f64(high),
                _f64(low),
                acceleration=self.acceleration,
                maximum=self.maximum,
            )
        else:
            h = high.values.astype(float)
            lo = low.values.astype(float)
            n = len(h)
            sar = np.full(n, np.nan)
            if n < 2:
                out["sar"] = sar
                return out
            # Initialise: assume uptrend
            af = self.acceleration
            bull = True
            ep = h[0]
            sar[0] = lo[0]
            for i in range(1, n):
                prev_sar = sar[i - 1]
                sar[i] = prev_sar + af * (ep - prev_sar)
                if bull:
                    if lo[i] < sar[i]:
                        bull = False
                        sar[i] = ep
                        ep = lo[i]
                        af = self.acceleration
                    else:
                        if h[i] > ep:
                            ep = h[i]
                            af = min(af + self.acceleration, self.maximum)
                        sar[i] = min(sar[i], lo[i - 1])
                        if i >= 2:
                            sar[i] = min(sar[i], lo[i - 2])
                else:
                    if h[i] > sar[i]:
                        bull = True
                        sar[i] = ep
                        ep = h[i]
                        af = self.acceleration
                    else:
                        if lo[i] < ep:
                            ep = lo[i]
                            af = min(af + self.acceleration, self.maximum)
                        sar[i] = max(sar[i], h[i - 1])
                        if i >= 2:
                            sar[i] = max(sar[i], h[i - 2])
            out["sar"] = sar
        return out


class DEMA(Feature):
    """
    Double Exponential Moving Average.
    DEMA = 2 * EMA(close, w) - EMA(EMA(close, w), w)
    """

    name = "dema"
    required_cols = [Columns.CLOSE]
    description = "Double Exponential Moving Average"

    def __init__(self, windows: list[int] | None = None) -> None:
        self.windows = windows or [10, 20]
        for w in self.windows:
            _validate_window(w, "windows element")

    @property
    def min_periods(self) -> int:
        return 2 * max(self.windows)

    @property
    def output_cols(self) -> list[str]:
        return [f"dema_{w}" for w in self.windows]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        close = df[Columns.CLOSE]
        for w in self.windows:
            if HAS_TALIB:
                out[f"dema_{w}"] = talib.DEMA(_f64(close), timeperiod=w)
            else:
                ema1 = close.ewm(span=w, adjust=False).mean()
                ema2 = ema1.ewm(span=w, adjust=False).mean()
                out[f"dema_{w}"] = 2 * ema1 - ema2
        return out


class TEMA(Feature):
    """
    Triple Exponential Moving Average.
    TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
    """

    name = "tema"
    required_cols = [Columns.CLOSE]
    description = "Triple Exponential Moving Average"

    def __init__(self, windows: list[int] | None = None) -> None:
        self.windows = windows or [10, 20]
        for w in self.windows:
            _validate_window(w, "windows element")

    @property
    def min_periods(self) -> int:
        return 3 * max(self.windows)

    @property
    def output_cols(self) -> list[str]:
        return [f"tema_{w}" for w in self.windows]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        close = df[Columns.CLOSE]
        for w in self.windows:
            if HAS_TALIB:
                out[f"tema_{w}"] = talib.TEMA(_f64(close), timeperiod=w)
            else:
                ema1 = close.ewm(span=w, adjust=False).mean()
                ema2 = ema1.ewm(span=w, adjust=False).mean()
                ema3 = ema2.ewm(span=w, adjust=False).mean()
                out[f"tema_{w}"] = 3 * ema1 - 3 * ema2 + ema3
        return out
