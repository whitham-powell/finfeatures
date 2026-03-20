"""
Volatility features.

Covers realised/historical volatility estimators (close-to-close, Parkinson,
Garman-Klass), Bollinger Bands, and Average True Range.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finfeatures.core._compat import HAS_TALIB, _f64, talib
from finfeatures.core.base import Columns, Feature, _validate_window, safe_divide


class RollingVolatility(Feature):
    """
    Annualised rolling close-to-close realised volatility.
    σ_t = std(log_returns, window) * sqrt(trading_days)
    """

    name = "rolling_volatility"
    required_cols = [Columns.CLOSE]
    description = "Rolling annualised close-to-close realised volatility"

    def __init__(self, window: int = 21, trading_days: int = 252, annualize: bool = True) -> None:
        _validate_window(window)
        self.window = window
        self.trading_days = trading_days
        self.annualize = annualize

    @property
    def min_periods(self) -> int:
        return self.window + 1

    @property
    def output_cols(self) -> list[str]:
        if self.annualize:
            return [f"realized_vol_{self.window}"]
        return [f"raw_vol_{self.window}"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        log_ret: pd.Series = np.log(df[Columns.CLOSE] / df[Columns.CLOSE].shift(1))  # type: ignore[assignment]
        raw = log_ret.rolling(self.window).std()
        if self.annualize:
            col = f"realized_vol_{self.window}"
            out[col] = raw * np.sqrt(self.trading_days)
        else:
            col = f"raw_vol_{self.window}"
            out[col] = raw
        return out


class ParkinsonVolatility(Feature):
    """
    Parkinson (1980) high-low volatility estimator.
    More efficient than close-to-close; uses the full intraday range.

    σ_P = sqrt( 1/(4n·ln2) · Σ ln(H_t/L_t)^2 )
    """

    name = "parkinson_volatility"
    required_cols = [Columns.HIGH, Columns.LOW]
    description = "Parkinson high-low volatility estimator"

    def __init__(self, window: int = 21, trading_days: int = 252) -> None:
        _validate_window(window)
        self.window = window
        self.trading_days = trading_days

    @property
    def min_periods(self) -> int:
        return self.window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        hl_log: pd.Series = np.log(df[Columns.HIGH] / df[Columns.LOW])  # type: ignore[assignment]
        hl_sq = hl_log**2
        factor = 1.0 / (4.0 * self.window * np.log(2))
        col = f"parkinson_vol_{self.window}"
        out[col] = hl_sq.rolling(self.window).sum().mul(factor).pow(0.5) * np.sqrt(
            self.trading_days
        )
        return out


class GarmanKlassVolatility(Feature):
    """
    Garman-Klass (1980) open-high-low-close volatility estimator.
    More efficient than Parkinson; accounts for overnight gaps.
    """

    name = "garman_klass_volatility"
    required_cols = [Columns.OPEN, Columns.HIGH, Columns.LOW, Columns.CLOSE]
    description = "Garman-Klass OHLC volatility estimator"

    def __init__(self, window: int = 21, trading_days: int = 252) -> None:
        _validate_window(window)
        self.window = window
        self.trading_days = trading_days

    @property
    def min_periods(self) -> int:
        return self.window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        log_hl: pd.Series = np.log(df[Columns.HIGH] / df[Columns.LOW])  # type: ignore[assignment]
        log_co: pd.Series = np.log(df[Columns.CLOSE] / df[Columns.OPEN])  # type: ignore[assignment]
        gk = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        col = f"garman_klass_vol_{self.window}"
        out[col] = gk.rolling(self.window).mean().pow(0.5) * np.sqrt(self.trading_days)
        return out


class BollingerBands(Feature):
    """
    Bollinger Bands: upper, middle (SMA), lower bands and %B width.
    """

    name = "bollinger_bands"
    required_cols = [Columns.CLOSE]
    description = "Bollinger Bands (upper, middle, lower, %B, width)"

    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        _validate_window(window)
        self.window = window
        self.num_std = num_std

    @property
    def min_periods(self) -> int:
        return self.window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        w = self.window
        close = df[Columns.CLOSE]
        if HAS_TALIB:
            upper_arr, middle_arr, lower_arr = talib.BBANDS(
                _f64(close),
                timeperiod=w,
                nbdevup=self.num_std,
                nbdevdn=self.num_std,
                matype=0,  # type: ignore[arg-type]  # MA_Type.SMA
            )
            upper = pd.Series(upper_arr, index=close.index)
            sma = pd.Series(middle_arr, index=close.index)
            lower = pd.Series(lower_arr, index=close.index)
        else:
            sma = close.rolling(w).mean()
            std = close.rolling(w).std()
            upper = sma + self.num_std * std
            lower = sma - self.num_std * std
        out[f"bb_middle_{w}"] = sma
        out[f"bb_upper_{w}"] = upper
        out[f"bb_lower_{w}"] = lower
        # %B: position within bands (0 = lower, 1 = upper)
        out[f"bb_pct_{w}"] = safe_divide(close - lower, upper - lower)
        # Bandwidth: normalised width of the bands
        out[f"bb_width_{w}"] = safe_divide(upper - lower, sma)
        return out


class AverageTrueRange(Feature):
    """
    Wilder's Average True Range (ATR).
    True Range = max(H-L, |H-Cprev|, |L-Cprev|)
    ATR = EWM mean of True Range with span = window.
    """

    name = "average_true_range"
    required_cols = [Columns.HIGH, Columns.LOW, Columns.CLOSE]
    description = "Average True Range (Wilder)"

    def __init__(self, window: int = 14) -> None:
        _validate_window(window)
        self.window = window

    @property
    def min_periods(self) -> int:
        return self.window + 1

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        col = f"atr_{self.window}"
        if HAS_TALIB:
            out[col] = talib.ATR(
                _f64(df[Columns.HIGH]),
                _f64(df[Columns.LOW]),
                _f64(df[Columns.CLOSE]),
                timeperiod=self.window,
            )
        else:
            prev_close = df[Columns.CLOSE].shift(1)
            tr = pd.concat(
                [
                    df[Columns.HIGH] - df[Columns.LOW],
                    (df[Columns.HIGH] - prev_close).abs(),
                    (df[Columns.LOW] - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            out[col] = tr.ewm(span=self.window, adjust=False).mean()
        # Normalised ATR (as % of close)
        out[f"atr_pct_{self.window}"] = safe_divide(out[col], df[Columns.CLOSE])
        return out


class VolatilityRatio(Feature):
    """
    Short-term vs long-term volatility ratio.
    Compares short-term vol to long-term vol:
      vol_ratio = realised_vol_short / realised_vol_long
    Values > 1 indicate elevated short-term volatility.
    Requires RollingVolatility to have been run for both windows.
    """

    name = "volatility_ratio"
    description = "Short/long vol ratio"

    def __init__(self, short_window: int = 21, long_window: int = 63) -> None:
        _validate_window(short_window, "short_window")
        _validate_window(long_window, "long_window")
        self.short_window = short_window
        self.long_window = long_window
        self.required_cols = [
            f"realized_vol_{self.short_window}",
            f"realized_vol_{self.long_window}",
        ]

    @property
    def min_periods(self) -> int:
        return self.long_window + 1

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        short = df[f"realized_vol_{self.short_window}"]
        long_ = df[f"realized_vol_{self.long_window}"]
        out["vol_ratio"] = safe_divide(short, long_)
        out["vol_ratio_zscore"] = safe_divide(short - long_, long_.rolling(self.long_window).std())
        return out


class MovingTrueRange(Feature):
    """
    Simple rolling mean of True Range at multiple horizons.

    Unlike AverageTrueRange (which uses EWM smoothing), this computes a
    plain rolling mean of true range.
    """

    name = "moving_true_range"
    required_cols = [Columns.HIGH, Columns.LOW, Columns.CLOSE]
    description = "Rolling mean of true range at multiple horizons"

    def __init__(self, windows: list[int] | None = None) -> None:
        self.windows = windows or [20, 50, 200]
        for w in self.windows:
            _validate_window(w, "windows element")

    @property
    def min_periods(self) -> int:
        return max(self.windows) + 1

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        prev_close = df[Columns.CLOSE].shift(1)
        tr = pd.concat(
            [
                df[Columns.HIGH] - df[Columns.LOW],
                (df[Columns.HIGH] - prev_close).abs(),
                (df[Columns.LOW] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        for w in self.windows:
            out[f"mtr_{w}"] = tr.rolling(w).mean()
        return out


class KeltnerChannels(Feature):
    """
    Keltner Channels — EMA-based bands using ATR for width.

    Similar to Bollinger Bands but uses ATR instead of standard deviation,
    making it less sensitive to outliers. Often used with Bollinger Bands
    to detect volatility squeezes.

    Outputs:
      - keltner_upper_N, keltner_mid_N, keltner_lower_N
      - keltner_pct_N: position within bands (0 = lower, 1 = upper)
    """

    name = "keltner_channels"
    required_cols = [Columns.HIGH, Columns.LOW, Columns.CLOSE]
    description = "Keltner Channels (EMA ± ATR multiplier)"

    def __init__(self, window: int = 20, multiplier: float = 2.0) -> None:
        _validate_window(window)
        if multiplier <= 0:
            raise ValueError(f"'multiplier' must be > 0, got {multiplier!r}")
        self.window = window
        self.multiplier = multiplier

    @property
    def min_periods(self) -> int:
        return self.window + 1

    @property
    def output_cols(self) -> list[str]:
        w = self.window
        return [
            f"keltner_upper_{w}",
            f"keltner_mid_{w}",
            f"keltner_lower_{w}",
            f"keltner_pct_{w}",
        ]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        w = self.window
        close = df[Columns.CLOSE]

        # Middle band: EMA of close
        if HAS_TALIB:
            mid = pd.Series(talib.EMA(_f64(close), timeperiod=w), index=close.index)
            atr = pd.Series(
                talib.ATR(_f64(df[Columns.HIGH]), _f64(df[Columns.LOW]), _f64(close), timeperiod=w),
                index=close.index,
            )
        else:
            mid = close.ewm(span=w, adjust=False).mean()
            prev_close = close.shift(1)
            tr = pd.concat(
                [
                    df[Columns.HIGH] - df[Columns.LOW],
                    (df[Columns.HIGH] - prev_close).abs(),
                    (df[Columns.LOW] - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr = tr.ewm(span=w, adjust=False).mean()

        upper = mid + self.multiplier * atr
        lower = mid - self.multiplier * atr
        out[f"keltner_upper_{w}"] = upper
        out[f"keltner_mid_{w}"] = mid
        out[f"keltner_lower_{w}"] = lower
        out[f"keltner_pct_{w}"] = safe_divide(close - lower, upper - lower)
        return out
