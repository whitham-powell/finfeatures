"""
Volatility features.

Covers realised/historical volatility estimators (close-to-close, Parkinson,
Garman-Klass), Bollinger Bands, and Average True Range.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finfeatures.core.base import Columns, Feature


class RollingVolatility(Feature):
    """
    Annualised rolling close-to-close realised volatility.
    σ_t = std(log_returns, window) * sqrt(trading_days)
    """

    name = "rolling_volatility"
    required_cols = [Columns.CLOSE]
    description = "Rolling annualised close-to-close realised volatility"

    def __init__(self, window: int = 21, trading_days: int = 252) -> None:
        self.window = window
        self.trading_days = trading_days

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        log_ret = np.log(df[Columns.CLOSE] / df[Columns.CLOSE].shift(1))
        col = f"realized_vol_{self.window}"
        out[col] = log_ret.rolling(self.window).std() * np.sqrt(self.trading_days)
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
        self.window = window
        self.trading_days = trading_days

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        hl_sq = np.log(df[Columns.HIGH] / df[Columns.LOW]) ** 2
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
        self.window = window
        self.trading_days = trading_days

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        log_hl = np.log(df[Columns.HIGH] / df[Columns.LOW])
        log_co = np.log(df[Columns.CLOSE] / df[Columns.OPEN])
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
        self.window = window
        self.num_std = num_std

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        w = self.window
        sma = df[Columns.CLOSE].rolling(w).mean()
        std = df[Columns.CLOSE].rolling(w).std()
        upper = sma + self.num_std * std
        lower = sma - self.num_std * std
        out[f"bb_middle_{w}"] = sma
        out[f"bb_upper_{w}"] = upper
        out[f"bb_lower_{w}"] = lower
        # %B: position within bands (0 = lower, 1 = upper)
        out[f"bb_pct_{w}"] = (df[Columns.CLOSE] - lower) / (upper - lower)
        # Bandwidth: normalised width of the bands
        out[f"bb_width_{w}"] = (upper - lower) / sma
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
        self.window = window

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
        col = f"atr_{self.window}"
        out[col] = tr.ewm(span=self.window, adjust=False).mean()
        # Normalised ATR (as % of close)
        out[f"atr_pct_{self.window}"] = out[col] / df[Columns.CLOSE]
        return out


class VolatilityRegime(Feature):
    """
    Composite volatility regime indicator.
    Compares short-term vol to long-term vol:
      vol_ratio = realised_vol_short / realised_vol_long
    Values > 1 indicate elevated volatility (stress regime).
    Requires RollingVolatility to have been run for both windows.
    """

    name = "volatility_regime"
    description = "Short/long vol ratio as a regime indicator"

    def __init__(self, short_window: int = 21, long_window: int = 63) -> None:
        self.short_window = short_window
        self.long_window = long_window

    @property
    def required_cols(self) -> list[str]:
        return [
            f"realized_vol_{self.short_window}",
            f"realized_vol_{self.long_window}",
        ]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        short = df[f"realized_vol_{self.short_window}"]
        long_ = df[f"realized_vol_{self.long_window}"]
        out["vol_regime_ratio"] = short / long_
        out["vol_regime_zscore"] = (short - long_) / long_.rolling(self.long_window).std()
        return out
