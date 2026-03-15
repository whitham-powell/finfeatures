"""
Volume-based features.

On-Balance Volume, VWAP, volume z-score, and rolling participation ratios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finfeatures.core.base import Columns, Feature


class VolumeFeatures(Feature):
    """
    Bundle of basic volume features:
      - volume_return:   day-over-day % change in volume
      - volume_zscore:   rolling z-score of volume
      - volume_sma_ratio: volume / rolling SMA (relative volume)
    """

    name = "volume_features"
    required_cols = [Columns.VOLUME]
    description = "Volume return, z-score, and relative volume"

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        vol = df[Columns.VOLUME]
        out["volume_return"] = vol.pct_change()
        vol_sma = vol.rolling(self.window).mean()
        vol_std = vol.rolling(self.window).std()
        out[f"volume_zscore_{self.window}"] = (vol - vol_sma) / vol_std
        out[f"volume_rel_{self.window}"] = vol / vol_sma
        return out


class OnBalanceVolume(Feature):
    """
    On-Balance Volume (Granville, 1963).
    OBV accumulates volume: +vol on up days, -vol on down days.
    """

    name = "obv"
    required_cols = [Columns.CLOSE, Columns.VOLUME]
    description = "On-Balance Volume"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        direction = np.sign(df[Columns.CLOSE].diff()).fillna(0)
        out["obv"] = (direction * df[Columns.VOLUME]).cumsum()
        return out


class VWAP(Feature):
    """
    Rolling Volume-Weighted Average Price.
    vwap_N = sum(typical_price * volume, window) / sum(volume, window)
    Note: This is a rolling VWAP, not session VWAP.
    """

    name = "vwap"
    required_cols = [Columns.HIGH, Columns.LOW, Columns.CLOSE, Columns.VOLUME]
    description = "Rolling Volume-Weighted Average Price"

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        tp = (df[Columns.HIGH] + df[Columns.LOW] + df[Columns.CLOSE]) / 3
        vol = df[Columns.VOLUME]
        vwap = (tp * vol).rolling(self.window).sum() / vol.rolling(self.window).sum()
        out[f"vwap_{self.window}"] = vwap
        out[f"vwap_ratio_{self.window}"] = df[Columns.CLOSE] / vwap - 1
        return out


class ChaikinMoneyFlow(Feature):
    """
    Chaikin Money Flow.
    CMF = sum(MFV, window) / sum(Volume, window)
    MFV = ((Close - Low) - (High - Close)) / (High - Low) * Volume
    """

    name = "chaikin_mf"
    required_cols = [Columns.HIGH, Columns.LOW, Columns.CLOSE, Columns.VOLUME]
    description = "Chaikin Money Flow"

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        hl = df[Columns.HIGH] - df[Columns.LOW]
        hl_safe = hl.replace(0, np.nan)
        mf_multiplier = (
            (df[Columns.CLOSE] - df[Columns.LOW]) - (df[Columns.HIGH] - df[Columns.CLOSE])
        ) / hl_safe
        mfv = mf_multiplier * df[Columns.VOLUME]
        out[f"cmf_{self.window}"] = (
            mfv.rolling(self.window).sum() / df[Columns.VOLUME].rolling(self.window).sum()
        )
        return out
