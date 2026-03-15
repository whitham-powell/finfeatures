"""
Regime-detection features.

These are higher-order features designed to be inputs to regime detectors
(HMM, iHMM, BOCPD, threshold classifiers, etc.).  They are derived purely
from price and previously-computed indicator columns — they carry no
knowledge of any specific regime detection algorithm or statistical test.

If you need a feature tied to a specific detection method (e.g. MMD, BOCPD
run-length posterior, martingale statistic), implement it in your detection
project as a Feature subclass and register it there.  Example:

    # In your project: mmd_regime/features.py
    from finfeatures.core import Feature, Columns

    class RollingMMD(Feature):
        name = "rolling_mmd"
        ...
        def compute(self, df): ...

    # Then compose it with finfeatures presets:
    from finfeatures import standard_pipeline
    pipeline = standard_pipeline().add(RollingMMD(window=63))
    enriched = pipeline.transform(raw)

Includes:
  - DistributionShiftScore: General rolling JS-divergence between adjacent windows
  - DrawdownFeatures:       Drawdown depth, duration, and recovery
  - RegimeIndicators:       Composite continuous regime scores (stress, trend, momentum)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finfeatures.core.base import Columns, Feature


class DistributionShiftScore(Feature):
    """
    Rolling distribution shift score between two adjacent windows.

    Uses Jensen-Shannon divergence over a histogram approximation — a
    general-purpose measure of how much the empirical distribution of any
    column has changed between a reference window and the current window.

    Algorithm-agnostic: captures distributional drift without committing to
    any particular statistical test or kernel method.

    Parameters
    ----------
    column:  Column to measure shift on (default: log_return)
    window:  Size of each comparison window in rows
    n_bins:  Number of histogram bins
    """

    name = "distribution_shift"
    description = "Rolling JS-divergence distribution shift score"

    def __init__(
        self,
        column: str = Columns.LOG_RETURN,
        window: int = 21,
        n_bins: int = 10,
    ) -> None:
        self.column = column
        self.window = window
        self.n_bins = n_bins

    @property
    def required_cols(self) -> list[str]:
        return [self.column]

    @staticmethod
    def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
        p = p + eps
        q = q + eps
        p /= p.sum()
        q /= q.sum()
        m = 0.5 * (p + q)
        return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        col = df[self.column].values
        n = len(col)
        scores = np.full(n, np.nan)
        w = self.window

        global_min = np.nanpercentile(col, 1)
        global_max = np.nanpercentile(col, 99)
        bins = np.linspace(global_min, global_max, self.n_bins + 1)

        for t in range(2 * w, n):
            ref_clean = col[t - 2 * w : t - w]
            cur_clean = col[t - w : t]
            ref_clean = ref_clean[~np.isnan(ref_clean)]
            cur_clean = cur_clean[~np.isnan(cur_clean)]
            if len(ref_clean) < 5 or len(cur_clean) < 5:
                continue
            p, _ = np.histogram(ref_clean, bins=bins, density=False)
            q, _ = np.histogram(cur_clean, bins=bins, density=False)
            scores[t] = self._js_divergence(p.astype(float), q.astype(float))

        out[f"dist_shift_{self.column}_{w}"] = scores
        return out


class DrawdownFeatures(Feature):
    """
    Drawdown depth, duration, and recovery derived from price alone.

    Outputs:
      - drawdown:           current drawdown from running high (≤ 0)
      - drawdown_duration:  consecutive bars below the running high
      - drawdown_recovery:  fractional recovery from trough toward prior high (0–1)
    """

    name = "drawdown"
    required_cols = [Columns.CLOSE]
    description = "Drawdown depth, duration, and recovery"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        close = df[Columns.CLOSE]
        running_max = close.cummax()
        drawdown = close / running_max - 1
        out["drawdown"] = drawdown

        below = (drawdown < 0).astype(int)
        out["drawdown_duration"] = (
            below.groupby((below != below.shift()).cumsum()).cumcount() + 1
        ) * below

        troughs: list[float] = []
        trough_val = close.iloc[0]
        for i, val in enumerate(close):
            trough_val = val if val >= running_max.iloc[i] else min(trough_val, val)
            troughs.append(trough_val)

        trough_series = pd.Series(troughs, index=close.index)
        out["drawdown_recovery"] = (
            (close - trough_series) / (running_max - trough_series + 1e-10)
        ).clip(0, 1)
        return out


class RegimeIndicators(Feature):
    """
    Composite continuous regime scores in [0, 1] built from standard
    indicator columns that must already be present in the DataFrame.

    These are inputs, not labels — a downstream classifier thresholds them.

    Requires upstream features: realized_vol_21, macd_line, rsi_14
    """

    name = "regime_indicators"
    description = "Composite soft regime scores: stress, trend, momentum"

    @property
    def required_cols(self) -> list[str]:
        return ["realized_vol_21", "macd_line", "rsi_14"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        vol = df["realized_vol_21"]
        vol_z = (vol - vol.rolling(252).mean()) / vol.rolling(252).std()
        out["stress_score"] = 1 / (1 + np.exp(-vol_z))

        macd = df["macd_line"]
        macd_z = (macd - macd.rolling(252).mean()) / (macd.rolling(252).std() + 1e-10)
        out["trend_score"] = 1 / (1 + np.exp(-macd_z))

        out["momentum_score_indicator"] = df["rsi_14"] / 100.0

        return out
