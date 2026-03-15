"""
Statistical features.

Rolling distributional statistics: z-score normalisation, rolling higher
moments (skewness, kurtosis), autocorrelation, and cross-asset correlation.
These are the primary inputs to regime detection models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finfeatures.core.base import Columns, Feature


class RollingZScore(Feature):
    """
    Rolling z-score of any named column.
    z_t = (x_t - μ_{t,window}) / σ_{t,window}
    """

    name = "rolling_zscore"
    description = "Rolling z-score normalisation of a target column"

    def __init__(self, column: str = Columns.LOG_RETURN, window: int = 21) -> None:
        self.column = column
        self.window = window

    @property
    def required_cols(self) -> list[str]:
        return [self.column]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        col = df[self.column]
        mu = col.rolling(self.window).mean()
        sig = col.rolling(self.window).std()
        out[f"{self.column}_zscore_{self.window}"] = (col - mu) / sig
        return out


class RollingSkewKurt(Feature):
    """
    Rolling skewness and excess kurtosis of any named column.
    Used to characterise tail risk in the return distribution —
    a key input for regime detection.
    """

    name = "rolling_skew_kurt"
    description = "Rolling skewness and excess kurtosis"

    def __init__(self, column: str = Columns.LOG_RETURN, window: int = 63) -> None:
        self.column = column
        self.window = window

    @property
    def required_cols(self) -> list[str]:
        return [self.column]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        col = df[self.column]
        w = self.window
        out[f"{self.column}_skew_{w}"] = col.rolling(w).skew()
        out[f"{self.column}_kurt_{w}"] = col.rolling(w).kurt()  # excess kurtosis
        return out


class RollingMoments(Feature):
    """
    Full set of rolling moments for a column: mean, std, skewness, kurtosis.
    """

    name = "rolling_moments"
    description = "Full rolling distributional moments (mean, std, skew, kurt)"

    def __init__(self, column: str = Columns.LOG_RETURN, window: int = 63) -> None:
        self.column = column
        self.window = window

    @property
    def required_cols(self) -> list[str]:
        return [self.column]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        col = df[self.column]
        w = self.window
        out[f"{self.column}_mean_{w}"] = col.rolling(w).mean()
        out[f"{self.column}_std_{w}"] = col.rolling(w).std()
        out[f"{self.column}_skew_{w}"] = col.rolling(w).skew()
        out[f"{self.column}_kurt_{w}"] = col.rolling(w).kurt()
        # Value at Risk (historical, non-parametric)
        out[f"{self.column}_var5_{w}"] = col.rolling(w).quantile(0.05)
        out[f"{self.column}_cvar5_{w}"] = col.rolling(w).apply(
            lambda x: x[x <= np.quantile(x, 0.05)].mean(), raw=True
        )
        return out


class RollingAutocorrelation(Feature):
    """
    Rolling autocorrelation at a given lag.
    Elevated autocorrelation may indicate trending regimes.
    """

    name = "rolling_autocorr"
    description = "Rolling autocorrelation at a given lag"

    def __init__(
        self,
        column: str = Columns.LOG_RETURN,
        window: int = 63,
        lag: int = 1,
    ) -> None:
        self.column = column
        self.window = window
        self.lag = lag

    @property
    def required_cols(self) -> list[str]:
        return [self.column]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        col = df[self.column]
        out[f"{self.column}_autocorr_lag{self.lag}_{self.window}"] = col.rolling(self.window).apply(
            lambda x: pd.Series(x).autocorr(lag=self.lag), raw=True
        )
        return out


class RollingCorrelation(Feature):
    """
    Rolling correlation between two named columns.
    Useful for tracking dynamic cross-asset relationships.
    """

    name = "rolling_correlation"
    description = "Rolling Pearson correlation between two columns"

    def __init__(
        self,
        col_a: str = Columns.LOG_RETURN,
        col_b: str = "realized_vol_21",
        window: int = 63,
    ) -> None:
        self.col_a = col_a
        self.col_b = col_b
        self.window = window

    @property
    def required_cols(self) -> list[str]:
        return [self.col_a, self.col_b]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[f"corr_{self.col_a}_{self.col_b}_{self.window}"] = (
            df[self.col_a].rolling(self.window).corr(df[self.col_b])
        )
        return out
