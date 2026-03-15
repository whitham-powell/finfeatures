"""
IO adapters — backend abstraction layer.

Provides a uniform interface over different DataFrame backends.
Currently ships with a PandasAdapter (default).

To add support for Polars, cuDF, Modin etc:
    1. Create a new adapter class implementing the DataFrameAdapter protocol
    2. Register it here
    3. Pass it to FeaturePipeline (future work — pipeline currently hard-codes pandas)

The adapter layer exists primarily to:
  - Document where coupling to pandas occurs
  - Make future migration straightforward
  - Provide conversion helpers for interop
"""

from __future__ import annotations

from typing import Any

import pandas as pd


class PandasAdapter:
    """Default adapter.  Wraps a pd.DataFrame with normalisation utilities."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def to_numpy(self, columns: list[str] | None = None) -> Any:
        """Return a numpy array of the specified columns (all if None)."""
        import numpy as np

        if columns:
            return self._df[columns].to_numpy(dtype=float, na_value=np.nan)
        return self._df.to_numpy(dtype=float, na_value=np.nan)

    def feature_columns(self, exclude_raw: bool = True) -> list[str]:
        """Return derived feature column names, optionally excluding raw OHLCV."""
        from finfeatures.core.base import Columns

        raw = set(Columns.OHLCV) if exclude_raw else set()
        return [c for c in self._df.columns if c not in raw]

    def dropna_features(self) -> pd.DataFrame:
        """Drop rows where any feature column is NaN (from warm-up periods)."""
        return self._df.dropna()

    def __repr__(self) -> str:
        return f"PandasAdapter(shape={self._df.shape}, columns={list(self._df.columns)[:6]}...)"


# ---------------------------------------------------------------------------
# Polars stub — placeholder that raises a clear error until implemented
# ---------------------------------------------------------------------------


class PolarsAdapter:
    """
    Placeholder adapter for Polars DataFrames.
    Raises NotImplementedError until the Polars backend is implemented.
    Exists to document the intended extension point.
    """

    def __init__(self, df: Any) -> None:
        raise NotImplementedError(
            "Polars adapter is not yet implemented.  "
            "Convert your Polars DataFrame to pandas first:\n"
            "    df_pandas = df.to_pandas()\n"
            "and use PandasAdapter instead."
        )
