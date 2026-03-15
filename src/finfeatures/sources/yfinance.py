"""
YFinance DataSource implementation.

This is the default/reference implementation of DataSource.
The library does not depend on yfinance at import time — the import
is deferred to the fetch() call, so yfinance is optional.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from finfeatures.core.base import Columns, DataSource

_YFINANCE_INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "1d": "1d",
    "1wk": "1wk",
    "1mo": "1mo",
}


class YFinanceSource(DataSource):
    """
    DataSource backed by yfinance.

    Normalises yfinance's mixed-case column names to lowercase Columns
    constants and ensures a clean DatetimeIndex.

    Parameters
    ----------
    auto_adjust:  Pass through to yfinance.  Default True (adjusted prices).
    """

    def __init__(self, auto_adjust: bool = True) -> None:
        self.auto_adjust = auto_adjust

    def fetch(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        interval: str = "1d",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single symbol.

        Returns a DataFrame with lowercase columns (open, high, low, close, volume)
        and a DatetimeIndex.
        """
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError(
                "yfinance is required for YFinanceSource.  Install it with:  pip install yfinance"
            ) from e

        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start,
            end=end,
            interval=_YFINANCE_INTERVAL_MAP.get(interval, interval),
            auto_adjust=self.auto_adjust,
            **kwargs,
        )

        if df.empty:
            raise ValueError(
                f"yfinance returned no data for symbol '{symbol}' "
                f"(start={start}, end={end}, interval={interval})."
            )

        return self._normalise(df)

    def fetch_multiple(
        self,
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
        interval: str = "1d",
        **kwargs: Any,
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV data for multiple symbols. Returns {symbol: df}."""
        return {
            symbol: self.fetch(symbol, start=start, end=end, interval=interval, **kwargs)
            for symbol in symbols
        }

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise a raw yfinance DataFrame to standard column names.
        yfinance returns: Open, High, Low, Close, Volume (+ Dividends, Stock Splits)
        We keep only OHLCV with lowercase names.
        """
        rename_map = {
            "Open": Columns.OPEN,
            "High": Columns.HIGH,
            "Low": Columns.LOW,
            "Close": Columns.CLOSE,
            "Volume": Columns.VOLUME,
        }
        available = {k: v for k, v in rename_map.items() if k in df.columns}
        result = df[list(available.keys())].rename(columns=available).copy()

        # Ensure DatetimeIndex is tz-naive for consistency
        if result.index.tz is not None:
            result.index = result.index.tz_localize(None)

        result.index.name = "date"
        return result
