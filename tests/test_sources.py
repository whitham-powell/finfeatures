"""
Tests for data sources.

Network-dependent tests are marked with @pytest.mark.network and skipped
by default.  Run with:  pytest -m network
"""

from __future__ import annotations

import pandas as pd
import pytest

from finfeatures.core import Columns
from finfeatures.sources.yfinance import YFinanceSource


class TestYFinanceNormalise:
    """Tests for the _normalise static method — no network required."""

    def _make_raw_yf(self) -> pd.DataFrame:
        """Simulate the DataFrame that yfinance returns."""
        dates = pd.date_range("2020-01-02", periods=5, freq="B", tz="America/New_York")
        return pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "High": [101.0, 102.0, 103.0, 104.0, 105.0],
                "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
                "Volume": [1_000_000] * 5,
                "Dividends": [0.0] * 5,
                "Stock Splits": [0.0] * 5,
            },
            index=dates,
        )

    def test_normalise_renames_columns(self):
        raw = self._make_raw_yf()
        result = YFinanceSource._normalise(raw)
        for col in Columns.OHLCV:
            assert col in result.columns, f"Column '{col}' missing after normalise"

    def test_normalise_drops_extra_columns(self):
        raw = self._make_raw_yf()
        result = YFinanceSource._normalise(raw)
        assert "Dividends" not in result.columns
        assert "Stock Splits" not in result.columns

    def test_normalise_removes_timezone(self):
        raw = self._make_raw_yf()
        result = YFinanceSource._normalise(raw)
        assert result.index.tz is None

    def test_normalise_index_name(self):
        raw = self._make_raw_yf()
        result = YFinanceSource._normalise(raw)
        assert result.index.name == "date"

    def test_normalise_column_count(self):
        raw = self._make_raw_yf()
        result = YFinanceSource._normalise(raw)
        assert len(result.columns) == 5  # only OHLCV


@pytest.mark.network
class TestYFinanceFetch:
    """Live network tests — run with: pytest -m network"""

    def test_fetch_spy_returns_ohlcv(self):
        source = YFinanceSource()
        df = source.fetch("SPY", start="2024-01-01", end="2024-03-31")
        assert isinstance(df, pd.DataFrame)
        for col in Columns.OHLCV:
            assert col in df.columns
        assert len(df) > 0
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is None

    def test_fetch_multiple_returns_dict(self):
        source = YFinanceSource()
        result = source.fetch_multiple(["SPY", "QQQ"], start="2024-01-01", end="2024-02-28")
        assert set(result.keys()) == {"SPY", "QQQ"}
        for df in result.values():
            assert isinstance(df, pd.DataFrame)

    def test_fetch_invalid_symbol_raises(self):
        source = YFinanceSource()
        with pytest.raises(ValueError, match="no data"):
            source.fetch("XYZXYZXYZ_NOT_REAL", start="2024-01-01", end="2024-03-31")
