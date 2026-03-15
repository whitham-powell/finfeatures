"""
Shared pytest fixtures for finfeatures tests.

All fixtures produce synthetic OHLCV data so the test suite has
zero network dependencies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ohlcv_daily() -> pd.DataFrame:
    """
    500 rows of synthetic daily OHLCV data with a DatetimeIndex.
    Price follows a GBM, volume is log-normal.
    """
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.date_range("2020-01-02", periods=n, freq="B")

    # Geometric Brownian Motion price series
    dt = 1 / 252
    mu, sigma = 0.08, 0.20
    log_returns = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n)
    close = 100.0 * np.exp(np.cumsum(log_returns))

    # Construct OHLC from close
    noise = rng.uniform(0.002, 0.010, n)
    open_  = close * (1 + rng.uniform(-0.005, 0.005, n))
    high   = close * (1 + noise)
    low    = close * (1 - noise)
    volume = rng.lognormal(mean=15, sigma=0.5, size=n).astype(int)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture
def ohlcv_short() -> pd.DataFrame:
    """30-row version for fast edge-case tests."""
    rng = np.random.default_rng(0)
    n = 30
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    noise = rng.uniform(0.002, 0.008, n)
    return pd.DataFrame(
        {
            "open":   close * (1 + rng.uniform(-0.003, 0.003, n)),
            "high":   close * (1 + noise),
            "low":    close * (1 - noise),
            "close":  close,
            "volume": rng.integers(100_000, 1_000_000, n),
        },
        index=dates,
    )


@pytest.fixture
def raw_cols() -> list[str]:
    return ["open", "high", "low", "close", "volume"]
