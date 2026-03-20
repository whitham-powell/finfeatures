"""
TA-Lib vs pandas fallback equivalence tests.

Runs each feature with TA-Lib enabled, then with HAS_TALIB=False,
and compares the stable output region (skipping warm-up rows).

All tests require TA-Lib and are marked accordingly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import finfeatures.core._compat as _compat_mod

pytestmark = pytest.mark.talib


@pytest.fixture(autouse=True)
def _require_talib():
    if not _compat_mod.HAS_TALIB:
        pytest.skip("TA-Lib not installed")


def _run_both_paths(feature_cls, ohlcv, **kwargs):
    """Run feature with TA-Lib, then without, return both results."""
    feature = feature_cls(**kwargs)
    with_talib = feature.compute(ohlcv)

    # Monkeypatch HAS_TALIB=False for the pandas path
    original = _compat_mod.HAS_TALIB
    try:
        _compat_mod.HAS_TALIB = False
        feature2 = feature_cls(**kwargs)
        without_talib = feature2.compute(ohlcv)
    finally:
        _compat_mod.HAS_TALIB = original

    return with_talib, without_talib


def _compare_cols(
    df_talib: pd.DataFrame,
    df_pandas: pd.DataFrame,
    cols: list[str],
    skip_rows: int,
    atol: float = 1e-6,
):
    """Compare specified columns, skipping warm-up rows."""
    for col in cols:
        assert col in df_talib.columns, f"TA-Lib missing column: {col}"
        assert col in df_pandas.columns, f"Pandas missing column: {col}"
        a = df_talib[col].iloc[skip_rows:].values.astype(float)
        b = df_pandas[col].iloc[skip_rows:].values.astype(float)
        # Remove rows where both are NaN
        mask = ~(np.isnan(a) & np.isnan(b))
        np.testing.assert_allclose(a[mask], b[mask], atol=atol, err_msg=f"Mismatch in {col}")


# ===========================================================================
# Existing features — PR 2 equivalence
# ===========================================================================


class TestSMAEquivalence:
    def test_sma(self, ohlcv_daily):
        from finfeatures.features.trend import SimpleMovingAverage

        a, b = _run_both_paths(SimpleMovingAverage, ohlcv_daily, windows=[10, 50])
        _compare_cols(a, b, ["sma_10", "sma_50"], skip_rows=50, atol=1e-10)


class TestEMAEquivalence:
    def test_ema(self, ohlcv_daily):
        from finfeatures.features.trend import ExponentialMovingAverage

        a, b = _run_both_paths(ExponentialMovingAverage, ohlcv_daily, windows=[12, 26])
        _compare_cols(a, b, ["ema_12", "ema_26"], skip_rows=78, atol=1e-6)


class TestMACDEquivalence:
    def test_macd(self, ohlcv_daily):
        from finfeatures.features.trend import MACD

        a, b = _run_both_paths(MACD, ohlcv_daily)
        _compare_cols(
            a,
            b,
            ["macd_line", "macd_signal", "macd_hist"],
            skip_rows=105,
            atol=1e-4,
        )


class TestTrendStrengthEquivalence:
    def test_adx(self, ohlcv_daily):
        from finfeatures.features.trend import TrendStrength

        a, b = _run_both_paths(TrendStrength, ohlcv_daily, window=14)
        _compare_cols(a, b, ["adx_14", "di_plus", "di_minus"], skip_rows=84, atol=0.5)


class TestRSIEquivalence:
    def test_rsi(self, ohlcv_daily):
        from finfeatures.features.momentum import RSI

        a, b = _run_both_paths(RSI, ohlcv_daily, window=14)
        _compare_cols(a, b, ["rsi_14"], skip_rows=45, atol=1e-4)


class TestROCEquivalence:
    def test_roc(self, ohlcv_daily):
        from finfeatures.features.momentum import RateOfChange

        a, b = _run_both_paths(RateOfChange, ohlcv_daily, window=10)
        _compare_cols(a, b, ["roc_10"], skip_rows=11, atol=1e-10)


class TestStochasticEquivalence:
    def test_stoch(self, ohlcv_daily):
        from finfeatures.features.momentum import StochasticOscillator

        a, b = _run_both_paths(StochasticOscillator, ohlcv_daily, k_window=14, d_window=3)
        _compare_cols(
            a,
            b,
            ["stoch_k_14", "stoch_d_14"],
            skip_rows=20,
            atol=1e-6,
        )


class TestWilliamsREquivalence:
    def test_willr(self, ohlcv_daily):
        from finfeatures.features.momentum import WilliamsR

        a, b = _run_both_paths(WilliamsR, ohlcv_daily, window=14)
        _compare_cols(a, b, ["williams_r_14"], skip_rows=14, atol=1e-10)


class TestCCIEquivalence:
    def test_cci(self, ohlcv_daily):
        from finfeatures.features.momentum import CommodityChannelIndex

        a, b = _run_both_paths(CommodityChannelIndex, ohlcv_daily, window=20)
        _compare_cols(a, b, ["cci_20"], skip_rows=20, atol=1e-6)


class TestBollingerEquivalence:
    def test_bbands(self, ohlcv_daily):
        from finfeatures.features.volatility import BollingerBands

        a, b = _run_both_paths(BollingerBands, ohlcv_daily, window=20)
        # TA-Lib uses population std, pandas uses sample std → some difference
        _compare_cols(
            a,
            b,
            ["bb_middle_20", "bb_upper_20", "bb_lower_20"],
            skip_rows=20,
            atol=0.5,
        )


class TestATREquivalence:
    def test_atr(self, ohlcv_daily):
        from finfeatures.features.volatility import AverageTrueRange

        a, b = _run_both_paths(AverageTrueRange, ohlcv_daily, window=14)
        _compare_cols(a, b, ["atr_14"], skip_rows=42, atol=1e-4)


class TestOBVEquivalence:
    def test_obv(self, ohlcv_daily):
        from finfeatures.features.volume import OnBalanceVolume

        a, b = _run_both_paths(OnBalanceVolume, ohlcv_daily)
        _compare_cols(a, b, ["obv"], skip_rows=1, atol=1e-6)


class TestTypicalPriceEquivalence:
    def test_typprice(self, ohlcv_daily):
        from finfeatures.features.price import TypicalPrice

        a, b = _run_both_paths(TypicalPrice, ohlcv_daily)
        _compare_cols(a, b, ["typical_price"], skip_rows=0, atol=1e-10)


# ===========================================================================
# New features — PR 3 equivalence
# ===========================================================================


class TestKAMAEquivalence:
    def test_kama(self, ohlcv_daily):
        from finfeatures.features.trend import KAMA

        a, b = _run_both_paths(KAMA, ohlcv_daily, window=10)
        _compare_cols(a, b, ["kama_10"], skip_rows=30, atol=1e-4)


class TestSAREquivalence:
    def test_sar(self, ohlcv_daily):
        from finfeatures.features.trend import ParabolicSAR

        a, b = _run_both_paths(ParabolicSAR, ohlcv_daily)
        _compare_cols(a, b, ["sar"], skip_rows=5, atol=0.5)


class TestDEMAEquivalence:
    def test_dema(self, ohlcv_daily):
        from finfeatures.features.trend import DEMA

        a, b = _run_both_paths(DEMA, ohlcv_daily, windows=[10, 20])
        _compare_cols(a, b, ["dema_10", "dema_20"], skip_rows=60, atol=1e-4)


class TestTEMAEquivalence:
    def test_tema(self, ohlcv_daily):
        from finfeatures.features.trend import TEMA

        a, b = _run_both_paths(TEMA, ohlcv_daily, windows=[10, 20])
        _compare_cols(a, b, ["tema_10", "tema_20"], skip_rows=60, atol=1e-4)


class TestMFIEquivalence:
    def test_mfi(self, ohlcv_daily):
        from finfeatures.features.momentum import MoneyFlowIndex

        a, b = _run_both_paths(MoneyFlowIndex, ohlcv_daily, window=14)
        _compare_cols(a, b, ["mfi_14"], skip_rows=15, atol=1e-6)


class TestAroonEquivalence:
    def test_aroon(self, ohlcv_daily):
        from finfeatures.features.momentum import Aroon

        a, b = _run_both_paths(Aroon, ohlcv_daily, window=25)
        _compare_cols(
            a,
            b,
            ["aroon_up_25", "aroon_down_25", "aroon_osc_25"],
            skip_rows=26,
            atol=1e-6,
        )


class TestCMOEquivalence:
    def test_cmo(self, ohlcv_daily):
        from finfeatures.features.momentum import ChandeMomentumOscillator

        a, b = _run_both_paths(ChandeMomentumOscillator, ohlcv_daily, window=14)
        _compare_cols(a, b, ["cmo_14"], skip_rows=45, atol=1e-4)


class TestUltOscEquivalence:
    def test_ultosc(self, ohlcv_daily):
        from finfeatures.features.momentum import UltimateOscillator

        a, b = _run_both_paths(UltimateOscillator, ohlcv_daily)
        _compare_cols(a, b, ["ultimate_osc"], skip_rows=30, atol=1e-6)


class TestADEquivalence:
    def test_ad(self, ohlcv_daily):
        from finfeatures.features.volume import AccumulationDistribution

        a, b = _run_both_paths(AccumulationDistribution, ohlcv_daily)
        _compare_cols(a, b, ["ad_line"], skip_rows=1, atol=1e-4)


class TestADOSCEquivalence:
    def test_adosc(self, ohlcv_daily):
        from finfeatures.features.volume import ChaikinADOscillator

        a, b = _run_both_paths(ChaikinADOscillator, ohlcv_daily, fast=3, slow=10)
        _compare_cols(a, b, ["adosc_3_10"], skip_rows=30, atol=1e-2)


class TestLinRegSlopeEquivalence:
    def test_linreg_slope(self, ohlcv_daily):
        from finfeatures.features.statistical import LinearRegressionSlope

        a, b = _run_both_paths(LinearRegressionSlope, ohlcv_daily, column="close", window=14)
        _compare_cols(
            a,
            b,
            ["close_linreg_slope_14"],
            skip_rows=14,
            atol=1e-6,
        )
