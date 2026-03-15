"""
Tests for individual feature classes.
One test class per feature module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finfeatures.core import Columns
from finfeatures.features.composite import (
    DistributionShiftScore,
    DrawdownFeatures,
)
from finfeatures.features.momentum import (
    RSI,
    RateOfChange,
    StochasticOscillator,
)
from finfeatures.features.price import (
    CandleShape,
    CrossDay,
    CumulativeReturn,
    LogReturns,
    LogTransform,
    PriceRange,
    Returns,
    ShapeDynamics,
    TypicalPrice,
)
from finfeatures.features.statistical import (
    RollingSkewKurt,
    RollingZScore,
)
from finfeatures.features.trend import (
    MACD,
    ExponentialMovingAverage,
    SimpleMovingAverage,
)
from finfeatures.features.volatility import (
    AverageTrueRange,
    BollingerBands,
    MovingTrueRange,
    RollingVolatility,
)
from finfeatures.features.volume import (
    OnBalanceVolume,
    VolumeFeatures,
)

# ===========================================================================
# Helpers
# ===========================================================================


def assert_raw_cols_preserved(original: pd.DataFrame, result: pd.DataFrame) -> None:
    for col in original.columns:
        assert col in result.columns, f"Raw column '{col}' lost"
        pd.testing.assert_series_equal(
            original[col], result[col], check_names=False, obj=f"column '{col}' was mutated"
        )


def assert_column_in_range(series: pd.Series, lo: float, hi: float) -> None:
    valid = series.dropna()
    assert (valid >= lo).all() and (valid <= hi).all(), (
        f"Values out of range [{lo}, {hi}]: min={valid.min():.4f}, max={valid.max():.4f}"
    )


# ===========================================================================
# Price features
# ===========================================================================


class TestReturns:
    def test_output_column(self, ohlcv_daily):
        out = Returns()(ohlcv_daily)
        assert Columns.RETURN in out.columns

    def test_first_row_is_nan(self, ohlcv_daily):
        out = Returns()(ohlcv_daily)
        assert np.isnan(out[Columns.RETURN].iloc[0])

    def test_values_reasonable(self, ohlcv_daily):
        out = Returns()(ohlcv_daily)
        assert_column_in_range(out[Columns.RETURN], -0.5, 0.5)

    def test_raw_preserved(self, ohlcv_daily):
        assert_raw_cols_preserved(ohlcv_daily, Returns()(ohlcv_daily))


class TestLogReturns:
    def test_output_column(self, ohlcv_daily):
        out = LogReturns()(ohlcv_daily)
        assert Columns.LOG_RETURN in out.columns

    def test_approx_equal_to_simple_for_small_returns(self, ohlcv_daily):
        out = LogReturns()(ohlcv_daily)
        r = Returns()(ohlcv_daily)
        # For small returns: log(1+r) ≈ r
        valid = ~out[Columns.LOG_RETURN].isna()
        np.testing.assert_allclose(
            out.loc[valid, Columns.LOG_RETURN],
            np.log1p(r.loc[valid, Columns.RETURN]),
            rtol=1e-6,
        )


class TestPriceRange:
    def test_output_columns(self, ohlcv_daily):
        out = PriceRange()(ohlcv_daily)
        assert "high_low_range" in out.columns
        assert "open_close_range" in out.columns
        assert "overnight_gap" in out.columns

    def test_high_low_range_positive(self, ohlcv_daily):
        out = PriceRange()(ohlcv_daily)
        assert (out["high_low_range"].dropna() >= 0).all()


class TestTypicalPrice:
    def test_output_column(self, ohlcv_daily):
        out = TypicalPrice()(ohlcv_daily)
        assert "typical_price" in out.columns

    def test_between_high_and_low(self, ohlcv_daily):
        out = TypicalPrice()(ohlcv_daily)
        tp = out["typical_price"]
        assert (tp >= ohlcv_daily["low"]).all()
        assert (tp <= ohlcv_daily["high"]).all()


class TestCumulativeReturn:
    def test_first_row_is_zero(self, ohlcv_daily):
        out = CumulativeReturn()(ohlcv_daily)
        assert out["cumulative_return"].iloc[0] == pytest.approx(0.0)

    def test_monotone_if_monotone_input(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        df = pd.DataFrame(
            {
                "open": range(1, 11),
                "high": range(2, 12),
                "low": range(1, 11),
                "close": range(1, 11),
                "volume": [1000] * 10,
            },
            index=dates,
        )
        out = CumulativeReturn()(df)
        assert (out["cumulative_return"].diff().dropna() > 0).all()


# ===========================================================================
# Volatility features
# ===========================================================================


class TestRollingVolatility:
    def test_output_column(self, ohlcv_daily):
        out = RollingVolatility(window=21)(ohlcv_daily)
        assert "realized_vol_21" in out.columns

    def test_positive_values(self, ohlcv_daily):
        out = RollingVolatility(window=21)(ohlcv_daily)
        assert (out["realized_vol_21"].dropna() > 0).all()

    def test_annualised_order_of_magnitude(self, ohlcv_daily):
        out = RollingVolatility(window=21)(ohlcv_daily)
        med = out["realized_vol_21"].median()
        # Synthetic data has σ=0.20 — annualised vol should be in [0.05, 0.80]
        assert 0.05 < med < 0.80


class TestBollingerBands:
    def test_output_columns(self, ohlcv_daily):
        out = BollingerBands(window=20)(ohlcv_daily)
        assert "bb_upper_20" in out.columns
        assert "bb_lower_20" in out.columns
        assert "bb_pct_20" in out.columns
        assert "bb_width_20" in out.columns

    def test_upper_above_lower(self, ohlcv_daily):
        out = BollingerBands(window=20)(ohlcv_daily)
        valid = out["bb_upper_20"].dropna().index
        assert (out.loc[valid, "bb_upper_20"] > out.loc[valid, "bb_lower_20"]).all()

    def test_close_mostly_within_bands(self, ohlcv_daily):
        """~95% of close prices should be within ±2σ bands."""
        out = BollingerBands(window=20)(ohlcv_daily).dropna()
        pct_b = out["bb_pct_20"]
        within = ((pct_b >= 0) & (pct_b <= 1)).mean()
        assert within > 0.88


class TestAverageTrueRange:
    def test_output_column(self, ohlcv_daily):
        out = AverageTrueRange(window=14)(ohlcv_daily)
        assert "atr_14" in out.columns
        assert "atr_pct_14" in out.columns

    def test_positive_values(self, ohlcv_daily):
        out = AverageTrueRange(window=14)(ohlcv_daily)
        assert (out["atr_14"].dropna() > 0).all()


# ===========================================================================
# Trend features
# ===========================================================================


class TestSMA:
    def test_output_columns(self, ohlcv_daily):
        out = SimpleMovingAverage(windows=[10, 20])(ohlcv_daily)
        assert "sma_10" in out.columns
        assert "sma_20" in out.columns
        assert "close_sma_10_ratio" in out.columns

    def test_sma_1_equals_close(self, ohlcv_daily):
        out = SimpleMovingAverage(windows=[1])(ohlcv_daily)
        pd.testing.assert_series_equal(out["sma_1"], ohlcv_daily["close"], check_names=False)

    def test_200_sma_has_199_leading_nans(self, ohlcv_daily):
        out = SimpleMovingAverage(windows=[200])(ohlcv_daily)
        assert out["sma_200"].isna().sum() == 199


class TestEMA:
    def test_output_columns(self, ohlcv_daily):
        out = ExponentialMovingAverage(windows=[12, 26])(ohlcv_daily)
        assert "ema_12" in out.columns
        assert "ema_26" in out.columns


class TestMACD:
    def test_output_columns(self, ohlcv_daily):
        out = MACD()(ohlcv_daily)
        for col in ["macd_line", "macd_signal", "macd_hist"]:
            assert col in out.columns

    def test_histogram_is_line_minus_signal(self, ohlcv_daily):
        out = MACD()(ohlcv_daily)
        diff = (out["macd_line"] - out["macd_signal"] - out["macd_hist"]).dropna()
        np.testing.assert_allclose(diff.values, 0, atol=1e-10)


# ===========================================================================
# Momentum features
# ===========================================================================


class TestRSI:
    def test_output_column(self, ohlcv_daily):
        out = RSI(window=14)(ohlcv_daily)
        assert "rsi_14" in out.columns

    def test_bounded_0_100(self, ohlcv_daily):
        out = RSI(window=14)(ohlcv_daily)
        assert_column_in_range(out["rsi_14"], 0, 100)


class TestRateOfChange:
    def test_output_column(self, ohlcv_daily):
        out = RateOfChange(window=10)(ohlcv_daily)
        assert "roc_10" in out.columns

    def test_leading_nans(self, ohlcv_daily):
        out = RateOfChange(window=10)(ohlcv_daily)
        assert out["roc_10"].isna().sum() == 10


class TestStochastic:
    def test_output_columns(self, ohlcv_daily):
        out = StochasticOscillator(k_window=14, d_window=3)(ohlcv_daily)
        assert "stoch_k_14" in out.columns
        assert "stoch_d_14" in out.columns

    def test_bounded_0_100(self, ohlcv_daily):
        out = StochasticOscillator()(ohlcv_daily)
        assert_column_in_range(out["stoch_k_14"], 0, 100)


# ===========================================================================
# Volume features
# ===========================================================================


class TestVolumeFeatures:
    def test_output_columns(self, ohlcv_daily):
        out = VolumeFeatures(window=20)(ohlcv_daily)
        assert "volume_return" in out.columns
        assert "volume_zscore_20" in out.columns
        assert "volume_rel_20" in out.columns


class TestOBV:
    def test_output_column(self, ohlcv_daily):
        out = OnBalanceVolume()(ohlcv_daily)
        assert "obv" in out.columns

    def test_monotone_on_rising_prices(self):
        """OBV should increase every day when prices always rise."""
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        close = np.arange(1.0, 11.0)
        df = pd.DataFrame(
            {
                "open": close - 0.1,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": np.ones(10) * 1000,
            },
            index=dates,
        )
        out = OnBalanceVolume()(df)
        # After first row, OBV should be strictly increasing
        assert (out["obv"].diff().dropna() > 0).all()


# ===========================================================================
# Statistical features
# ===========================================================================


class TestRollingZScore:
    def test_output_column(self, ohlcv_daily):
        from finfeatures.features.price import LogReturns

        df = LogReturns()(ohlcv_daily)
        out = RollingZScore(column="log_return", window=21)(df)
        assert "log_return_zscore_21" in out.columns

    def test_mean_near_zero(self, ohlcv_daily):
        from finfeatures.features.price import LogReturns

        df = LogReturns()(ohlcv_daily)
        out = RollingZScore(column="log_return", window=21)(df)
        mean = out["log_return_zscore_21"].dropna().mean()
        assert abs(mean) < 0.5  # z-scores should be centred


class TestRollingSkewKurt:
    def test_output_columns(self, ohlcv_daily):
        from finfeatures.features.price import LogReturns

        df = LogReturns()(ohlcv_daily)
        out = RollingSkewKurt(column="log_return", window=63)(df)
        assert "log_return_skew_63" in out.columns
        assert "log_return_kurt_63" in out.columns


# ===========================================================================
# Composite features
# ===========================================================================


class TestDrawdownFeatures:
    def test_output_columns(self, ohlcv_daily):
        out = DrawdownFeatures()(ohlcv_daily)
        assert "drawdown" in out.columns
        assert "drawdown_duration" in out.columns
        assert "drawdown_recovery" in out.columns

    def test_drawdown_non_positive(self, ohlcv_daily):
        out = DrawdownFeatures()(ohlcv_daily)
        assert (out["drawdown"] <= 1e-9).all()

    def test_recovery_bounded_0_1(self, ohlcv_daily):
        out = DrawdownFeatures()(ohlcv_daily)
        valid = out["drawdown_recovery"].dropna()
        assert (valid >= 0).all() and (valid <= 1 + 1e-9).all()


class TestDistributionShiftScore:
    def test_output_column(self, ohlcv_daily):
        from finfeatures.features.price import LogReturns

        df = LogReturns()(ohlcv_daily)
        out = DistributionShiftScore(column="log_return", window=21)(df)
        assert "dist_shift_log_return_21" in out.columns

    def test_non_negative(self, ohlcv_daily):
        from finfeatures.features.price import LogReturns

        df = LogReturns()(ohlcv_daily)
        out = DistributionShiftScore(column="log_return", window=21)(df)
        valid = out["dist_shift_log_return_21"].dropna()
        assert (valid >= 0).all()


# ===========================================================================
# Log-space / candle shape features
# ===========================================================================


class TestLogTransform:
    def test_output_columns(self, ohlcv_daily):
        out = LogTransform()(ohlcv_daily)
        for col in ["log_open", "log_high", "log_low", "log_close", "log_volume"]:
            assert col in out.columns

    def test_log_close_correct(self, ohlcv_daily):
        out = LogTransform()(ohlcv_daily)
        np.testing.assert_allclose(
            out["log_close"].values, np.log(ohlcv_daily["close"].values), rtol=1e-10
        )

    def test_log_volume_uses_log1p(self, ohlcv_daily):
        out = LogTransform()(ohlcv_daily)
        np.testing.assert_allclose(
            out["log_volume"].values, np.log1p(ohlcv_daily["volume"].values), rtol=1e-10
        )

    def test_raw_preserved(self, ohlcv_daily):
        assert_raw_cols_preserved(ohlcv_daily, LogTransform()(ohlcv_daily))


class TestCandleShape:
    @pytest.fixture()
    def log_df(self, ohlcv_daily):
        return LogTransform()(ohlcv_daily)

    def test_output_columns(self, log_df):
        out = CandleShape()(log_df)
        for col in ["body", "range", "upper_wick", "lower_wick", "CLV"]:
            assert col in out.columns

    def test_range_non_negative(self, log_df):
        out = CandleShape()(log_df)
        assert (out["range"].dropna() >= 0).all()

    def test_clv_bounded(self, log_df):
        out = CandleShape()(log_df)
        valid = out["CLV"].dropna()
        assert (valid >= -0.01).all() and (valid <= 1.01).all()

    def test_body_sign_matches_direction(self, log_df):
        out = CandleShape()(log_df)
        bullish = log_df["log_close"] > log_df["log_open"]
        assert (out.loc[bullish, "body"] >= 0).all()


class TestCrossDay:
    @pytest.fixture()
    def log_df(self, ohlcv_daily):
        return LogTransform()(ohlcv_daily)

    def test_output_columns(self, log_df):
        out = CrossDay()(log_df)
        for col in ["overnight", "C_minus_yH", "C_minus_yL", "O_minus_yH", "O_minus_yL"]:
            assert col in out.columns

    def test_first_row_nan(self, log_df):
        out = CrossDay()(log_df)
        assert np.isnan(out["C_minus_yH"].iloc[0])

    def test_raw_preserved(self, log_df):
        assert_raw_cols_preserved(log_df, CrossDay()(log_df))


class TestShapeDynamics:
    @pytest.fixture()
    def shape_df(self, ohlcv_daily):
        df = LogTransform()(ohlcv_daily)
        return CandleShape()(df)

    def test_output_columns(self, shape_df):
        out = ShapeDynamics()(shape_df)
        for col in ["d_body", "d_range", "d_upper_wick", "d_lower_wick", "d_CLV", "d_log_vol"]:
            assert col in out.columns

    def test_first_row_nan(self, shape_df):
        out = ShapeDynamics()(shape_df)
        assert np.isnan(out["d_body"].iloc[0])

    def test_d_body_is_diff_of_body(self, shape_df):
        out = ShapeDynamics()(shape_df)
        expected = shape_df["body"].diff()
        pd.testing.assert_series_equal(out["d_body"], expected, check_names=False)


# ===========================================================================
# Moving True Range & non-annualized vol
# ===========================================================================


class TestMovingTrueRange:
    def test_output_columns(self, ohlcv_daily):
        out = MovingTrueRange(windows=[20, 50])(ohlcv_daily)
        assert "mtr_20" in out.columns
        assert "mtr_50" in out.columns

    def test_positive_values(self, ohlcv_daily):
        out = MovingTrueRange(windows=[20])(ohlcv_daily)
        assert (out["mtr_20"].dropna() > 0).all()

    def test_leading_nans(self, ohlcv_daily):
        out = MovingTrueRange(windows=[20])(ohlcv_daily)
        # rolling(20) on true range produces 19 leading NaNs
        assert out["mtr_20"].isna().sum() >= 19


class TestRollingVolatilityRaw:
    def test_raw_output_column(self, ohlcv_daily):
        out = RollingVolatility(window=21, annualize=False)(ohlcv_daily)
        assert "raw_vol_21" in out.columns
        assert "realized_vol_21" not in out.columns

    def test_raw_vs_annualized(self, ohlcv_daily):
        raw = RollingVolatility(window=21, annualize=False)(ohlcv_daily)
        ann = RollingVolatility(window=21, annualize=True)(ohlcv_daily)
        ratio = ann["realized_vol_21"].dropna() / raw["raw_vol_21"].dropna()
        np.testing.assert_allclose(ratio.values, np.sqrt(252), rtol=1e-10)
