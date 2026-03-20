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
    CompositeScores,
    DistributionShiftScore,
    DrawdownFeatures,
)
from finfeatures.features.momentum import (
    PPO,
    RSI,
    TRIX,
    Aroon,
    ChandeMomentumOscillator,
    MoneyFlowIndex,
    RateOfChange,
    StochasticOscillator,
    StochasticRSI,
    UltimateOscillator,
)
from finfeatures.features.patterns import CandlePatterns
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
    CrossAssetCorrelation,
    LinearRegressionSlope,
    RollingSkewKurt,
    RollingZScore,
)
from finfeatures.features.trend import (
    DEMA,
    KAMA,
    MACD,
    TEMA,
    DonchianChannels,
    ExponentialMovingAverage,
    IchimokuCloud,
    ParabolicSAR,
    SimpleMovingAverage,
    Supertrend,
)
from finfeatures.features.volatility import (
    AverageTrueRange,
    BollingerBands,
    KeltnerChannels,
    MovingTrueRange,
    RollingVolatility,
)
from finfeatures.features.volume import (
    AccumulationDistribution,
    ChaikinADOscillator,
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
        assert within > 0.87


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


# ===========================================================================
# Parameter validation (Item 3)
# ===========================================================================


class TestParameterValidation:
    def test_window_zero_raises(self):
        with pytest.raises(ValueError, match="integer >= 1"):
            RollingVolatility(window=0)

    def test_window_negative_raises(self):
        with pytest.raises(ValueError, match="integer >= 1"):
            RSI(window=-5)

    def test_macd_fast_ge_slow_raises(self):
        with pytest.raises(ValueError, match="fast.*<.*slow"):
            MACD(fast=26, slow=12)


# ===========================================================================
# safe_divide edge cases (Item 2)
# ===========================================================================


class TestSafeDivideEdgeCases:
    def test_bollinger_flat_price_no_inf(self):
        """Constant close should produce NaN, not inf, for bb_pct and bb_width."""
        dates = pd.date_range("2020-01-01", periods=50, freq="B")
        df = pd.DataFrame(
            {
                "open": [100.0] * 50,
                "high": [100.0] * 50,
                "low": [100.0] * 50,
                "close": [100.0] * 50,
                "volume": [1000] * 50,
            },
            index=dates,
        )
        out = BollingerBands(window=20)(df)
        assert not np.isinf(out["bb_pct_20"].dropna()).any()
        assert not np.isinf(out["bb_width_20"].dropna()).any()

    def test_stochastic_flat_range_no_inf(self):
        """When high == low, stochastic should produce NaN, not inf."""
        dates = pd.date_range("2020-01-01", periods=30, freq="B")
        df = pd.DataFrame(
            {
                "open": [50.0] * 30,
                "high": [50.0] * 30,
                "low": [50.0] * 30,
                "close": [50.0] * 30,
                "volume": [1000] * 30,
            },
            index=dates,
        )
        out = StochasticOscillator(k_window=14)(df)
        assert not np.isinf(out["stoch_k_14"].dropna()).any()


# ===========================================================================
# CompositeScores custom columns (Item 6)
# ===========================================================================


class TestCompositeScoresCustomColumns:
    def test_custom_column_names(self, ohlcv_daily):
        """CompositeScores should work with non-default column names."""
        # Build a df with non-default column names for the required inputs
        from finfeatures.features.price import Returns

        out = Returns()(ohlcv_daily)
        out["my_vol"] = out["close"].rolling(21).std()
        out["my_macd"] = out["close"].ewm(span=12).mean() - out["close"].ewm(span=26).mean()
        out["my_rsi"] = 50.0  # constant RSI for simplicity

        cs = CompositeScores(vol_col="my_vol", macd_col="my_macd", rsi_col="my_rsi")
        result = cs(out)
        assert "stress_score" in result.columns
        assert "trend_score" in result.columns
        assert "momentum_score_indicator" in result.columns


# ===========================================================================
# New trend features (PR 3)
# ===========================================================================


class TestKAMA:
    def test_output_column(self, ohlcv_daily):
        out = KAMA(window=10)(ohlcv_daily)
        assert "kama_10" in out.columns

    def test_leading_nans(self, ohlcv_daily):
        out = KAMA(window=10)(ohlcv_daily)
        assert out["kama_10"].isna().sum() >= 9

    def test_raw_preserved(self, ohlcv_daily):
        assert_raw_cols_preserved(ohlcv_daily, KAMA()(ohlcv_daily))


class TestParabolicSAR:
    def test_output_column(self, ohlcv_daily):
        out = ParabolicSAR()(ohlcv_daily)
        assert "sar" in out.columns

    def test_values_in_price_range(self, ohlcv_daily):
        out = ParabolicSAR()(ohlcv_daily)
        sar = out["sar"].dropna()
        # SAR should be in the vicinity of prices
        assert sar.min() > ohlcv_daily["low"].min() * 0.5
        assert sar.max() < ohlcv_daily["high"].max() * 1.5

    def test_raw_preserved(self, ohlcv_daily):
        assert_raw_cols_preserved(ohlcv_daily, ParabolicSAR()(ohlcv_daily))


class TestDEMA:
    def test_output_columns(self, ohlcv_daily):
        out = DEMA(windows=[10, 20])(ohlcv_daily)
        assert "dema_10" in out.columns
        assert "dema_20" in out.columns

    def test_raw_preserved(self, ohlcv_daily):
        assert_raw_cols_preserved(ohlcv_daily, DEMA()(ohlcv_daily))


class TestTEMA:
    def test_output_columns(self, ohlcv_daily):
        out = TEMA(windows=[10, 20])(ohlcv_daily)
        assert "tema_10" in out.columns
        assert "tema_20" in out.columns

    def test_raw_preserved(self, ohlcv_daily):
        assert_raw_cols_preserved(ohlcv_daily, TEMA()(ohlcv_daily))


class TestIchimokuCloud:
    def test_output_columns(self, ohlcv_daily):
        out = IchimokuCloud()(ohlcv_daily)
        for col in ["tenkan_sen", "kijun_sen", "senkou_a", "senkou_b", "chikou_span"]:
            assert col in out.columns

    def test_tenkan_between_high_low(self, ohlcv_daily):
        out = IchimokuCloud()(ohlcv_daily)
        valid = out["tenkan_sen"].dropna()
        assert (valid >= ohlcv_daily.loc[valid.index, "low"].min()).all()

    def test_raw_preserved(self, ohlcv_daily):
        assert_raw_cols_preserved(ohlcv_daily, IchimokuCloud()(ohlcv_daily))


class TestDonchianChannels:
    def test_output_columns(self, ohlcv_daily):
        out = DonchianChannels(window=20)(ohlcv_daily)
        assert "donchian_upper_20" in out.columns
        assert "donchian_lower_20" in out.columns
        assert "donchian_mid_20" in out.columns
        assert "donchian_width_20" in out.columns

    def test_upper_above_lower(self, ohlcv_daily):
        out = DonchianChannels(window=20)(ohlcv_daily)
        valid = out["donchian_upper_20"].dropna().index
        assert (out.loc[valid, "donchian_upper_20"] >= out.loc[valid, "donchian_lower_20"]).all()

    def test_raw_preserved(self, ohlcv_daily):
        assert_raw_cols_preserved(ohlcv_daily, DonchianChannels()(ohlcv_daily))


class TestSupertrend:
    def test_output_columns(self, ohlcv_daily):
        out = Supertrend()(ohlcv_daily)
        assert "supertrend" in out.columns
        assert "supertrend_dir" in out.columns

    def test_direction_values(self, ohlcv_daily):
        out = Supertrend()(ohlcv_daily)
        valid = out["supertrend_dir"].dropna()
        assert set(valid.unique()).issubset({-1.0, 1.0})

    def test_raw_preserved(self, ohlcv_daily):
        assert_raw_cols_preserved(ohlcv_daily, Supertrend()(ohlcv_daily))


# ===========================================================================
# New momentum features (PR 3)
# ===========================================================================


class TestMoneyFlowIndex:
    def test_output_column(self, ohlcv_daily):
        out = MoneyFlowIndex(window=14)(ohlcv_daily)
        assert "mfi_14" in out.columns

    def test_bounded_0_100(self, ohlcv_daily):
        out = MoneyFlowIndex(window=14)(ohlcv_daily)
        assert_column_in_range(out["mfi_14"], 0, 100)

    def test_raw_preserved(self, ohlcv_daily):
        assert_raw_cols_preserved(ohlcv_daily, MoneyFlowIndex()(ohlcv_daily))


class TestAroon:
    def test_output_columns(self, ohlcv_daily):
        out = Aroon(window=25)(ohlcv_daily)
        assert "aroon_up_25" in out.columns
        assert "aroon_down_25" in out.columns
        assert "aroon_osc_25" in out.columns

    def test_up_down_bounded_0_100(self, ohlcv_daily):
        out = Aroon(window=25)(ohlcv_daily)
        assert_column_in_range(out["aroon_up_25"], 0, 100)
        assert_column_in_range(out["aroon_down_25"], 0, 100)

    def test_osc_bounded_neg100_100(self, ohlcv_daily):
        out = Aroon(window=25)(ohlcv_daily)
        assert_column_in_range(out["aroon_osc_25"], -100, 100)


class TestChandeMomentumOscillator:
    def test_output_column(self, ohlcv_daily):
        out = ChandeMomentumOscillator(window=14)(ohlcv_daily)
        assert "cmo_14" in out.columns

    def test_bounded_neg100_100(self, ohlcv_daily):
        out = ChandeMomentumOscillator(window=14)(ohlcv_daily)
        assert_column_in_range(out["cmo_14"], -100, 100)


class TestUltimateOscillator:
    def test_output_column(self, ohlcv_daily):
        out = UltimateOscillator()(ohlcv_daily)
        assert "ultimate_osc" in out.columns

    def test_bounded_0_100(self, ohlcv_daily):
        out = UltimateOscillator()(ohlcv_daily)
        assert_column_in_range(out["ultimate_osc"], 0, 100)


class TestStochasticRSI:
    def test_output_columns(self, ohlcv_daily):
        out = StochasticRSI(window=14)(ohlcv_daily)
        assert "stochrsi_k_14" in out.columns
        assert "stochrsi_d_14" in out.columns

    def test_bounded_0_100(self, ohlcv_daily):
        out = StochasticRSI(window=14)(ohlcv_daily)
        # Allow tiny floating-point overshoot
        assert_column_in_range(out["stochrsi_k_14"], -0.01, 100.01)
        assert_column_in_range(out["stochrsi_d_14"], -0.01, 100.01)

    def test_raw_preserved(self, ohlcv_daily):
        assert_raw_cols_preserved(ohlcv_daily, StochasticRSI()(ohlcv_daily))


class TestTRIX:
    def test_output_column(self, ohlcv_daily):
        out = TRIX(window=30)(ohlcv_daily)
        assert "trix_30" in out.columns

    def test_oscillates_around_zero(self, ohlcv_daily):
        out = TRIX(window=15)(ohlcv_daily)
        valid = out["trix_15"].dropna()
        assert valid.min() < 0
        assert valid.max() > 0

    def test_raw_preserved(self, ohlcv_daily):
        assert_raw_cols_preserved(ohlcv_daily, TRIX()(ohlcv_daily))


class TestPPO:
    def test_output_column(self, ohlcv_daily):
        out = PPO(fast=12, slow=26)(ohlcv_daily)
        assert "ppo_12_26" in out.columns

    def test_fast_ge_slow_raises(self):
        with pytest.raises(ValueError, match="fast.*<.*slow"):
            PPO(fast=26, slow=12)

    def test_raw_preserved(self, ohlcv_daily):
        assert_raw_cols_preserved(ohlcv_daily, PPO()(ohlcv_daily))


class TestKeltnerChannels:
    def test_output_columns(self, ohlcv_daily):
        out = KeltnerChannels(window=20)(ohlcv_daily)
        assert "keltner_upper_20" in out.columns
        assert "keltner_mid_20" in out.columns
        assert "keltner_lower_20" in out.columns
        assert "keltner_pct_20" in out.columns

    def test_upper_above_lower(self, ohlcv_daily):
        out = KeltnerChannels(window=20)(ohlcv_daily)
        valid = out["keltner_upper_20"].dropna().index
        assert (out.loc[valid, "keltner_upper_20"] > out.loc[valid, "keltner_lower_20"]).all()

    def test_raw_preserved(self, ohlcv_daily):
        assert_raw_cols_preserved(ohlcv_daily, KeltnerChannels()(ohlcv_daily))


# ===========================================================================
# New volume features (PR 3)
# ===========================================================================


class TestAccumulationDistribution:
    def test_output_column(self, ohlcv_daily):
        out = AccumulationDistribution()(ohlcv_daily)
        assert "ad_line" in out.columns

    def test_raw_preserved(self, ohlcv_daily):
        assert_raw_cols_preserved(ohlcv_daily, AccumulationDistribution()(ohlcv_daily))


class TestChaikinADOscillator:
    def test_output_column(self, ohlcv_daily):
        out = ChaikinADOscillator(fast=3, slow=10)(ohlcv_daily)
        assert "adosc_3_10" in out.columns

    def test_raw_preserved(self, ohlcv_daily):
        assert_raw_cols_preserved(ohlcv_daily, ChaikinADOscillator()(ohlcv_daily))


# ===========================================================================
# New statistical features (PR 3)
# ===========================================================================


class TestLinearRegressionSlope:
    def test_output_column(self, ohlcv_daily):
        out = LinearRegressionSlope(column="close", window=14)(ohlcv_daily)
        assert "close_linreg_slope_14" in out.columns

    def test_leading_nans(self, ohlcv_daily):
        out = LinearRegressionSlope(column="close", window=14)(ohlcv_daily)
        assert out["close_linreg_slope_14"].isna().sum() >= 13

    def test_positive_slope_on_rising_prices(self):
        dates = pd.date_range("2020-01-01", periods=30, freq="B")
        df = pd.DataFrame(
            {
                "open": range(1, 31),
                "high": range(2, 32),
                "low": range(1, 31),
                "close": range(1, 31),
                "volume": [1000] * 30,
            },
            index=dates,
        )
        out = LinearRegressionSlope(column="close", window=14)(df)
        slopes = out["close_linreg_slope_14"].dropna()
        assert (slopes > 0).all()


# ===========================================================================
# Pattern features (PR 3)
# ===========================================================================


class TestCandlePatterns:
    def test_output_columns(self, ohlcv_daily):
        out = CandlePatterns()(ohlcv_daily)
        # Should have at least the pandas patterns
        pattern_cols = [c for c in out.columns if c.startswith("cdl")]
        assert len(pattern_cols) >= 3

    def test_values_are_integer_signals(self, ohlcv_daily):
        out = CandlePatterns()(ohlcv_daily)
        cdl_cols = [c for c in out.columns if c.startswith("cdl")]
        for col in cdl_cols:
            vals = out[col].dropna().unique()
            # Values should be integers: typically -100, 0, 100
            for v in vals:
                assert v == int(v), f"Non-integer value {v} in {col}"

    def test_raw_preserved(self, ohlcv_daily):
        assert_raw_cols_preserved(ohlcv_daily, CandlePatterns()(ohlcv_daily))


# ===========================================================================
# Cross-asset correlation
# ===========================================================================


class TestCrossAssetCorrelation:
    @pytest.fixture()
    def reference_df(self):
        """Synthetic reference asset with same date range as ohlcv_daily."""
        rng = np.random.default_rng(99)
        n = 500
        dates = pd.date_range("2020-01-02", periods=n, freq="B")
        close = 50.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        return pd.DataFrame(
            {
                "open": close * (1 + rng.uniform(-0.005, 0.005, n)),
                "high": close * (1 + rng.uniform(0.002, 0.01, n)),
                "low": close * (1 - rng.uniform(0.002, 0.01, n)),
                "close": close,
                "volume": rng.lognormal(14, 0.5, n).astype(int),
            },
            index=dates,
        )

    @pytest.mark.parametrize("column", ["open", "high", "low", "close"])
    def test_ohlc_columns(self, ohlcv_daily, reference_df, column):
        feat = CrossAssetCorrelation(
            reference=reference_df, reference_name="spy", column=column, windows=7
        )
        out = feat(ohlcv_daily)
        col_name = f"corr_spy_{column}_7"
        assert col_name in out.columns
        assert_column_in_range(out[col_name], -1, 1)

    def test_multiple_windows(self, ohlcv_daily, reference_df):
        feat = CrossAssetCorrelation(
            reference=reference_df, reference_name="spy", column="close", windows=[7, 21]
        )
        out = feat(ohlcv_daily)
        assert "corr_spy_close_7" in out.columns
        assert "corr_spy_close_21" in out.columns

    def test_different_columns_naming(self, ohlcv_daily, reference_df):
        feat = CrossAssetCorrelation(
            reference=reference_df,
            reference_name="spy",
            column="high",
            reference_column="low",
            windows=7,
        )
        out = feat(ohlcv_daily)
        assert "corr_spy_high_vs_low_7" in out.columns

    def test_perfect_self_correlation(self, ohlcv_daily):
        """Correlating a DataFrame with itself should give ~1.0."""
        feat = CrossAssetCorrelation(
            reference=ohlcv_daily, reference_name="self", column="close", windows=21
        )
        out = feat(ohlcv_daily)
        valid = out["corr_self_close_21"].dropna()
        assert (valid > 0.999).all()

    def test_missing_reference_column_raises(self, reference_df):
        with pytest.raises(ValueError, match="reference_column"):
            CrossAssetCorrelation(
                reference=reference_df,
                reference_name="spy",
                column="close",
                reference_column="nonexistent",
                windows=7,
            )

    def test_raw_preserved(self, ohlcv_daily, reference_df):
        feat = CrossAssetCorrelation(
            reference=reference_df, reference_name="spy", column="close", windows=7
        )
        assert_raw_cols_preserved(ohlcv_daily, feat(ohlcv_daily))
