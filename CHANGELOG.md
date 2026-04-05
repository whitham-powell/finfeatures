# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-04-04

### Added
- **HurstExponent** — rolling Hurst exponent via Rescaled Range (R/S) analysis for
  regime detection (H > 0.5 trending, H ≈ 0.5 random walk, H < 0.5 mean-reverting).
- **WeightedMovingAverage (WMA)** — linearly weighted MA with TA-Lib integration.
- **VolumeWeightedMovingAverage (VWMA)** — `sum(close × volume) / sum(volume)` over
  a rolling window.
- **TrueRange** — raw per-bar true range (unsmoothed) with TA-Lib integration.
  Useful for breakout detection and as input to custom indicators.
- GitHub issue templates for feature requests and bug reports.

### Fixed
- **Non-positive price validation** — `FeaturePipeline.transform()` now rejects
  DataFrames containing zero or negative OHLC prices at the pipeline boundary,
  preventing silent NaN propagation through `log()` and `safe_divide()` downstream.

## [0.2.0] — 2026-03-19

### Added
- **TA-Lib optional integration** — `pip install finfeatures[talib]` adds TA-Lib as
  a speed backend. All indicators have pure-pandas fallbacks; TA-Lib is never required.
  Both paths produce bit-identical output (verified at `1e-10` tolerance across 27
  equivalence tests).
- **20 new Feature classes** (40 → 60 total registered features):
  - Trend: KAMA, ParabolicSAR, DEMA, TEMA, IchimokuCloud, DonchianChannels, Supertrend
  - Momentum: MoneyFlowIndex, Aroon, ChandeMomentumOscillator, UltimateOscillator,
    StochasticRSI, TRIX, PPO
  - Volume: AccumulationDistribution, ChaikinADOscillator
  - Volatility: KeltnerChannels
  - Statistical: LinearRegressionSlope, CrossAssetCorrelation
  - Patterns: CandlePatterns (61 patterns via TA-Lib, 6 in pure pandas)
- **CrossAssetCorrelation** — rolling Pearson correlation against an external reference
  asset's DataFrame. Supports arbitrary columns, multiple windows, and automatic
  date alignment via forward-fill. Uses `talib.CORREL` when available.
- **`talib_pipeline()`** preset — showcases new indicators, composable with
  `standard_pipeline()`.
- **CI dual matrix** — tests run with and without TA-Lib installed (Python 3.12).
- `_sma_seeded_ema` and `_wilder_smooth` helpers in `core/base.py`
- `_compat.py` module with `HAS_TALIB` detection and `_f64()` array converter
- `skip_without_talib` and `force_no_talib` test fixtures
- pytest `talib` marker for conditional test execution

### Changed
- **Mathematical parity** — pandas fallback paths now use SMA-seeded EMA and Wilder
  smoothing, matching TA-Lib's (and Wilder's original published) initialization.
  Bollinger Bands use population std (`ddof=0`). Previously, EWM-based indicators
  diverged from TA-Lib during the warm-up region due to different seeding.
- CandlePatterns warns at compute time when TA-Lib is absent (6 of 61 patterns).

### Removed
- PyPI publish workflow — removed until the project is ready for public distribution.
  Tags can still be used for internal versioning without triggering any publish action.

## [0.1.0]

### Added
- `Feature` base class with auto-registration via `FeatureRegistry`
- `FeaturePipeline` — ordered, composable feature composition
- `DataSource` ABC — swap any market data provider
- Feature modules: `price`, `volatility`, `trend`, `momentum`, `volume`,
  `statistical`, `composite`
- `YFinanceSource` reference implementation (optional dependency)
- `PandasAdapter` portability wrapper
- Preset pipelines: `minimal_pipeline`, `standard_pipeline`, `extended_pipeline`
