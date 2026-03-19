# finfeatures

[![CI](https://github.com/whitham-powell/finfeatures/actions/workflows/ci.yml/badge.svg)](https://github.com/whitham-powell/finfeatures/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A source-agnostic, pipeline-based Python library for deriving features from
raw financial OHLCV timeseries data.

**Raw source columns are never lost.** All feature transforms are additive —
the output DataFrame always contains the original data plus any derived columns.

---

## Design

```
DataSource  →  raw OHLCV DataFrame    (open, high, low, close, volume)
                    ↓
Feature         pure, additive transform:  DataFrame → DataFrame
                    ↓
FeaturePipeline ordered composition of Features
                    ↓
PandasAdapter   thin portability boundary
                    ↓
Downstream      regime detector / backtester / ML model / …
```

The library has **no opinion on what you do with the features**. It only
answers one question: *given a raw OHLCV DataFrame, what derived columns
would you like?*

---

## Installation

```bash
# Core library (pandas, numpy, scipy)
uv add finfeatures

# With the yfinance data source helper
uv add "finfeatures[yfinance]"
```

### Using a local / GitHub checkout

```bash
# From PyPI (once published)
uv add finfeatures

# From GitHub main
uv add "finfeatures @ git+https://github.com/whitham-powell/finfeatures"

# From a specific tag / commit
uv add "finfeatures @ git+https://github.com/whitham-powell/finfeatures@v0.1.0"

# Local editable (development of finfeatures itself)
uv add --editable ./finfeatures
```

---

## Quickstart

```python
import pandas as pd
from finfeatures import standard_pipeline

# 1. Load raw OHLCV data (any source — CSV, database, API, …)
raw = pd.read_csv("spy.csv", index_col="date", parse_dates=True)
# expected columns: open, high, low, close, volume  (DatetimeIndex)

# 2. Build features
enriched = standard_pipeline().transform(raw)
# enriched: 40+ columns; raw columns always present

# 3. Feed downstream
feature_matrix = enriched.dropna()
```

### Using yfinance (optional)

```python
from finfeatures.sources.yfinance import YFinanceSource

source = YFinanceSource()
raw = source.fetch("SPY", start="2020-01-01", end="2024-12-31")
enriched = standard_pipeline().transform(raw)
```

---

## Feature inventory

| Module | Feature | Output column(s) |
|--------|---------|-----------------|
| `price` | `Returns` | `return` |
| `price` | `LogReturns` | `log_return` |
| `price` | `PriceRange` | `high_low_range`, `open_close_range`, `overnight_gap` |
| `price` | `TypicalPrice` | `typical_price` |
| `price` | `CumulativeReturn` | `cumulative_return` |
| `price` | `PriceRelativeToHigh` | `pct_from_high_N`, `pct_from_low_N` |
| `volatility` | `RollingVolatility` | `realized_vol_N` |
| `volatility` | `ParkinsonVolatility` | `parkinson_vol_N` |
| `volatility` | `GarmanKlassVolatility` | `garman_klass_vol_N` |
| `volatility` | `BollingerBands` | `bb_upper_N`, `bb_lower_N`, `bb_pct_N`, `bb_width_N` |
| `volatility` | `AverageTrueRange` | `atr_N`, `atr_pct_N` |
| `volatility` | `VolatilityRatio` | `vol_ratio`, `vol_ratio_zscore` |
| `trend` | `SimpleMovingAverage` | `sma_N`, `close_sma_N_ratio` |
| `trend` | `ExponentialMovingAverage` | `ema_N` |
| `trend` | `MACD` | `macd_line`, `macd_signal`, `macd_hist` (+ `_pct` variants) |
| `trend` | `TrendStrength` | `adx_N`, `di_plus`, `di_minus` |
| `trend` | `MACrossover` | `ma_cross_F_S`, `ma_cross_sign_F_S` |
| `momentum` | `RSI` | `rsi_N` |
| `momentum` | `RateOfChange` | `roc_N` |
| `momentum` | `StochasticOscillator` | `stoch_k_N`, `stoch_d_N` |
| `momentum` | `WilliamsR` | `williams_r_N` |
| `momentum` | `CommodityChannelIndex` | `cci_N` |
| `momentum` | `MomentumScore` | `mom_ret_N`, `mom_zscore_N`, `momentum_composite` |
| `volume` | `VolumeFeatures` | `volume_return`, `volume_zscore_N`, `volume_rel_N` |
| `volume` | `OnBalanceVolume` | `obv` |
| `volume` | `VWAP` | `vwap_N`, `vwap_ratio_N` |
| `volume` | `ChaikinMoneyFlow` | `cmf_N` |
| `statistical` | `RollingZScore` | `{col}_zscore_N` |
| `statistical` | `RollingSkewKurt` | `{col}_skew_N`, `{col}_kurt_N` |
| `statistical` | `RollingMoments` | mean, std, skew, kurt, VaR5, CVaR5 |
| `statistical` | `RollingAutocorrelation` | `{col}_autocorr_lagK_N` |
| `statistical` | `RollingCorrelation` | `corr_{colA}_{colB}_N` |
| `composite` | `DistributionShiftScore` | `dist_shift_{col}_N` |
| `composite` | `DrawdownFeatures` | `drawdown`, `drawdown_duration`, `drawdown_recovery` |
| `composite` | `CompositeScores` | `stress_score`, `trend_score`, `momentum_score_indicator` |

---

## Preset pipelines

```python
from finfeatures import minimal_pipeline, standard_pipeline, extended_pipeline

# Returns + log returns only
p = minimal_pipeline()

# Full baseline: price → vol → trend → momentum → volume → statistical  (~40 cols)
p = standard_pipeline()

# standard + drawdown + JS-divergence shift + composite scores
p = extended_pipeline()

# Compose
p = minimal_pipeline() + standard_pipeline()

# Extend with a single feature
from finfeatures.features.volatility import ParkinsonVolatility
p = standard_pipeline().add(ParkinsonVolatility(window=21))
```

---

## Custom features

Subclass `Feature`, set a unique `name`, and it auto-registers:

```python
from finfeatures.core import Feature, Columns, FeaturePipeline
import pandas as pd

class WeeklyMomentum(Feature):
    name = "weekly_momentum"              # unique slug — auto-registers
    required_cols = [Columns.CLOSE]
    description = "5-day momentum"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()                   # never mutate input
        out["weekly_mom"] = df[Columns.CLOSE].pct_change(5)
        return out                        # raw cols preserved automatically

pipeline = standard_pipeline().add(WeeklyMomentum())
enriched = pipeline.transform(raw)
```

---

## Custom data sources

Subclass `DataSource` to integrate any market data provider:

```python
from finfeatures.core import DataSource, Columns
import pandas as pd

class MyAPISource(DataSource):
    def __init__(self, api_key: str): self.api_key = api_key

    def fetch(self, symbol, start=None, end=None, interval="1d", **kw):
        df = ...  # call your API
        return df  # must return lowercase open/high/low/close/volume + DatetimeIndex

    def fetch_multiple(self, symbols, **kw):
        return {s: self.fetch(s, **kw) for s in symbols}

source = MyAPISource("key")
raw = source.fetch("AAPL")
```

---

## Multi-asset pipelines

```python
data = source.fetch_multiple(["SPY", "QQQ", "IWM"], start="2020-01-01")
results = extended_pipeline().transform_many(data)
# {"SPY": enriched_df, "QQQ": enriched_df, "IWM": enriched_df}
```

---

## Repository layout

```
finfeatures/
├── .github/
│   └── workflows/
│       ├── ci.yml          # lint + test matrix (Python 3.10–3.12)
│       └── publish.yml     # publish to PyPI on version tag
├── src/
│   └── finfeatures/
│       ├── core/
│       │   ├── base.py     Feature, DataSource, FeatureRegistry, Columns
│       │   └── pipeline.py FeaturePipeline + preset pipelines
│       ├── features/
│       │   ├── price.py
│       │   ├── volatility.py
│       │   ├── trend.py
│       │   ├── momentum.py
│       │   ├── volume.py
│       │   ├── statistical.py
│       │   └── composite.py
│       ├── sources/
│       │   └── yfinance.py  YFinanceSource (optional dep)
│       └── io/
│           └── adapters.py  PandasAdapter
├── tests/
│   ├── conftest.py          synthetic fixtures, zero network
│   ├── test_core.py
│   ├── test_features.py
│   ├── test_integration.py
│   └── test_sources.py
├── .gitignore
├── .python-version          3.12 (uv pin)
├── pyproject.toml
├── CHANGELOG.md
└── README.md
```

---

## Development

```bash
# Clone and set up
git clone https://github.com/whitham-powell/finfeatures
cd finfeatures
uv sync                         # installs all dev deps, creates .venv

# Run tests
uv run pytest                          # offline tests only
uv run pytest -m network               # include live yfinance tests

# Lint / format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type-check
uv run mypy src/finfeatures

# Build distribution
uv build                        # produces dist/*.whl and dist/*.tar.gz
```

### Release

```bash
git tag v0.1.1
git push --tags
# → triggers publish.yml → builds and publishes to PyPI via OIDC trusted publishing
```
