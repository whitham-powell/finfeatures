# Import all feature modules so their classes auto-register via __init_subclass__
from . import momentum, price, regime, statistical, trend, volatility, volume  # noqa: F401
from .momentum import (
    RSI,
    CommodityChannelIndex,
    MomentumScore,
    RateOfChange,
    StochasticOscillator,
    WilliamsR,
)
from .price import (
    CandleShape,
    CrossDay,
    CumulativeReturn,
    LogReturns,
    LogTransform,
    PriceRange,
    PriceRelativeToHigh,
    Returns,
    ShapeDynamics,
    TypicalPrice,
)
from .regime import (
    DistributionShiftScore,
    DrawdownFeatures,
    RegimeIndicators,
)
from .statistical import (
    RollingAutocorrelation,
    RollingCorrelation,
    RollingMoments,
    RollingSkewKurt,
    RollingZScore,
)
from .trend import (
    MACD,
    ExponentialMovingAverage,
    MACrossover,
    SimpleMovingAverage,
    TrendStrength,
)
from .volatility import (
    AverageTrueRange,
    BollingerBands,
    GarmanKlassVolatility,
    MovingTrueRange,
    ParkinsonVolatility,
    RollingVolatility,
    VolatilityRegime,
)
from .volume import (
    VWAP,
    ChaikinMoneyFlow,
    OnBalanceVolume,
    VolumeFeatures,
)

__all__ = [
    # price
    "Returns",
    "LogReturns",
    "LogTransform",
    "CandleShape",
    "CrossDay",
    "ShapeDynamics",
    "PriceRange",
    "TypicalPrice",
    "CumulativeReturn",
    "PriceRelativeToHigh",
    # volatility
    "RollingVolatility",
    "ParkinsonVolatility",
    "GarmanKlassVolatility",
    "BollingerBands",
    "AverageTrueRange",
    "MovingTrueRange",
    "VolatilityRegime",
    # trend
    "SimpleMovingAverage",
    "ExponentialMovingAverage",
    "MACD",
    "TrendStrength",
    "MACrossover",
    # momentum
    "RSI",
    "RateOfChange",
    "StochasticOscillator",
    "WilliamsR",
    "CommodityChannelIndex",
    "MomentumScore",
    # volume
    "VolumeFeatures",
    "OnBalanceVolume",
    "VWAP",
    "ChaikinMoneyFlow",
    # statistical
    "RollingZScore",
    "RollingSkewKurt",
    "RollingMoments",
    "RollingAutocorrelation",
    "RollingCorrelation",
    # regime
    "DistributionShiftScore",
    "DrawdownFeatures",
    "RegimeIndicators",
]
