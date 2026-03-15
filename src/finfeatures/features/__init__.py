# Import all feature modules so their classes auto-register via __init_subclass__
from . import composite, momentum, price, statistical, trend, volatility, volume  # noqa: F401
from .composite import (
    CompositeScores,
    DistributionShiftScore,
    DrawdownFeatures,
)
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
    VolatilityRatio,
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
    "VolatilityRatio",
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
    # composite
    "DistributionShiftScore",
    "DrawdownFeatures",
    "CompositeScores",
]
