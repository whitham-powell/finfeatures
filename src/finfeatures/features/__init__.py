# Import all feature modules so their classes auto-register via __init_subclass__
from . import price, volatility, trend, momentum, volume, statistical, regime

from .price import (
    Returns,
    LogReturns,
    PriceRange,
    TypicalPrice,
    CumulativeReturn,
    PriceRelativeToHigh,
)
from .volatility import (
    RollingVolatility,
    ParkinsonVolatility,
    GarmanKlassVolatility,
    BollingerBands,
    AverageTrueRange,
    VolatilityRegime,
)
from .trend import (
    SimpleMovingAverage,
    ExponentialMovingAverage,
    MACD,
    TrendStrength,
    MACrossover,
)
from .momentum import (
    RSI,
    RateOfChange,
    StochasticOscillator,
    WilliamsR,
    CommodityChannelIndex,
    MomentumScore,
)
from .volume import (
    VolumeFeatures,
    OnBalanceVolume,
    VWAP,
    ChaikinMoneyFlow,
)
from .statistical import (
    RollingZScore,
    RollingSkewKurt,
    RollingMoments,
    RollingAutocorrelation,
    RollingCorrelation,
)
from .regime import (
    DistributionShiftScore,
    DrawdownFeatures,
    RegimeIndicators,
)

__all__ = [
    # price
    "Returns", "LogReturns", "PriceRange", "TypicalPrice",
    "CumulativeReturn", "PriceRelativeToHigh",
    # volatility
    "RollingVolatility", "ParkinsonVolatility", "GarmanKlassVolatility",
    "BollingerBands", "AverageTrueRange", "VolatilityRegime",
    # trend
    "SimpleMovingAverage", "ExponentialMovingAverage", "MACD",
    "TrendStrength", "MACrossover",
    # momentum
    "RSI", "RateOfChange", "StochasticOscillator", "WilliamsR",
    "CommodityChannelIndex", "MomentumScore",
    # volume
    "VolumeFeatures", "OnBalanceVolume", "VWAP", "ChaikinMoneyFlow",
    # statistical
    "RollingZScore", "RollingSkewKurt", "RollingMoments",
    "RollingAutocorrelation", "RollingCorrelation",
    # regime
    "DistributionShiftScore", "DrawdownFeatures", "RegimeIndicators",
]
