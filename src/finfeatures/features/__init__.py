# Import all feature modules so their classes auto-register via __init_subclass__
from . import (  # noqa: F401
    composite,
    momentum,
    patterns,
    price,
    statistical,
    trend,
    volatility,
    volume,
)
from .composite import (
    CompositeScores,
    DistributionShiftScore,
    DrawdownFeatures,
)
from .momentum import (
    RSI,
    Aroon,
    ChandeMomentumOscillator,
    CommodityChannelIndex,
    MomentumScore,
    MoneyFlowIndex,
    RateOfChange,
    StochasticOscillator,
    UltimateOscillator,
    WilliamsR,
)
from .patterns import CandlePatterns
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
    LinearRegressionSlope,
    RollingAutocorrelation,
    RollingCorrelation,
    RollingMoments,
    RollingSkewKurt,
    RollingZScore,
)
from .trend import (
    DEMA,
    KAMA,
    MACD,
    TEMA,
    ExponentialMovingAverage,
    MACrossover,
    ParabolicSAR,
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
    AccumulationDistribution,
    ChaikinADOscillator,
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
    "KAMA",
    "ParabolicSAR",
    "DEMA",
    "TEMA",
    # momentum
    "RSI",
    "RateOfChange",
    "StochasticOscillator",
    "WilliamsR",
    "CommodityChannelIndex",
    "MomentumScore",
    "MoneyFlowIndex",
    "Aroon",
    "ChandeMomentumOscillator",
    "UltimateOscillator",
    # volume
    "VolumeFeatures",
    "OnBalanceVolume",
    "VWAP",
    "ChaikinMoneyFlow",
    "AccumulationDistribution",
    "ChaikinADOscillator",
    # statistical
    "RollingZScore",
    "RollingSkewKurt",
    "RollingMoments",
    "RollingAutocorrelation",
    "RollingCorrelation",
    "LinearRegressionSlope",
    # patterns
    "CandlePatterns",
    # composite
    "DistributionShiftScore",
    "DrawdownFeatures",
    "CompositeScores",
]
