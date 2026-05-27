# =============================================================================
# AGENTIC-QUANT — Enums cho Memory Module
# Tat ca enum values nam trong module nay de tranh circular import
# =============================================================================

from __future__ import annotations

from enum import Enum


class Timeframe(str, Enum):
    """Timeframe - nam trong core nhung nhung chi memory module can."""

    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"


class ZoneStatus(str, Enum):
    """
    Trang thai cua zone theo tien trinh mitigation.
    Gia tri nam trong module memory de tranh circular import voi core.events.types.
    """

    UNMITIGATED = "UNMITIGATED"
    WICK_TOUCHED = "WICK_TOUCHED"
    WICK_FILLED_HALF = "WICK_FILLED_HALF"
    WODY_FILLED = "WODY_FILLED"  # Typo trong TypeScript goc, giu nhu nguon chuan
    MITIGATED = "MITIGATED"


class ZoneType(str, Enum):
    """
    Loai zone SMC.
    - FVG: Fair Value Gap ( imbalance )
    - OB: Order Block
    - VI: Volume Imbalance
    """

    FVG_BULL = "FVG_BULL"
    FVG_BEAR = "FVG_BEAR"
    OB_BULL = "OB_BULL"
    OB_BEAR = "OB_BEAR"
    VI_BULL = "VI_BULL"
    VI_BEAR = "VI_BEAR"


class OutcomeType(str, Enum):
    """
    Ket qua cua mot prediction sau khi da xac dinh.
    Triggered boi OutcomeDeterminator khi BAR_CLOSE(M1) event xay ra.
    """

    BSL_HIT = "BSL_HIT"  # Buy-side liquidity da bi hit
    SSL_HIT = "SSL_HIT"  # Sell-side liquidity da bi hit
    LATERAL = "LATERAL"  # Khong co luc nao duoc kich hoat trong horizon
    TIMEOUT = "TIMEOUT"  # Qua 240 phut ma khong co ket qua
    ZONE_HOLD = "ZONE_HOLD"  # Zone duoc giu (touched + held 3 bars)


class PredictionOutcome(str, Enum):
    """
    Ket qua cua Model A prediction (BSL/SSL/Lateral).
    Dung trong SQLite schema va OutcomeDeterminator.
    """

    BSL_HIT = "BSL_HIT"
    SSL_HIT = "SSL_HIT"
    LATERAL = "LATERAL"
    TIMEOUT = "TIMEOUT"
    ZONE_HOLD = "ZONE_HOLD"


class ConfidenceQualifier(str, Enum):
    """Muc do tu tin cua prediction."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class LiquidityTargetType(str, Enum):
    """Loai muc tieu liquidity."""

    BSL = "BSL"
    SSL = "SSL"


class DebateDirection(str, Enum):
    """Huong dan cua mot thesis trong debate."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class MacroRegime(str, Enum):
    """
    Macro regime - nam trong memory module de tranh circular import voi core.
    """

    NORMAL = "NORMAL"
    PRE_NEWS = "PRE_NEWS"
    NEWS_WINDOW = "NEWS_WINDOW"
    POST_NEWS = "POST_NEWS"


class Session(str, Enum):
    """Trading session."""

    ASIAN = "ASIAN"
    LONDON_OPEN_KZ = "LONDON_OPEN_KZ"
    LONDON = "LONDON"
    NY_OPEN_KZ = "NY_OPEN_KZ"
    NY_AM = "NY_AM"
    NY_PM = "NY_PM"


class ModelDegradedFlag(str, Enum):
    """Trang thai degradation cua model."""

    STABLE = "STABLE"
    MINOR = "MINOR"
    SIGNIFICANT = "SIGNIFICANT"


class SystemStatus(str, Enum):
    """Trang thai he thong."""

    CONNECTED = "CONNECTED"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    DISCONNECTED = "DISCONNECTED"


class RegimeType(str, Enum):
    """Loai thi truong regime."""

    TRENDING_LV = "TRENDING_LV"
    TRENDING_HV = "TRENDING_HV"
    CHOPPY_HV = "CHOPPY_HV"
    NORMAL = "NORMAL"


class NewsImpact(str, Enum):
    """Muc do tac dong cua tin kinh te."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
