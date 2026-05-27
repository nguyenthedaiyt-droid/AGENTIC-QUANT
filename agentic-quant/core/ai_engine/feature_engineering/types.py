"""Shared types for Feature Engineering modules.

Port tu Pine Script: type Pivot, type Imbalance, type EqualLevels.
Dung lam "source of truth" cho tat ca cac detector.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


# =============================================================================
# Pivot / Swing Point Types
# =============================================================================
class PivotTerm(str, Enum):
    """Muc do term cua pivot: ST (Short-term), IT (Intermediate-term), LT (Long-term)."""
    ST = "ST"
    IT = "IT"
    LT = "LT"


class SwingType(str, Enum):
    """Danh gia swing point (relative to previous same-type pivot)."""
    HH = "HH"       # Higher High - current STH > previous STH
    LH = "LH"       # Lower High - current STH < previous STH
    LL = "LL"       # Lower Low  - current STL < previous STL
    HL = "HL"       # Higher Low - current STL > previous STL


@dataclass
class Pivot:
    """Port tu Pine Script: type Pivot.

    Mot pivot la mot diem xoay tren chart (swing high hoac swing low).
    Moi pivot co the la ST, IT, hoac LT tuy theo so luong pivot cung loai da xuat hien.
    """
    index: int = 0           # bar_index (offset tu current bar)
    time_ms: int = 0         # bar time (ms timestamp)
    price: float = 0.0
    time_last: int = 0       # claimed_time - thoi diem pivot bi claimed
    claimed: bool = False

    # Direction
    is_high: bool = False    # True = swing high (BSL), False = swing low (SSL)
    is_low: bool = False

    # HH/LH/HL/LL classification (relative to previous same-type pivot)
    is_higher_high: bool = False   # HH  - STH moi cao hon STH truoc
    is_lower_low: bool = False     # LL  - STL moi thap hon STL truoc
    is_s_higher_high: bool = False # sHH - STH vua duoc xep loai
    is_s_lower_low: bool = False   # sLL - STL vua duoc xep loai

    # IT/LT promotion flags (Pine Script: isIHigherHigh, isILowerLow, isLHigherHigh, isLLowerLow)
    is_i_higher_high: bool = False
    is_i_lower_low: bool = False
    is_l_higher_high: bool = False
    is_l_lower_low: bool = False

    # Term
    term: PivotTerm = PivotTerm.ST

    # Confidence score cho ML
    confidence: float = 1.0

    def is_bullish(self) -> bool:
        """Pivot thuoc phia buy-side (swing high = BSL)."""
        return self.is_high

    def is_bearish(self) -> bool:
        """Pivot thuoc phia sell-side (swing low = SSL)."""
        return self.is_low

    @property
    def swing_type(self) -> SwingType | None:
        """Tra ve SwingType neu co thong tin de xac dinh."""
        if self.is_high:
            if self.is_higher_high:
                return SwingType.HH
            return SwingType.LH
        if self.is_low:
            if self.is_lower_low:
                return SwingType.LL
            return SwingType.HL
        return None

    @property
    def label(self) -> str:
        """Label nhu Pine Script: ST-HH, IT-LH, LT-HH, etc."""
        prefix = self.term.value
        suffix = ""
        if self.is_high:
            if self.term == PivotTerm.ST:
                suffix = "-HH" if self.is_higher_high else "-LH"
            elif self.term == PivotTerm.IT:
                suffix = "-HH" if self.is_i_higher_high else "-LH"
            else:
                suffix = "-HH" if self.is_l_higher_high else "-LH"
        else:
            if self.term == PivotTerm.ST:
                suffix = "-LL" if self.is_lower_low else "-HL"
            elif self.term == PivotTerm.IT:
                suffix = "-LL" if self.is_i_lower_low else "-HL"
            else:
                suffix = "-LL" if self.is_l_lower_low else "-HL"
        return f"{prefix}{suffix}"


# =============================================================================
# Mitigation Types (tu Pine Script: mitigated_type)
# =============================================================================
class MitigationType(str, Enum):
    """6 loai mitigation, exact nhu Pine Script.

    Danh sach theo thu tu do tot den xau:
      1. NONE           - khong bao gio mitigated
      2. WICK_TOUCHED  - wick cham vao close line
      3. WICK_FILLED    - wick di qua open line
      4. BODY_FILLED    - body di qua open line
      5. WICK_FILLED_HALF - wick di qua middle line
      6. BODY_FILLED_HALF - body di qua middle line
    """
    NONE = "NONE"
    WICK_TOUCHED = "WICK_TOUCHED"
    WICK_FILLED = "WICK_FILLED"
    BODY_FILLED = "BODY_FILLED"
    WICK_FILLED_HALF = "WICK_FILLED_HALF"
    BODY_FILLED_HALF = "BODY_FILLED_HALF"

    @classmethod
    def from_pine_string(cls, s: str) -> MitigationType:
        """Parse tu string cua Pine Script."""
        mapping = {
            "None": cls.NONE,
            "Wick Touched": cls.WICK_TOUCHED,
            "Wick filled": cls.WICK_FILLED,
            "Body filled": cls.BODY_FILLED,
            "Wick filled half": cls.WICK_FILLED_HALF,
            "Body filled half": cls.BODY_FILLED_HALF,
        }
        return mapping.get(s, cls.NONE)


# =============================================================================
# Imbalance / FVG Types (tu Pine Script: type Imbalance)
# =============================================================================
class ImbalanceType(str, Enum):
    """Loai imbalance: FVG, iFVG, VI (Volume Imbalance), GAP."""
    FVG = "FVG"
    IFVG = "iFVG"
    VI = "VI"
    GAP = "GAP"


@dataclass
class Imbalance:
    """Port tu Pine Script: type Imbalance.

    Mot FVG/VI/GAP la vung gap gia giua 2-3 candles.
    Bullish FVG: low > high[2]  (gap up)
    Bearish FVG: high < low[2] (gap down)
    """
    imb_type: ImbalanceType = ImbalanceType.FVG

    # Timing
    open_time: int = 0     # timestamp ms cua candle tao open
    close_time: int = 0    # timestamp ms cua candle tao close

    # Price range
    top: float = 0.0    # gia cao hon (bullish: close, bearish: open)
    bottom: float = 0.0  # gia thap hon (bullish: open, bearish: close)

    # Middle = CE line
    @property
    def middle(self) -> float:
        return (self.top + self.bottom) / 2.0

    # Direction
    @property
    def is_bullish(self) -> bool:
        return self.top < self.bottom

    # Mitigation state
    mitigated: bool = False
    mitigated_time: int = 0
    mitigated_type: MitigationType = MitigationType.NONE

    # iFVG state (FVG bi invert)
    invertible: bool = False    # bullish != previous.bullish (co the bi invert)
    inverted: bool = False      # da bi invert thanh iFVG
    inverted_mitigated: bool = False
    inverted_mitigated_time: int = 0

    # CE (Consequent Encroachment) - tinh khi co price hien tai
    _ce_pct: float | None = None

    # Premium/Discount zone
    zone: str = "mid"   # "premium" | "discount" | "mid"

    # Displacement info
    displacement_factor: float = 1.0
    body_size: float = 0.0  # |open - close| cua candle displacement

    # Strength metric cho ML
    strength: float = 0.0

    # Unique ID
    id: str = ""

    def compute_ce(self, current_price: float) -> float:
        """Tinh CE% = |current_price - middle| / range * 100."""
        if self.top == self.bottom:
            return 0.0
        dist = abs(current_price - self.middle)
        ce = dist / abs(self.top - self.bottom) * 100.0
        self._ce_pct = ce
        return ce

    @property
    def range_size(self) -> float:
        return abs(self.top - self.bottom)

    def __post_init__(self) -> None:
        if not self.id and self.open_time:
            direction = "bull" if self.is_bullish else "bear"
            self.id = f"{self.imb_type.value}_{direction}_{self.open_time}"


@dataclass
class FVGCollection:
    """Collections of FVGs and iFVGs (tu ImbalanceStructure)."""
    fvgs: list[Imbalance] = field(default_factory=list)
    ifvgs: list[Imbalance] = field(default_factory=list)


# =============================================================================
# Equal Levels (tu Pine Script: type EqualLevels)
# =============================================================================
@dataclass
class EqualLevel:
    """Port tu Pine Script: type EqualLevels.

    Equal High/Low la 2 pivot cung loai, cung muc gia (± ATR tolerance).
    """
    price: float = 0.0    # gia chung cua 2 pivot
    start: float = 0.0    # gia pivot dau tien
    end: float = 0.0      # gia pivot thu 2
    start_time: int = 0
    end_time: int = 0
    start_index: int = 0
    end_index: int = 0
    is_high: bool = False   # Equal High
    is_low: bool = False    # Equal Low
    is_claimed: bool = False
    claimed_time: int = 0

    # Confidence
    spacing: float = 0.0   # ATR * tolerance (khoang cach cho phep)
    num_touches: int = 0  # so lan 2 pivot cung gia

    @property
    def range_size(self) -> float:
        return abs(self.end - self.start)

    @property
    def label(self) -> str:
        return "EQ High" if self.is_high else "EQ Low"


# =============================================================================
# Displacement (tu Pine Script: f_highlightDisplacement)
# =============================================================================
@dataclass
class DisplacementConfig:
    """Config cho displacement (tu Pine Script settings)."""
    length: int = 100        # so bars tinh std body
    factor: int = 2          # displacement_factor (1-4)
    require_fvg: bool = True
    show: bool = False

    @classmethod
    def fvg_level(cls, level: int) -> DisplacementConfig:
        """Factory: FVG displacement level 1-4."""
        return cls(factor=level, require_fvg=True)


@dataclass
class DisplacementResult:
    """Ket qua displacement (tu Pine Script: f_highlightDisplacement())."""
    is_displaced: bool = False  # body[1] > std * factor
    is_bullish: bool = False    # open < close (green candle)
    d_strength: float = 0.0    # |body| / (std × factor)
    fvg_confirmed: bool = False  # low[1] > high[2] or high[1] < low[2]


# =============================================================================
# Structure Events (MSS, BOS)
# =============================================================================
class StructureEventType(str, Enum):
    BULLISH_MSS = "BULLISH_MSS"
    BEARISH_MSS = "BEARISH_MSS"
    BULLISH_BOS = "BULLISH_BOS"
    BEARISH_BOS = "BEARISH_BOS"


@dataclass
class StructureEvent:
    """MSS / BOS event.

    MSS (Market Structure Shift): Khi 1 swing high/low bi break boi price action
    BOS (Break of Structure): Khi nhieu swing highs/lows bi break lien tiep
    """
    event_type: StructureEventType
    trigger_price: float
    trigger_index: int
    trigger_time: int
    pivot_high: Pivot | None = None
    pivot_low: Pivot | None = None
    strength: float = 0.0   # = |high[bos] - low[bos]| / ATR
    confidence: float = 1.0


@dataclass
class StructureMap:
    """Port tu note trong todo.md.

    StructureMap = {mss_events, bos_events, premium_zones, discount_zones, fib_levels}
    """
    mss_events: list[StructureEvent] = field(default_factory=list)
    bos_events: list[StructureEvent] = field(default_factory=list)
    premium_zones: list[dict] = field(default_factory=list)  # [{top, bottom, strength}]
    discount_zones: list[dict] = field(default_factory=list)
    fib_levels: dict[float, float] = field(default_factory=dict)  # {0.382: price, ...}
    equilibrium: float = 0.0
    major_swing_high: Pivot | None = None
    major_swing_low: Pivot | None = None

    @property
    def is_bullish_context(self) -> bool:
        """Neu co BULLISH_MSS/BOS gan nhat -> True."""
        if self.mss_events:
            return self.mss_events[0].event_type == StructureEventType.BULLISH_MSS
        if self.bos_events:
            return self.bos_events[0].event_type == StructureEventType.BULLISH_BOS
        return True


# =============================================================================
# Feature Vectors (cho ML models)
# =============================================================================
@dataclass
class FeatureVectors:
    """Feature vectors cho XGBoost Model A va Model B."""
    f_struct: list[float]   # 64 dims - structure + FVG + EQ
    f_agg: list[float]     # 16 dims - aggregated
    f_liq: list[float]     # 24 dims - liquidity
    d_strength: list[float]  # 5 dims - displacement configs


@dataclass
class FeatureEngineeringConfig:
    """Config tong hop cho FeatureEngineering (tu agentic_quant_full_plan.md)."""
    displacement: DisplacementConfig = field(default_factory=DisplacementConfig)
    fvg_merge_vi: bool = True
    fvg_max_count: int = 5
    fvg_displacement_level: int = 2
    eq_tolerance: float = 0.0005
    ob_atr_window: int = 14
    news_guardrail_active: bool = False
    session_ltf_weight: float = 1.0
    session_htf_weight: float = 1.0
    mitigated_type: MitigationType = MitigationType.WICK_FILLED


@dataclass
class FeatureOutput:
    """Output cua FeatureEngineeringPipeline."""
    symbol: str = ""
    timeframe: str = ""
    swing_points: list = field(default_factory=list)
    structure_map: StructureMap | None = None
    fvg_collection: FVGCollection | None = None
    equal_levels: list = field(default_factory=list)
    f_struct: "np.ndarray | None" = None  # [64]
    f_agg: "np.ndarray | None" = None    # [16]
    f_liq: "np.ndarray | None" = None    # [24]
    d_strength: "np.ndarray | None" = None  # [5]
