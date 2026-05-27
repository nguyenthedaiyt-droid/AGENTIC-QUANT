# =============================================================================
# AGENTIC-QUANT — Zone Data Model
# Mirror tu TypeScript source of truth: ui/src/types/index.ts Zone interface
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from .enums import Timeframe, ZoneStatus, ZoneType

if TYPE_CHECKING:
    pass


# =============================================================================
# Zone: Cau truc du lieu cho cac vung SMC
# =============================================================================
@dataclass
class Zone:
    """
    Zone SMC (Support/Resistance area).

    Thuoc tinh nam trong TypeScript source of truth tai ui/src/types/index.ts.
    Day la ban Python mirror chinh xac cua interface Zone.

    Cac trang thai ZoneStatus transitions:
      UNMITIGATED -> WICK_TOUCHED -> WICK_FILLED_HALF -> WODY_FILLED -> MITIGATED

    Cac loai ZoneType: FVG_BULL/BEAR, OB_BULL/BEAR, VI_BULL/BEAR
    """

    id: str = ""
    symbol: str = ""
    timeframe: Timeframe = Timeframe.M1
    zone_type: ZoneType = ZoneType.OB_BULL
    top: float = 0.0  # Gia cao nhat cua vung (pips)
    bottom: float = 0.0  # Gia thap nhat cua vung (pips)
    ce: float = 0.0  # Consequent Encroachment - muc encroachment cho phep

    # Thoi gian hinh thanh (Unix ms)
    formed_time: int = 0

    status: ZoneStatus = ZoneStatus.UNMITIGATED
    p_hold: float = 0.0  # Xac suất zone duoc giu [0, 1]
    p_hold_updated: int = 0  # Unix ms - lan cuoi cap nhat p_hold

    # HTF alignment weight - duoc nhan len khi co confluence tu HTF
    w_zone: float = 1.0  # {0.5, 1.0, 2.0}

    # Institutional Imbalance Formation - chi so hinh thanh cua FVG/VI
    iii_formation: float = 0.0

    # So lan zone bi cham (touch)
    touch_count: int = 0
    last_touch_time: int = 0  # Unix ms

    # Higher timeframe context
    htf_tf: Timeframe | None = None

    def is_bullish(self) -> bool:
        """Tra ve True neu zone la bullish (FVG_BULL, OB_BULL, VI_BULL)."""
        return self.zone_type in (ZoneType.FVG_BULL, ZoneType.OB_BULL, ZoneType.VI_BULL)

    def is_bearish(self) -> bool:
        """Tra ve True neu zone la bearish (FVG_BEAR, OB_BEAR, VI_BEAR)."""
        return self.zone_type in (ZoneType.FVG_BEAR, ZoneType.OB_BEAR, ZoneType.VI_BEAR)

    def mid_price(self) -> float:
        """Gia trung binh cua zone."""
        return (self.top + self.bottom) / 2.0

    def range_size(self) -> float:
        """Kich thuoc zone (top - bottom)."""
        return self.top - self.bottom

    def contains_price(self, price: float) -> bool:
        """Kiem tra xem gia co nam trong zone khong."""
        return self.bottom <= price <= self.top

    def is_active(self) -> bool:
        """Zone con active (chua bi mitigated)."""
        return self.status != ZoneStatus.MITIGATED

    def to_dict(self) -> dict:
        """Chuyen doi thanh dictionary cho Redis/JSON serialization."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "timeframe": self.timeframe.value if isinstance(self.timeframe, Timeframe) else self.timeframe,
            "zone_type": self.zone_type.value if isinstance(self.zone_type, ZoneType) else self.zone_type,
            "top": self.top,
            "bottom": self.bottom,
            "ce": self.ce,
            "formed_time": self.formed_time,
            "status": self.status.value if isinstance(self.status, ZoneStatus) else self.status,
            "p_hold": self.p_hold,
            "p_hold_updated": self.p_hold_updated,
            "w_zone": self.w_zone,
            "iii_formation": self.iii_formation,
            "touch_count": self.touch_count,
            "last_touch_time": self.last_touch_time,
            "htf_tf": self.htf_tf.value if isinstance(self.htf_tf, Timeframe) else self.htf_tf,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Zone:
        """Tao Zone tu dictionary (tu Redis Hash hoac JSON)."""

        def _float(v):
            if isinstance(v, str):
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return 0.0
            return float(v)

        def _int(v):
            if isinstance(v, str):
                try:
                    return int(v)
                except (ValueError, TypeError):
                    return 0
            return int(v)

        htf_val = d.get("htf_tf")
        if htf_val and str(htf_val) not in ("None", "null", ""):
            htf_tf = Timeframe(htf_val)
        else:
            htf_tf = None

        return cls(
            id=str(d["id"]),
            symbol=str(d["symbol"]),
            timeframe=Timeframe(d["timeframe"]),
            zone_type=ZoneType(d["zone_type"]),
            top=_float(d["top"]),
            bottom=_float(d["bottom"]),
            ce=_float(d["ce"]),
            formed_time=_int(d["formed_time"]),
            status=ZoneStatus(d["status"]),
            p_hold=_float(d["p_hold"]),
            p_hold_updated=_int(d["p_hold_updated"]),
            w_zone=_float(d["w_zone"]),
            iii_formation=_float(d["iii_formation"]),
            touch_count=_int(d["touch_count"]),
            last_touch_time=_int(d["last_touch_time"]),
            htf_tf=htf_tf,
        )


@dataclass
class LiquidityTarget:
    """
    Muc tieu liquidity (BSL/SSL) voi xac suất tu Model A.
    """

    target_type: str = "BSL"  # "BSL" | "SSL"
    price: float = 0.0
    timeframe: Timeframe = Timeframe.M1
    p_probability: float = 0.0  # P_BSL hoac P_SSL tu Model A
    session: str = "ASIAN"

    def to_dict(self) -> dict:
        return {
            "target_type": self.target_type,
            "price": self.price,
            "timeframe": self.timeframe.value if isinstance(self.timeframe, Timeframe) else self.timeframe,
            "p_probability": self.p_probability,
            "session": self.session,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LiquidityTarget:
        return cls(
            target_type=d["target_type"],
            price=float(d["price"]),
            timeframe=Timeframe(d["timeframe"]),
            p_probability=float(d["p_probability"]),
            session=d.get("session", "ASIAN"),
        )
