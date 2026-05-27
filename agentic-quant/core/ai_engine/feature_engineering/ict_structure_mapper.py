"""ICT Structure Mapper - Market Structure, Fib Levels, Premium/Discount.

Port tu Pine Script: ICT Market Structure + todo.md 4.3.
Thuat toan:
  - MSS: Market Structure Shift (HH break -> Bullish MSS, LL break -> Bearish MSS)
  - BOS: Break of Structure (nhieu swings bi break)
  - Fib: 0.382, 0.5, 0.618, 0.786
  - Premium/Discount zones
"""
from __future__ import annotations

import numpy as np

from core.ai_engine.feature_engineering.types import (
    Pivot,
    PivotTerm,
    StructureEvent,
    StructureEventType,
    StructureMap,
)


# Weight: MSS quan trong hon BOS (MSS weight = 2x BOS weight)
MSS_WEIGHT = 2.0
BOS_WEIGHT = 1.0


class ICTStructureMapper:
    """ICT Structure Mapper.

    Phat hien va phan loai:
      - Market Structure Shift (MSS) - su thay doi cau truc
      - Break of Structure (BOS) - pha vỡ cau truc
      - Fibonacci levels
      - Premium/Discount zones
    """

    def __init__(self) -> None:
        self._structure_map: StructureMap = StructureMap()
        self._last_mss_direction: str = "neutral"

    def reset(self) -> None:
        self._structure_map = StructureMap()
        self._last_mss_direction = "neutral"

    # =========================================================================
    # Equilibrium & Fibonacci
    # =========================================================================
    def detect_equilibrium(self, pivots: list[Pivot]) -> float:
        """Tinh Equilibrium = midpoint cua major Swing High/Low.

        Pine Script: Tim major Swing High = highest STH, Swing Low = lowest STL
        EQ = (MajorHigh + MajorLow) / 2
        """
        if not pivots:
            return 0.0

        highs = [p for p in pivots if p.is_high]
        lows = [p for p in pivots if p.is_low]

        if not highs or not lows:
            return 0.0

        major_high = max(highs, key=lambda p: p.price).price
        major_low = min(lows, key=lambda p: p.price).price

        eq = (major_high + major_low) / 2.0

        self._structure_map.equilibrium = eq
        self._structure_map.major_swing_high = max(highs, key=lambda p: p.price)
        self._structure_map.major_swing_low = min(lows, key=lambda p: p.price)

        return eq

    def detect_fib_levels(self, equilibrium: float, range_size: float) -> dict[float, float]:
        """Tinh cac muc Fibonacci tu Equilibrium.

        Pine Script / todo.md:
          Fib 0.382, 0.5, 0.618, 0.786

        Args:
            equilibrium: gia EQ (midpoint)
            range_size: khoang cach tu EQ den major high/low

        Returns:
            dict: {0.382: price, 0.5: price, 0.618: price, 0.786: price}
        """
        if range_size <= 0:
            return {}

        levels = {
            0.382: equilibrium - range_size * 0.382,
            0.5:   equilibrium,
            0.618: equilibrium + range_size * 0.618,
            0.786: equilibrium + range_size * 0.786,
        }

        self._structure_map.fib_levels = levels
        return levels

    def detect_fib_levels_from_range(
        self,
        major_high: float,
        major_low: float,
    ) -> dict[float, float]:
        """Tinh Fib levels tu major high/low range."""
        eq = (major_high + major_low) / 2.0
        range_half = (major_high - major_low) / 2.0
        return self.detect_fib_levels(eq, range_half * 2)

    # =========================================================================
    # Premium / Discount
    # =========================================================================
    def detect_premium_discount(
        self,
        current_price: float,
        equilibrium: float,
    ) -> str:
        """Phan vung Premium/Discount.

        Pine Script:
          Premium: gia > EQ
          Discount: gia < EQ

        Returns:
            "premium" | "discount" | "mid"
        """
        if current_price > equilibrium:
            return "premium"
        elif current_price < equilibrium:
            return "discount"
        return "mid"

    def get_premium_discount_zones(
        self,
        fib_levels: dict[float, float],
        major_high: float,
        major_low: float,
        equilibrium: float,
    ) -> tuple[list[dict], list[dict]]:
        """Lay premium/discount zones tu Fib levels.

        Premium zones: >= EQ
        Discount zones: <= EQ
        """
        premium = []
        discount = []

        for ratio, price in fib_levels.items():
            zone = {
                "ratio": ratio,
                "top": max(price, equilibrium),
                "bottom": min(price, equilibrium),
                "strength": float(ratio),
            }
            if price >= equilibrium:
                premium.append(zone)
            else:
                discount.append(zone)

        self._structure_map.premium_zones = premium
        self._structure_map.discount_zones = discount

        return premium, discount

    # =========================================================================
    # MSS Detection
    # =========================================================================
    def detect_mss(
        self,
        current_price: float,
        pivots: list[Pivot],
        timeframe: str = "M15",
    ) -> list[StructureEvent]:
        """Phat hien Market Structure Shift.

        Pine Script / todo.md:
          Bullish MSS: HH break (swing high bi pha vỡ)
          Bearish MSS: LL break (swing low bi pha vỡ)

        Thuat toan:
          - Duyet cac STH theo thu tu thoi gian
          - Neu STH moi > STH truoc -> HH break -> Bullish MSS
          - Tuong tu cho STL
        """
        mss_events: list[StructureEvent] = []
        sths = [p for p in pivots if p.is_high and p.term in (PivotTerm.ST, PivotTerm.IT)]
        stls = [p for p in pivots if p.is_low and p.term in (PivotTerm.ST, PivotTerm.IT)]

        # Sort by time (newest first)
        sths.sort(key=lambda p: p.index, reverse=True)
        stls.sort(key=lambda p: p.index, reverse=True)

        # Bullish MSS: current price breaks above a HH (previous STH high)
        # Pine Script: HH break = HH duoc tao khi STH moi > STH cu
        # MSS = khi price vuot qua 1 HH
        for i, sth in enumerate(sths):
            if i + 1 < len(sths):
                prev_sth = sths[i + 1]
                # HH: current STH > previous STH
                if sth.price > prev_sth.price:
                    # Check if price has broken above this HH
                    if current_price > sth.price:
                        event = StructureEvent(
                            event_type=StructureEventType.BULLISH_MSS,
                            trigger_price=current_price,
                            trigger_index=0,
                            trigger_time=0,
                            pivot_high=sth,
                            strength=self._calc_mss_strength(sth, prev_sth),
                        )
                        mss_events.append(event)
                        self._last_mss_direction = "bullish"

        # Bearish MSS: current price breaks below a LL (previous STL low)
        for i, stl in enumerate(stls):
            if i + 1 < len(stls):
                prev_stl = stls[i + 1]
                # LL: current STL < previous STL
                if stl.price < prev_stl.price:
                    if current_price < stl.price:
                        event = StructureEvent(
                            event_type=StructureEventType.BEARISH_MSS,
                            trigger_price=current_price,
                            trigger_index=0,
                            trigger_time=0,
                            pivot_low=stl,
                            strength=self._calc_mss_strength(stl, prev_stl),
                        )
                        mss_events.append(event)
                        self._last_mss_direction = "bearish"

        self._structure_map.mss_events = mss_events
        return mss_events

    def _calc_mss_strength(
        self,
        pivot: Pivot,
        prev_pivot: Pivot,
    ) -> float:
        """Tinh MSS strength = |break_size| / ATR."""
        break_size = abs(pivot.price - prev_pivot.price)
        # Normalize by price level
        strength = break_size / pivot.price if pivot.price > 0 else 0.0
        return float(strength)

    # =========================================================================
    # BOS Detection
    # =========================================================================
    def detect_bos(
        self,
        current_price: float,
        pivots: list[Pivot],
        timeframe: str = "M15",
    ) -> list[StructureEvent]:
        """Phat hien Break of Structure.

        Pine Script / todo.md:
          BOS: nhieu swing highs/lows bi break lien tiep
          Bullish BOS: >= 2 swing highs bi break
          Bearish BOS: >= 2 swing lows bi break

        Weight: BOS weight = 1.0 (MSS = 2.0)
        """
        bos_events: list[StructureEvent] = []

        sths = [p for p in pivots if p.is_high and p.term in (PivotTerm.ST, PivotTerm.IT)]
        stls = [p for p in pivots if p.is_low and p.term in (PivotTerm.ST, PivotTerm.IT)]

        # Sort newest first
        sths.sort(key=lambda p: p.index, reverse=True)
        stls.sort(key=lambda p: p.index, reverse=True)

        # Bullish BOS: >= 2 STHs broken
        broken_sths = 0
        last_broken_sth: Pivot | None = None
        for sth in sths:
            if current_price > sth.price:
                broken_sths += 1
                last_broken_sth = sth

        if broken_sths >= 2 and last_broken_sth is not None:
            event = StructureEvent(
                event_type=StructureEventType.BULLISH_BOS,
                trigger_price=current_price,
                trigger_index=0,
                trigger_time=0,
                pivot_high=last_broken_sth,
                strength=float(broken_sths) * BOS_WEIGHT,
            )
            bos_events.append(event)

        # Bearish BOS: >= 2 STLs broken
        broken_stls = 0
        last_broken_stl: Pivot | None = None
        for stl in stls:
            if current_price < stl.price:
                broken_stls += 1
                last_broken_stl = stl

        if broken_stls >= 2 and last_broken_stl is not None:
            event = StructureEvent(
                event_type=StructureEventType.BEARISH_BOS,
                trigger_price=current_price,
                trigger_index=0,
                trigger_time=0,
                pivot_low=last_broken_stl,
                strength=float(broken_stls) * BOS_WEIGHT,
            )
            bos_events.append(event)

        self._structure_map.bos_events = bos_events
        return bos_events

    # =========================================================================
    # Full Structure Map
    # =========================================================================
    def get_structure_map(
        self,
        pivots: list[Pivot],
        current_price: float,
        highs: np.ndarray | None = None,
        lows: np.ndarray | None = None,
        atr: float = 0.0,
    ) -> StructureMap:
        """Xay dung StructureMap day du.

        Args:
            pivots: danh sach tat ca pivots
            current_price: gia hien tai
            highs: mang high prices (optional, cho ATR)
            lows: mang low prices (optional)
            atr: gia tri ATR hien tai (optional)

        Returns:
            StructureMap day du
        """
        # Equilibrium
        eq = self.detect_equilibrium(pivots)

        # Fib levels
        if pivots:
            major_high = max(
                (p for p in pivots if p.is_high),
                key=lambda p: p.price,
                default=None,
            )
            major_low = min(
                (p for p in pivots if p.is_low),
                key=lambda p: p.price,
                default=None,
            )
            if major_high and major_low:
                self.detect_fib_levels_from_range(
                    major_high.price, major_low.price
                )

        # Premium/Discount
        self.detect_premium_discount(current_price, eq)
        self.get_premium_discount_zones(
            self._structure_map.fib_levels,
            self._structure_map.major_swing_high.price if self._structure_map.major_swing_high else current_price,
            self._structure_map.major_swing_low.price if self._structure_map.major_swing_low else current_price,
            eq,
        )

        # MSS & BOS
        self.detect_mss(current_price, pivots)
        self.detect_bos(current_price, pivots)

        return self._structure_map

    # =========================================================================
    # Helpers
    # =========================================================================
    @property
    def is_bullish_context(self) -> bool:
        """Neu MSS/BOS gan nhat la bullish -> True."""
        return self._structure_map.is_bullish_context

    @property
    def last_mss_direction(self) -> str:
        return self._last_mss_direction

    def classify_fvg_zone(
        self,
        fvg_bottom: float,
        fvg_top: float,
        current_price: float,
    ) -> str:
        """Classify FVG/OB zone = premium/discount."""
        fvg_mid = (fvg_bottom + fvg_top) / 2.0
        return self.detect_premium_discount(current_price, fvg_mid)
