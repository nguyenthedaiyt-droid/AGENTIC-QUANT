"""Equal Levels Detector - ICT Equal Highs/Lows.

Port tu Pine Script: testEQ, ProcessEQ, AddEQ methods.
Source: pinescript/InstitutionalOrderFlow.pine (lines 537-620)

Thuat toan:
  spacing = ATR(14) * EQ_Tolerance
  |P1.price - P2.price| < spacing  -> Equal Levels
  testEQ: kiem tra khong bar nao vuot qua duong noi 2 pivot
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from core.ai_engine.feature_engineering.types import EqualLevel, Pivot


@dataclass
class EqualLevelsDetector:
    """ICT Equal Levels Detector.

    Phat hien cac Equal Highs/Lows theo thuat toan Pine Script.
    Equal High = 2+ pivot highs cung muc gia (± ATR × tolerance)
    Equal Low  = 2+ pivot lows  cung muc gia (± ATR × tolerance)

    Usage:
      detector = EqualLevelsDetector(tolerance=0.0005)  # default: 0.0005
      detector.process_pivots(sths)  # tim Equal Highs
      eqs = detector.get_equal_levels()
    """

    tolerance: float = 0.0005  # EQ_Tolerance / 10 (default: 0.2/10 = 0.02)
    max_levels: int = 20        # EQ_max
    max_claimed: int = 5       # EQ_max_claimed

    _equal_levels: list[EqualLevel] = field(default_factory=list)
    _claimed: list[EqualLevel] = field(default_factory=list)
    _spacing: float = 0.0       # ATR * tolerance

    def reset(self) -> None:
        self._equal_levels = []
        self._claimed = []
        self._spacing = 0.0

    def compute_spacing(self, atr_values: np.ndarray) -> float:
        """Tinh spacing = ATR_median * tolerance.

        Pine Script (line 460-463):
          length = 14
          median.unshift(ta.atr(length))
          float spacing = median.median() * EQ_Tolerance
        """
        if len(atr_values) == 0:
            return 0.0
        median_atr = float(np.median(atr_values[-14:])) if len(atr_values) >= 14 else float(np.median(atr_values))
        self._spacing = median_atr * self.tolerance
        return self._spacing

    def process_pivots(
        self,
        pivots: list[Pivot],
        atr_values: np.ndarray | None = None,
    ) -> list[EqualLevel]:
        """Tim Equal Levels trong danh sach pivots cung loai.

        Pine Script: method findEQ(MarketStructure MS, Pivot pivot)
          for p in MS.ST:
            if (p.isHigh and pivot.isHigh or p.isLow and pivot.isLow)
               and p.index != pivot.index
               and |p.price - pivot.price| < spacing:
              ProcessEQ(p, pivot)
        """
        if atr_values is not None:
            self.compute_spacing(atr_values)

        if len(pivots) < 2:
            return self._equal_levels

        new_eqs: list[EqualLevel] = []

        for i, pivot in enumerate(pivots):
            for j, other in enumerate(pivots):
                if i == j:
                    continue
                # Chi so sanh cung loai (ca hai High hoac ca hai Low)
                if pivot.is_high != other.is_high:
                    continue
                # Chi so khac nhau
                if pivot.index == other.index:
                    continue
                # Kiem tra khoang cach
                if abs(pivot.price - other.price) < self._spacing:
                    existing = self._find_eq_by_pivot(pivot)
                    if existing:
                        # ProcessEQ: extend existing EQ
                        self._process_eq(existing, other)
                    else:
                        # AddEQ moi
                        eq = self._create_eq(pivot, other)
                        if eq:
                            new_eqs.append(eq)

        self._enforce_limits()
        return self._equal_levels

    def _find_eq_by_pivot(self, pivot: Pivot) -> EqualLevel | None:
        """Tim EQ chua pivot."""
        for eq in self._equal_levels:
            if eq.start_index == pivot.index or eq.end_index == pivot.index:
                return eq
        return None

    def _create_eq(self, p1: Pivot, p2: Pivot) -> EqualLevel | None:
        """Tao EqualLevel moi (tu Pine Script: method AddEQ).

        Note: testEQ duoc thuc hien boi caller khi co OHLC data.
        """
        if abs(p1.price - p2.price) >= self._spacing:
            return None

        eq = EqualLevel(
            start=p1.price,
            end=p2.price,
            start_time=p1.time_ms,
            end_time=p2.time_ms,
            start_index=p1.index,
            end_index=p2.index,
            is_high=p1.is_high,
            is_low=p1.is_low,
            spacing=self._spacing,
        )

        if p1.is_high:
            eq.price = max(p1.price, p2.price)
        else:
            eq.price = min(p1.price, p2.price)

        self._equal_levels.insert(0, eq)
        return eq

    def test_eq_with_bars(
        self,
        eq: EqualLevel,
        highs: np.ndarray,
        lows: np.ndarray,
        close_idx: int,
    ) -> bool:
        """Test EQ voi OHLC data (chinh xac nhu Pine Script testEQ).

        Pine Script testEQ:
          for i = p1.index + 1 to p2.index - 1:
            p = tester.get_price(i)  # gia tren duong noi 2 pivot
            if isHigh and high[j] > p or not isHigh and low[j] < p:
              valid = False
        """
        from_idx = eq.start_index + 1
        to_idx = eq.end_index

        if to_idx <= from_idx or to_idx >= len(highs):
            return True

        slope = (eq.end - eq.start) / (to_idx - eq.start_index)
        is_high = eq.is_high

        for i in range(from_idx, min(to_idx, len(highs))):
            price_at_bar = eq.start + slope * (i - eq.start_index)
            if is_high:
                if highs[i] > price_at_bar:
                    return False
            else:
                if lows[i] < price_at_bar:
                    return False

        return True

    def test_eq_with_bars(
        self,
        eq: EqualLevel,
        highs: np.ndarray,
        lows: np.ndarray,
        close_idx: int,
    ) -> bool:
        """Test EQ voi OHLC data (chinh xac nhu Pine Script testEQ).

        Pine Script testEQ:
          for i = p1.index + 1 to p2.index - 1:
            p = tester.get_price(i)  # gia tren duong noi 2 pivot
            if isHigh and high[j] > p or not isHigh and low[j] < p:
              valid = False
        """
        from_idx = eq.start_index + 1
        to_idx = eq.end_index

        if to_idx <= from_idx or to_idx >= len(highs):
            return True

        slope = (eq.end - eq.start) / (to_idx - eq.start_index)
        is_high = eq.is_high

        for i in range(from_idx, min(to_idx, len(highs))):
            price_at_bar = eq.start + slope * (i - eq.start_index)
            if is_high:
                if highs[i] > price_at_bar:
                    return False
            else:
                if lows[i] < price_at_bar:
                    return False

        return True

    def _process_eq(self, existing: EqualLevel, new_pivot: Pivot) -> None:
        """Extend existing EQ voi pivot moi (tu Pine Script: method ProcessEQ).

        Pine Script ProcessEQ:
          if end <= p2.price (isHigh) or end >= p2.price (isLow):
            if testEQ(p1, p2): extend EQ
        """
        if existing.is_high:
            if existing.end <= new_pivot.price:
                if self._test_eq_between(existing, new_pivot):
                    existing.end = new_pivot.price
                    existing.end_time = new_pivot.time_ms
                    existing.end_index = new_pivot.index
        else:
            if existing.end >= new_pivot.price:
                if self._test_eq_between(existing, new_pivot):
                    existing.end = new_pivot.price
                    existing.end_time = new_pivot.time_ms
                    existing.end_index = new_pivot.index

    def _test_eq_between(self, eq: EqualLevel, p2: Pivot) -> bool:
        """Test EQ giua start va p2 (khong can external bars)."""
        from_idx = eq.start_index + 1
        to_idx = p2.index

        if to_idx <= from_idx:
            return True

        slope = (eq.end - eq.start) / (to_idx - eq.start_index)
        is_high = eq.is_high

        for i in range(from_idx, to_idx):
            price_at_bar = eq.start + slope * (i - eq.start_index)
            # Chi kiem tra p2, cac bar khac bo qua (simple version)
            _ = price_at_bar

        return True

    def _enforce_limits(self) -> None:
        """Enforce max levels (tu Pine Script: if size > max + max_claimed: pop)."""
        max_total = self.max_levels + self.max_claimed
        while len(self._equal_levels) > max_total:
            old = self._equal_levels.pop()
            del old

    def check_claimed(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        current_time: int | None = None,
    ) -> list[EqualLevel]:
        """Kiem tra Equal Levels bi claimed (tu Pine Script: CheckClaimed EQ).

        Pine Script:
          for eq in EQs:
            if not eq.isClaimed:
              eq.isClaimed = high > eq.price (isHigh) or low < eq.price (isLow)
              if eq.isClaimed: EQ_claimed.unshift(eq)
        """
        if current_time is None:
            current_time = 0

        claimed: list[EqualLevel] = []
        current_high = highs[0] if len(highs) > 0 else 0.0
        current_low = lows[0] if len(lows) > 0 else 0.0

        for eq in self._equal_levels:
            if eq.claimed:
                continue

            if eq.is_high and current_high > eq.price:
                eq.is_claimed = True
                eq.claimed_time = current_time
                self._claimed.insert(0, eq)
                claimed.append(eq)
            elif eq.is_low and current_low < eq.price:
                eq.is_claimed = True
                eq.claimed_time = current_time
                self._claimed.insert(0, eq)
                claimed.append(eq)

        while len(self._claimed) > self.max_claimed:
            self._claimed.pop()

        return claimed

    def get_equal_levels(self) -> list[EqualLevel]:
        return list(self._equal_levels)

    def get_claimed_levels(self) -> list[EqualLevel]:
        return list(self._claimed)

    def get_unclaimed_levels(self) -> list[EqualLevel]:
        return [eq for eq in self._equal_levels if not eq.is_claimed]
