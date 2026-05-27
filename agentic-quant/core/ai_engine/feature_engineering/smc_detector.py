"""Swing Point Detector - ICT Market Structure.

Port tu Pine Script: FindST, FindIT, FindLT, CheckClaimed, SkipEQ methods.
Source: pinescript/InstitutionalOrderFlow.pine

Thuat toan co ban:
  - STH: high[1] > high[SkipEQHigh(2)] AND high[1] > high
  - STL:  low[1] < low[SkipEQLow(2)]  AND low[1] < low
  - SkipEQ: bo qua equal highs/lows
  - FindIT: 3 STH lien tiep -> H2 promote to ITH
  - FindLT: 3 ITH lien tiep -> H2 promote to LTH
  - CheckClaimed: high > pivot.price (STH) / low < pivot.price (STL)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from core.ai_engine.feature_engineering.types import Pivot, PivotTerm, SwingType


MAX_BUFFER = 1000  # internal maximum node count limit (tu Pine Script)


@dataclass
class SwingPointDetector:
    """ICT Swing Point Detector - port tu Pine Script MarketStructure methods.

    Thu tu goi:
      1. find_st()   -> phat hien Short-Term pivots (STH/STL)
      2. find_it()   -> promote STH -> ITH, STL -> ITL
      3. find_lt()   -> promote ITH -> LTH, ITL -> LTL
      4. check_claimed() -> danh dau pivot bi price sweep
    """

    # Internal pivot arrays (same as Pine Script var arrays)
    _st: list[Pivot] = field(default_factory=list)   # ST_array
    _sth: list[Pivot] = field(default_factory=list)  # STH_array
    _ith: list[Pivot] = field(default_factory=list)  # ITH_array
    _lth: list[Pivot] = field(default_factory=list)  # LTH_array
    _stl: list[Pivot] = field(default_factory=list)  # STL_array
    _itl: list[Pivot] = field(default_factory=list)  # ITL_array
    _ltl: list[Pivot] = field(default_factory=list)  # LTL_array

    def reset(self) -> None:
        """Reset all internal pivot arrays."""
        self._st = []
        self._sth = []
        self._ith = []
        self._lth = []
        self._stl = []
        self._itl = []
        self._ltl = []

    # =========================================================================
    # SkipEQ - bo qua equal highs/lows (Pine Script lines 508-532)
    # =========================================================================
    @staticmethod
    def skip_eq_high(idx: int, highs: np.ndarray) -> int:
        """Skip equal highs - while highs[i] == highs[i-1]: i += 1.

        Dung khi detect pivot, hoac khi promote IT/LT pivot.
        """
        i = idx
        n = len(highs)
        while i < n and i > 0 and i - 1 >= 0 and highs[i] == highs[i - 1]:
            i += 1
        return min(i, n - 1)

    @staticmethod
    def skip_eq_low(idx: int, lows: np.ndarray) -> int:
        """Skip equal lows - while lows[i] == lows[i-1]: i += 1."""
        i = idx
        n = len(lows)
        while i < n and i > 0 and i - 1 >= 0 and lows[i] == lows[i - 1]:
            i += 1
        return min(i, n - 1)

    @staticmethod
    def skip_eq_pivot(idx: int, pivots: list[Pivot]) -> int:
        """Skip equal-price pivots.

        Dung khi kiem tra 3 pivot lien tiep (FindIT/FindLT).
        Pine Script: method SkipEQPivot(array<Pivot> p, int idx).
        """
        i = idx
        n = len(pivots)
        while (i < n and i - 1 >= 0 and
               pivots[i].price == pivots[i - 1].price and
               n <= i - 1):
            i += 1
        return min(i, n - 1)

    # =========================================================================
    # Internal helper
    # =========================================================================
    def _classify_st_hh_lh(self, pivot: Pivot) -> None:
        """Phan loai HH/LH cho STH moi them.

        Pine Script: trong method Add(), khi isHigh:
          if STH.size() > 0:
            p = STH.first()
            if p.price <= p_price:  # previous STH <= current -> HH
              pivot.isSHigherHigh = True
          pivot.isHigherHigh = True
        """
        if self._sth:
            prev = self._sth[0]
            pivot.is_higher_high = prev.price <= pivot.price
        else:
            pivot.is_higher_high = True

        if pivot.is_higher_high:
            pivot.is_s_higher_high = True

    def _classify_st_ll_hl(self, pivot: Pivot) -> None:
        """Phan loai LL/HL cho STL moi them."""
        if self._stl:
            prev = self._stl[0]
            pivot.is_lower_low = prev.price >= pivot.price
        else:
            pivot.is_lower_low = True

        if pivot.is_lower_low:
            pivot.is_s_lower_low = True

    # =========================================================================
    # FindST - phat hien Short-Term pivots (Pine Script lines 845-853)
    # =========================================================================
    def find_st(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        times: np.ndarray | None = None,
        current_idx: int | None = None,
    ) -> list[Pivot]:
        """Phat hien STH/STL moi nhat.

        Pine Script FindST:
          h = high[1] > high[helper.SkipEQHigh(2)] and high[1] > high
          l = low[1]  < low[helper.SkipEQLow(2)]  and low[1]  < low

        Args:
            highs: mang numpy gia high, index 0 = bar hien tai
            lows: mang numpy gia low
            times: mang timestamp ms (optional)
            current_idx: chi so cua bar hien tai (default: len-1)
        """
        n = len(highs)
        if n < 3:
            return []

        if current_idx is None:
            current_idx = n - 1

        # Pine Script: look at bar[1] relative to bar[2] and bar[0]
        idx = 1  # bar_index[1]
        if idx >= n:
            return []

        skip_h = self.skip_eq_high(idx + 1, highs)  # SkipEQHigh(2) = idx+1
        skip_l = self.skip_eq_low(idx + 1, lows)

        new_pivots: list[Pivot] = []

        # STH: high[1] > high[SkipEQHigh(2)] AND high[1] > high
        if highs[idx] > highs[skip_h] and highs[idx] > highs[0]:
            pivot = self._add_st_high(highs[idx], idx, times, current_idx)
            if pivot:
                new_pivots.append(pivot)

        # STL: low[1] < low[SkipEQLow(2)] AND low[1] < low
        if lows[idx] < lows[skip_l] and lows[idx] < lows[0]:
            pivot = self._add_st_low(lows[idx], idx, times, current_idx)
            if pivot:
                new_pivots.append(pivot)

        return new_pivots

    def _add_st_high(
        self, price: float, bar_idx: int,
        times: np.ndarray | None, current_idx: int
    ) -> Pivot | None:
        """Them STH moi vao arrays."""
        n = len(self._st)
        if n >= MAX_BUFFER:
            # Remove oldest
            old = self._st.pop()
            if old in self._sth:
                self._sth.remove(old)

        pivot = Pivot(
            index=bar_idx,
            time_ms=int(times[bar_idx]) if times is not None else 0,
            price=price,
            is_high=True,
            is_low=False,
            term=PivotTerm.ST,
        )
        self._classify_st_hh_lh(pivot)

        self._st.insert(0, pivot)
        self._sth.insert(0, pivot)

        return pivot

    def _add_st_low(
        self, price: float, bar_idx: int,
        times: np.ndarray | None, current_idx: int
    ) -> Pivot | None:
        """Them STL moi vao arrays."""
        n = len(self._st)
        if n >= MAX_BUFFER:
            old = self._st.pop()
            if old in self._stl:
                self._stl.remove(old)

        pivot = Pivot(
            index=bar_idx,
            time_ms=int(times[bar_idx]) if times is not None else 0,
            price=price,
            is_high=False,
            is_low=True,
            term=PivotTerm.ST,
        )
        self._classify_st_ll_hl(pivot)

        self._st.insert(0, pivot)
        self._stl.insert(0, pivot)

        return pivot

    # =========================================================================
    # FindIT - promote ST -> IT (Pine Script lines 782-808)
    # =========================================================================
    def find_it(self) -> list[Pivot]:
        """Kiem tra va promote STH -> ITH, STL -> ITL.

        Pine Script FindIT:
          if STH.size() > 3:
            h1=STH.first(), h2=STH.get(1), h3=STH.get(SkipEQPivot(2))
            if h2.price > h3.price and h2.price > h1.price and not h2.isIT:
                h2.isIT = True; h2.isIHigherHigh = ITH.first().price <= h2.price
                ITH.unshift(h2)

          if STL.size() > 2:
            l1=STL.first(), l2=STL.get(1), l3=STL.get(SkipEQPivot(2))
            if l2.price < l3.price and l2.price < l1.price and not l2.isIT:
                l2.isIT = True; l2.isILowerLow = ITL.first().price >= l2.price
                ITL.unshift(l2)
        """
        promoted: list[Pivot] = []

        # STH -> ITH
        if len(self._sth) > 3:
            h1 = self._sth[0]
            h2 = self._sth[1]
            h3_idx = self.skip_eq_pivot(2, self._sth)
            h3 = self._sth[h3_idx]

            if (h2.price > h3.price and h2.price > h1.price and
                    not h2.is_i_higher_high and not h2.is_i_lower_low):
                h2.term = PivotTerm.IT
                h2.is_i_higher_high = (
                    not self._ith or self._ith[0].price <= h2.price
                )
                if h2 not in self._ith:
                    self._ith.insert(0, h2)
                promoted.append(h2)

        # STL -> ITL
        if len(self._stl) > 2:
            l1 = self._stl[0]
            l2 = self._stl[1]
            l3_idx = self.skip_eq_pivot(2, self._stl)
            l3 = self._stl[l3_idx]

            if (l2.price < l3.price and l2.price < l1.price and
                    not l2.is_i_higher_high and not l2.is_i_lower_low):
                l2.term = PivotTerm.IT
                l2.is_i_lower_low = (
                    not self._itl or self._itl[0].price >= l2.price
                )
                if l2 not in self._itl:
                    self._itl.insert(0, l2)
                promoted.append(l2)

        return promoted

    # =========================================================================
    # FindLT - promote IT -> LT (Pine Script lines 811-842)
    # =========================================================================
    def find_lt(self) -> list[Pivot]:
        """Kiem tra va promote ITH -> LTH, ITL -> LTL.

        Pine Script FindLT:
          if ITH.size() > 2:
            h1=ITH.first(), h2=ITH.get(1), h3=ITH.get(2)
            if h2.price > h3.price and h2.price > h1.price and not h2.isLT:
                h2.isLT = True; h2.isLHigherHigh = LTH.first().price <= h2.price
                LTH.unshift(h2)

          if ITL.size() > 2:
            l1=ITL.first(), l2=ITL.get(1), l3=ITL.get(2)
            if l2.price < l3.price and l2.price < l1.price and not l2.isLT:
                l2.isLT = True; l2.isLLowerLow = LTL.first().price >= l2.price
                LTL.unshift(l2)
        """
        promoted: list[Pivot] = []

        # ITH -> LTH
        if len(self._ith) > 2:
            h1 = self._ith[0]
            h2 = self._ith[1]
            h3 = self._ith[2]

            if (h2.price > h3.price and h2.price > h1.price and
                    not h2.is_l_higher_high and not h2.is_l_lower_low):
                h2.term = PivotTerm.LT
                h2.is_l_higher_high = (
                    not self._lth or self._lth[0].price <= h2.price
                )
                if h2 not in self._lth:
                    self._lth.insert(0, h2)
                promoted.append(h2)

        # ITL -> LTL
        if len(self._itl) > 2:
            l1 = self._itl[0]
            l2 = self._itl[1]
            l3 = self._itl[2]

            if (l2.price < l3.price and l2.price < l1.price and
                    not l2.is_l_higher_high and not l2.is_l_lower_low):
                l2.term = PivotTerm.LT
                l2.is_l_lower_low = (
                    not self._ltl or self._ltl[0].price >= l2.price
                )
                if l2 not in self._ltl:
                    self._ltl.insert(0, l2)
                promoted.append(l2)

        return promoted

    # =========================================================================
    # CheckClaimed - kiem tra pivot bi price sweep (Pine Script lines 856-890)
    # =========================================================================
    def check_claimed(
        self, highs: np.ndarray, lows: np.ndarray,
        current_time: int | None = None
    ) -> list[Pivot]:
        """Kiem tra va danh dau pivot bi claimed (liquidity sweep).

        Pine Script CheckClaimed:
          for pivot in ST:
            if not pivot.claimed:
              if pivot.isHigh and high > pivot.price
                 or pivot.isLow  and low  < pivot.price:
                pivot.claimed = True; pivot.time_last = time
        """
        if current_time is None:
            current_time = 0

        claimed: list[Pivot] = []
        current_high = highs[0] if len(highs) > 0 else 0.0
        current_low = lows[0] if len(lows) > 0 else 0.0

        for pivot in self._st:
            if pivot.claimed:
                continue
            if pivot.is_high and current_high > pivot.price:
                pivot.claimed = True
                pivot.time_last = current_time
                claimed.append(pivot)
            elif pivot.is_low and current_low < pivot.price:
                pivot.claimed = True
                pivot.time_last = current_time
                claimed.append(pivot)

        return claimed

    # =========================================================================
    # Public API
    # =========================================================================
    def detect(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        times: np.ndarray | None = None,
        current_time: int | None = None,
    ) -> list[Pivot]:
        """Chay day du pipeline: FindST -> FindIT -> FindLT -> CheckClaimed.

        Args:
            highs: mang high prices, index 0 = current bar
            lows: mang low prices
            times: mang timestamps (ms)
            current_time: timestamp hien tai (ms)

        Returns:
            Tat ca pivots moi duoc tao
        """
        self.find_st(highs, lows, times)
        promoted_it = self.find_it()
        promoted_lt = self.find_lt()
        claimed = self.check_claimed(highs, lows, current_time)
        return promoted_it + promoted_lt + claimed

    def get_all_pivots(self, include_claimed: bool = True) -> list[Pivot]:
        """Tra ve tat ca pivots tu _st array."""
        if include_claimed:
            return list(self._st)
        return [p for p in self._st if not p.claimed]

    def get_pivots_by_term(
        self, term: PivotTerm
    ) -> list[Pivot]:
        """Tra ve pivots theo term."""
        if term == PivotTerm.ST:
            return list(self._st)
        elif term == PivotTerm.IT:
            return list(self._ith)
        else:
            return list(self._lth)

    def get_sths(self) -> list[Pivot]:
        return list(self._sth)

    def get_stls(self) -> list[Pivot]:
        return list(self._stl)

    def get_unclaimed_sths(self) -> list[Pivot]:
        return [p for p in self._sth if not p.claimed]

    def get_unclaimed_stls(self) -> list[Pivot]:
        return [p for p in self._stl if not p.claimed]

    @property
    def sth_count(self) -> int:
        return len(self._sth)

    @property
    def ith_count(self) -> int:
        return len(self._ith)

    @property
    def lth_count(self) -> int:
        return len(self._lth)
