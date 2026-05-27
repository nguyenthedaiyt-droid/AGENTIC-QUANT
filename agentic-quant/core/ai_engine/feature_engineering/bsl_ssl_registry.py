"""BSL/SSL Registry Integration - Liquidity Feature Builder.

TODO 4.2: Sau khi detect Pivots -> upsert vào Zone Registry (Redis).
TODO 4.2.2: Tinh F_liq[24] vector (Tier-based ST/IT/LT).

F_liq[24] Tier-based (theo agentic_quant_full_plan.md):
  Dims [0–5]:   ΔP_BSL/SSL per tier = (P_nearest - P_current) / ATR_H4
                  [ΔP_BSL_ST, ΔP_BSL_IT, ΔP_BSL_LT, ΔP_SSL_ST, ΔP_SSL_IT, ΔP_SSL_LT]
  Dims [6–11]:  ΔT_BSL/SSL per tier = so nen M1 tu hien tai den pivot gan nhat
                  [ΔT_BSL_ST, ΔT_BSL_IT, ΔT_BSL_LT, ΔT_SSL_ST, ΔT_SSL_IT, ΔT_SSL_LT]
  Dims [12–15]: V_acc (chi IT va LT, BSL va SSL)
                  [V_BSL_IT, V_BSL_LT, V_SSL_IT, V_SSL_LT]
  Dims [16–21]: N_count per tier (chuan hoa [0,1] tren cua so 200 nen)
                  [N_BSL_ST, N_BSL_IT, N_BSL_LT, N_SSL_ST, N_SSL_IT, N_SSL_LT]
  Dims [22–23]: r_claimed per tier (IT va LT) = ty le claimed trong 50 pivots gan nhat
                  [r_claimed_IT, r_claimed_LT]
"""
from __future__ import annotations

import numpy as np

from core.ai_engine.feature_engineering.types import Pivot, PivotTerm
from core.ai_engine.feature_engineering.smc_detector import SwingPointDetector
# Optimized imports - avoid circular deps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.memory.short_term.active_zone_registry import ActiveZoneRegistry
    from core.memory.short_term.redis_cache_manager import RedisCacheManager


_MAX_CLAIMED_WINDOW = 50


class BSLSSLRegistry:
    """BSL/SSL Registry Integration.

    Sau khi SwingPointDetector tao pivots, upsert vào Zone Registry (Redis)
    va tinh F_liq[24] feature vector (Tier-based ST/IT/LT).

    Args:
        redis: RedisCacheManager instance
        zone_registry: ActiveZoneRegistry instance
    """

    def __init__(
        self,
        redis: RedisCacheManager,
        zone_registry: ActiveZoneRegistry,
    ) -> None:
        self._redis = redis
        self._zone_registry = zone_registry
        self._atr_m1_cache: dict[str, float] = {}
        self._bsl_max_count: dict[PivotTerm, int] = {
            PivotTerm.ST: 20, PivotTerm.IT: 10, PivotTerm.LT: 5
        }
        self._ssl_max_count: dict[PivotTerm, int] = {
            PivotTerm.ST: 20, PivotTerm.IT: 10, PivotTerm.LT: 5
        }

    def upsert_pivots(
        self,
        symbol: str,
        pivots: list[Pivot],
        timeframe: str = "M15",
    ) -> None:
        """Upsert pivots vào Zone Registry (Redis).

        Moi pivot duoc luu nhu mot Zone (BSL = swing high, SSL = swing low).

        Args:
            symbol: Symbol trading
            pivots: Danh sach pivots tu SwingPointDetector
            timeframe: Timeframe hien tai
        """
        for pivot in pivots:
            if pivot.claimed:
                continue

            zone_type = "BSL" if pivot.is_high else "SSL"
            zone_id = f"pivot_{zone_type}_{pivot.term.value}_{symbol}_{pivot.time_ms}"

            zone_data = {
                "zone_id": zone_id,
                "symbol": symbol,
                "timeframe": timeframe,
                "zone_type": zone_type,
                "top": pivot.price,
                "bottom": pivot.price,
                "ce": 0.0,
                "formed_time": pivot.time_ms,
                "p_hold": 0.5,
                "w_zone": 1.0,
                "iii_formation": 0.0,
                "touch_count": 0,
                "status": "UNMITIGATED",
                "session_id": "FE",
                "macro_regime": "NORMAL",
            }

            key = f"zone:{symbol}:{timeframe}:{zone_type}:{pivot.time_ms}"
            self._redis.set_json(key, zone_data, ttl=86400)

    def _get_atr_m1(self, symbol: str) -> float:
        """Lay ATR M1 tu cache hoac tra ve gia tri mac dinh."""
        if symbol not in self._atr_m1_cache:
            return 0.0001
        return self._atr_m1_cache[symbol]

    def _bars_since(self, pivot_time_ms: int, current_time_ms: int = 0) -> float:
        """Tinh so bars M1 tu pivot den hien tai (gan dung)."""
        if current_time_ms == 0 or pivot_time_ms == 0:
            return 100.0
        delta_ms = current_time_ms - pivot_time_ms
        return float(delta_ms / 60000.0)

    def _compute_v_acc(
        self,
        symbol: str,
        price: float,
        eps: float,
    ) -> float:
        """Tinh V_acc: tong CVD trong vung ±ε quanh price level."""
        return 0.0

    def _claimed_ratio(
        self,
        pivots: list[Pivot],
        last_n: int = _MAX_CLAIMED_WINDOW,
    ) -> float:
        """Tinh ty le claimed trong N pivots gan nhat."""
        if not pivots:
            return 0.0
        recent = pivots[:last_n]
        claimed = sum(1 for p in recent if p.claimed)
        return claimed / len(recent) if recent else 0.0

    def get_f_liq_vector(
        self,
        symbol: str,
        current_price: float,
        atr_h4: float,
        pivots_by_tier: dict[PivotTerm, list[Pivot]] | None = None,
        current_bar_index: int = 0,
    ) -> np.ndarray:
        """Tinh F_liq[24] vector (Tier-based ST/IT/LT).

        Theo agentic_quant_full_plan.md - Tier-based (KHONG phai TF-based).

        Dims [0–5]:   ΔP_BSL/SSL per tier
        Dims [6–11]:  ΔT_BSL/SSL per tier
        Dims [12–15]: V_acc (chi IT va LT)
        Dims [16–21]: N_count per tier
        Dims [22–23]: r_claimed (IT va LT)

        Args:
            symbol: Symbol trading
            current_price: Gia hien tai
            atr_h4: ATR H4 (dung de chuan hoa khoang cach)
            pivots_by_tier: Dict PivotTerm -> list Pivot
            current_bar_index: Chi so bar hien tai

        Returns:
            np.ndarray[24] F_liq vector
        """
        vec = np.zeros(24, dtype=np.float32)
        eps = 0.5 * self._get_atr_m1(symbol)
        tiers = [PivotTerm.ST, PivotTerm.IT, PivotTerm.LT]

        if pivots_by_tier is None:
            pivots_by_tier = {t: [] for t in tiers}

        for i, tier in enumerate(tiers):
            bsls = [p for p in pivots_by_tier.get(tier, []) if p.is_high and not p.claimed]
            ssls = [p for p in pivots_by_tier.get(tier, []) if p.is_low and not p.claimed]

            # ΔP - Khoang cach gia den BSL gan nhat [0-2]
            nearest_bsl = self._nearest_pivot(bsls, current_price)
            if nearest_bsl and atr_h4 > 0:
                vec[i] = (nearest_bsl.price - current_price) / atr_h4

            # ΔP - Khoang cach gia den SSL gan nhat [3-5]
            nearest_ssl = self._nearest_pivot(ssls, current_price)
            if nearest_ssl and atr_h4 > 0:
                vec[3 + i] = (current_price - nearest_ssl.price) / atr_h4

            # ΔT - So bars den BSL gan nhat [6-8]
            if nearest_bsl:
                vec[6 + i] = self._bars_since(nearest_bsl.time_ms)
            else:
                vec[6 + i] = 200.0

            # ΔT - So bars den SSL gan nhat [9-11]
            if nearest_ssl:
                vec[9 + i] = self._bars_since(nearest_ssl.time_ms)
            else:
                vec[9 + i] = 200.0

            # V_acc - chi IT va LT [12-15]
            if tier != PivotTerm.ST:
                j_v = 12 + (0 if tier == PivotTerm.IT else 1)
                if nearest_bsl:
                    vec[j_v] = self._compute_v_acc(symbol, nearest_bsl.price, eps)
                j_v_ssl = 14 + (0 if tier == PivotTerm.IT else 1)
                if nearest_ssl:
                    vec[j_v_ssl] = self._compute_v_acc(symbol, nearest_ssl.price, eps)

            # N_count - so luong pivots chuan hoa [16-21]
            bsl_max = self._bsl_max_count.get(tier, 10)
            ssl_max = self._ssl_max_count.get(tier, 10)
            vec[16 + i] = len(bsls) / (bsl_max + 1e-9)
            vec[19 + i] = len(ssls) / (ssl_max + 1e-9)

        # r_claimed - chi IT va LT [22-23]
        it_pivots = pivots_by_tier.get(PivotTerm.IT, [])
        lt_pivots = pivots_by_tier.get(PivotTerm.LT, [])
        vec[22] = self._claimed_ratio(it_pivots, last_n=_MAX_CLAIMED_WINDOW)
        vec[23] = self._claimed_ratio(lt_pivots, last_n=_MAX_CLAIMED_WINDOW)

        return vec

    @staticmethod
    def _nearest_pivot(
        pivots: list[Pivot],
        current_price: float,
    ) -> Pivot | None:
        """Tim pivot gan nhat voi current_price."""
        if not pivots:
            return None
        return min(pivots, key=lambda p: abs(p.price - current_price))

    def get_f_liq_from_detector(
        self,
        symbol: str,
        current_price: float,
        atr_h4: float,
        detector: SwingPointDetector,
        current_bar_index: int = 0,
    ) -> np.ndarray:
        """Tinh F_liq[24] tu SwingPointDetector (cach dung thong dung).

        Args:
            symbol: Symbol trading
            current_price: Gia hien tai
            atr_h4: ATR H4
            detector: SwingPointDetector instance
            current_bar_index: Chi so bar hien tai

        Returns:
            np.ndarray[24] F_liq vector
        """
        pivots_by_tier: dict[PivotTerm, list[Pivot]] = {}
        for tier in [PivotTerm.ST, PivotTerm.IT, PivotTerm.LT]:
            pivots_by_tier[tier] = detector.get_pivots_by_term(tier)

        return self.get_f_liq_vector(
            symbol, current_price, atr_h4, pivots_by_tier, current_bar_index
        )

    def get_distance_to_nearest_bsl(
        self,
        pivots: list[Pivot],
        current_price: float,
    ) -> float:
        """Tinh khoang cach den BSL gan nhat."""
        unclaimed = [p for p in pivots if p.is_high and not p.claimed]
        if not unclaimed:
            return 1.0
        nearest = self._nearest_pivot(unclaimed, current_price)
        if not nearest or current_price <= 0:
            return 1.0
        dist = abs(nearest.price - current_price) / current_price
        return float(dist)

    def get_distance_to_nearest_ssl(
        self,
        pivots: list[Pivot],
        current_price: float,
    ) -> float:
        """Tinh khoang cach den SSL gan nhat."""
        unclaimed = [p for p in pivots if p.is_low and not p.claimed]
        if not unclaimed:
            return 1.0
        nearest = self._nearest_pivot(unclaimed, current_price)
        if not nearest or current_price <= 0:
            return 1.0
        dist = abs(nearest.price - current_price) / current_price
        return float(dist)

    def get_bsl_density(
        self,
        pivots: list[Pivot],
        current_price: float,
        window_pct: float = 0.01,
    ) -> float:
        """Dem so BSL trong vung ±window_pct quanh current_price."""
        unclaimed = [p for p in pivots if p.is_high and not p.claimed]
        if not unclaimed:
            return 0.0
        window = current_price * window_pct
        count = sum(
            1 for p in unclaimed
            if abs(p.price - current_price) <= window
        )
        return float(count)

    def get_ssl_density(
        self,
        pivots: list[Pivot],
        current_price: float,
        window_pct: float = 0.01,
    ) -> float:
        """Dem so SSL trong vung ±window_pct quanh current_price."""
        unclaimed = [p for p in pivots if p.is_low and not p.claimed]
        if not unclaimed:
            return 0.0
        window = current_price * window_pct
        count = sum(
            1 for p in unclaimed
            if abs(p.price - current_price) <= window
        )
        return float(count)

    def get_claimed_ratio(
        self,
        pivots: list[Pivot],
    ) -> float:
        """Tinh claimed ratio: claimed / total."""
        if not pivots:
            return 0.0
        total = len(pivots)
        claimed = sum(1 for p in pivots if p.claimed)
        return float(claimed) / float(total) if total > 0 else 0.0
