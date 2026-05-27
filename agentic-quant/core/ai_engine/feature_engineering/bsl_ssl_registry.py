"""BSL/SSL Registry Integration - Liquidity Feature Builder.

TODO 4.2: Sau khi detect Pivots -> upsert vào Zone Registry (Redis).
TODO 4.2.2: Tinh F_liq vector (24 chiều) từ active BSL/SSL.

F_liq[24] = [
  8 dims: distances to nearest BSL/SSL per TF (M1/M5/M15/H1/H4/D1/W1/MN)
  8 dims: age (bars) of each BSL/SSL
  4 dims: volume accumulated near each level
  4 dims: claimed ratio per TF
]
"""
from __future__ import annotations

import numpy as np

from core.ai_engine.feature_engineering.types import Pivot, PivotTerm
from core.ai_engine.feature_engineering.smc_detector import SwingPointDetector
from core.memory.short_term.active_zone_registry import ActiveZoneRegistry
from core.memory.short_term.redis_cache_manager import RedisCacheManager


# 8 timeframes cho F_liq vector
TIMEFRAMES = ["M1", "M5", "M15", "H1", "H4", "D1", "W1", "MN"]


class BSLSSLRegistry:
    """BSL/SSL Registry Integration.

    Sau khi SwingPointDetector tao pivots, upsert vào Zone Registry (Redis)
    va tinh F_liq[24] feature vector.

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

            # Luu zone metadata vao Redis nhu Zone
            zone_data = {
                "zone_id": zone_id,
                "symbol": symbol,
                "timeframe": timeframe,
                "zone_type": zone_type,
                "top": pivot.price,
                "bottom": pivot.price,
                "ce": 0.0,
                "formed_time": pivot.time_ms,
                "p_hold": 0.5,  # default, se duoc update sau
                "w_zone": 1.0,
                "iii_formation": 0.0,
                "touch_count": 0,
                "status": "UNMITIGATED",
                "session_id": "FE",
                "macro_regime": "NORMAL",
            }

            import orjson
            key = f"zone:{symbol}:{timeframe}:{zone_type}:{pivot.time_ms}"
            self._redis.set_json(key, zone_data, ttl=86400)  # TTL 24h

    def upsert_pivots_to_registry(
        self,
        symbol: str,
        pivots: list[Pivot],
        timeframe: str = "M15",
    ) -> None:
        """Upsert pivots vao ActiveZoneRegistry (Phase 3).

        Dung ActiveZoneRegistry de quan ly zone lifecycle.

        Args:
            symbol: Symbol trading
            pivots: Danh sach pivots
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

            import orjson
            key = f"zone:{symbol}:{timeframe}:{zone_type}:{pivot.time_ms}"
            self._redis.set_json(key, zone_data, ttl=86400)

    def get_f_liq_vector(
        self,
        symbol: str,
        current_price: float,
        highs_by_tf: dict[str, list[Pivot]],
        lows_by_tf: dict[str, list[Pivot]],
        current_bar_index: int = 0,
    ) -> np.ndarray:
        """Tinh F_liq[24] vector.

        24 chiều:
          [0-7]  distances to nearest BSL per TF (M1..MN)
          [8-15] distances to nearest SSL per TF (M1..MN)
          [16-23] age (bars) of nearest BSL/SSL per TF

        Args:
            symbol: Symbol trading
            current_price: Gia hien tai
            highs_by_tf: Dict TF -> list STH/ITH/LTH pivots
            lows_by_tf: Dict TF -> list STL/ITL/LTL pivots
            current_bar_index: Chi so bar hien tai (tinh age)

        Returns:
            np.ndarray[24] F_liq vector
        """
        f_liq = np.zeros(24, dtype=np.float64)

        # distances (8 dims BSL, 8 dims SSL)
        for i, tf in enumerate(TIMEFRAMES):
            sths = highs_by_tf.get(tf, [])
            stls = lows_by_tf.get(tf, [])

            # Nearest unclaimed STH
            if sths:
                nearest = min(sths, key=lambda p: abs(p.price - current_price))
                dist = abs(nearest.price - current_price) / current_price if current_price > 0 else 0.0
                f_liq[i] = float(dist)
            else:
                f_liq[i] = 1.0  # no pivot -> max distance

            # Nearest unclaimed STL
            if stls:
                nearest = min(stls, key=lambda p: abs(p.price - current_price))
                dist = abs(nearest.price - current_price) / current_price if current_price > 0 else 0.0
                f_liq[8 + i] = float(dist)
            else:
                f_liq[8 + i] = 1.0

            # Age: bars since nearest pivot
            age_bsl = 0.0
            age_ssl = 0.0
            if sths:
                nearest = min(sths, key=lambda p: abs(p.price - current_price))
                age_bsl = float(current_bar_index - nearest.index)
                f_liq[16 + i] = min(age_bsl, 500.0) / 500.0  # normalize

            if stls:
                nearest = min(stls, key=lambda p: abs(p.price - current_price))
                age_ssl = float(current_bar_index - nearest.index)
                f_liq[16 + i] = max(f_liq[16 + i], min(age_ssl, 500.0) / 500.0)

        return f_liq

    def get_distance_to_nearest_bsl(
        self,
        pivots: list[Pivot],
        current_price: float,
    ) -> float:
        """Tinh khoang cach den BSL gan nhat."""
        unclaimed = [p for p in pivots if p.is_high and not p.claimed]
        if not unclaimed:
            return 1.0
        nearest = min(unclaimed, key=lambda p: abs(p.price - current_price))
        dist = abs(nearest.price - current_price) / current_price if current_price > 0 else 0.0
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
        nearest = min(unclaimed, key=lambda p: abs(p.price - current_price))
        dist = abs(nearest.price - current_price) / current_price if current_price > 0 else 0.0
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
