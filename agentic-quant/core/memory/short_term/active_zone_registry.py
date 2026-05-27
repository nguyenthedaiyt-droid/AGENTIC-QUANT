# =============================================================================
# AGENTIC-QUANT — Active Zone Registry
# Quan ly cac zone dang active trong Redis
# =============================================================================

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from core.memory.short_term.redis_cache_manager import RedisCacheManager
from core.memory.short_term.redis_cache_manager import RedisNamespace
from core.memory.models.zone import Zone
from core.memory.models.enums import ZoneStatus, ZoneType

if TYPE_CHECKING:
    pass


# =============================================================================
# Active Zone Registry
# =============================================================================
class ActiveZoneRegistry:
    """
    Quan ly cac zone dang active (chua bi mitigated).
    Luu trong Redis Hash + Sorted Set cho ranking.

    Methods:
    - upsert_zone(zone: Zone) -> None
    - get_zones_near_price(symbol, price, window_pips) -> list[Zone]
    - update_zone_status(zone_id, new_status) -> None
    - update_zone_p_hold(zone_id, p_hold) -> None
    - get_top_zones(symbol, zone_type, k=5) -> list[Zone]

    Redis Keys:
    - zone:{symbol}:{tf}:{type}:{formed}  (Hash) - TTL 24h
    - zone:rank:{symbol}                  (Sorted Set) - score = p_hold * w_zone
    """

    def __init__(self, redis: RedisCacheManager) -> None:
        self._redis = redis

    # =========================================================================
    # Upsert
    # =========================================================================
    async def upsert_zone(self, zone: Zone) -> bool:
        """
        Chen hoac cap nhat zone trong Redis.
        Tra ve True neu la zone moi, False neu la update.

        Flow:
        1. Check xem zone da ton tai chua
        2. Neu ton tai -> update (khong tao su kien)
        3. Neu moi -> insert + add vao sorted set
        """
        key = self._zone_key(zone)

        # Kiem tra ton tai
        existing = await self._redis.get_zone(key)
        is_new = existing is None

        # Serialize zone data
        zone_data = zone.to_dict()

        # Luu vao Redis Hash
        await self._redis.set_zone(key, zone_data)

        # Cap nhat sorted set ranking
        score = zone.p_hold * zone.w_zone
        zone_type_str = zone.zone_type.value if hasattr(zone.zone_type, "value") else str(zone.zone_type)
        await self._redis.update_zone_rank(zone.symbol, key, score, zone_type_str)

        if is_new:
            logger.debug(
                f"ZoneRegistry: inserted new zone {zone.id} "
                f"({zone.zone_type}) for {zone.symbol} @ {zone.timeframe}"
            )
        else:
            logger.debug(
                f"ZoneRegistry: updated zone {zone.id} "
                f"p_hold={zone.p_hold:.3f}, status={zone.status}"
            )

        return is_new

    # =========================================================================
    # Query - Near Price
    # =========================================================================
    async def get_zones_near_price(
        self,
        symbol: str,
        price: float,
        window_pips: float,
        timeframe: str | None = None,
        zone_types: list[str] | None = None,
    ) -> list[Zone]:
        """
        Lay tat ca zones gan gia hien tai (trong khoang window_pips).

        Giai thuat:
        1. Lay tat ca zone keys cho symbol
        2. Filter theo timeframe neu duoc cung cap
        3. Filter theo zone_type neu duoc cung cap
        4. Filter theo khoang gia price +/- window_pips
        5. Chi tra ve zones con active (chua mitigated)

        Args:
            symbol: Symbol can query
            price: Gia hien tai (pips)
            window_pips: Khoang gia de tim zone (pips)
            timeframe: Loc theo timeframe (None = tat ca)
            zone_types: Loc theo zone_type (None = tat ca)

        Returns:
            List Zone objects, sorted by p_hold descending
        """
        all_keys = await self._redis.get_all_zone_keys(symbol)
        result: list[tuple[Zone, float]] = []

        for key in all_keys:
            zone_data = await self._redis.get_zone(key)
            if not zone_data:
                continue

            zone = self._dict_to_zone(zone_data)
            if zone is None:
                continue

            # Bo qua zone da mitigated
            if not zone.is_active():
                continue

            # Loc theo timeframe
            if timeframe:
                tf_val = zone.timeframe.value if hasattr(zone.timeframe, "value") else str(zone.timeframe)
                if tf_val != timeframe:
                    continue

            # Loc theo zone_type
            if zone_types:
                zt_val = zone.zone_type.value if hasattr(zone.zone_type, "value") else str(zone.zone_type)
                if zt_val not in zone_types:
                    continue

            # Kiem tra khoang gia
            zone_top = zone.top
            zone_bottom = zone.bottom

            # Mo rong khoang tim kiem bao gom ca zone
            if price < zone_bottom - window_pips or price > zone_top + window_pips:
                continue

            score = zone.p_hold * zone.w_zone
            result.append((zone, score))

        # Sort by score descending
        result.sort(key=lambda x: x[1], reverse=True)
        return [z for z, _ in result]

    # =========================================================================
    # Query - Top Zones
    # =========================================================================
    async def get_top_zones(
        self,
        symbol: str,
        zone_type: str | None = None,
        k: int = 5,
    ) -> list[Zone]:
        """
        Lay top-k zones theo p_hold * w_zone score.
        Dung Sorted Set de lay nhanh, khong can scan het key space.
        """
        ranked = await self._redis.get_top_zones_by_rank(symbol, zone_type, k)
        zones: list[Zone] = []

        for key, score in ranked:
            zone_data = await self._redis.get_zone(key)
            if zone_data:
                zone = self._dict_to_zone(zone_data)
                if zone and zone.is_active():
                    zones.append(zone)

        return zones

    # =========================================================================
    # Update - Status
    # =========================================================================
    async def update_zone_status(
        self,
        zone_id: str,
        symbol: str,
        timeframe: str,
        formed_time: int,
        zone_type: str,
        new_status: ZoneStatus,
    ) -> bool:
        """
        Cap nhat trang thai zone.
        Tra ve True neu cap nhat thanh cong, False neu zone khong ton tai.
        """
        key = self._build_key(symbol, timeframe, zone_type, formed_time)
        existing = await self._redis.get_zone(key)
        if not existing:
            logger.warning(f"ZoneRegistry: zone {zone_id} not found for status update")
            return False

        old_status = existing.get("status", "")
        new_status_val = new_status.value if hasattr(new_status, "value") else str(new_status)

        # Update Redis Hash
        await self._redis.client.hset(key, "status", new_status_val)
        await self._redis.client.hset(key, "status_updated", str(int(datetime.utcnow().timestamp() * 1000)))

        logger.debug(
            f"ZoneRegistry: zone {zone_id} status {old_status} -> {new_status_val}"
        )

        # Neu mitigated -> xoa khoi sorted set (score = 0)
        if new_status == ZoneStatus.MITIGATED:
            await self._redis.update_zone_rank(symbol, key, 0.0, zone_type)
            logger.debug(f"ZoneRegistry: zone {zone_id} removed from ranking (MITIGATED)")

        return True

    # =========================================================================
    # Update - P Hold
    # =========================================================================
    async def update_zone_p_hold(
        self,
        zone_id: str,
        symbol: str,
        timeframe: str,
        formed_time: int,
        zone_type: str,
        p_hold: float,
        w_zone: float | None = None,
    ) -> bool:
        """
        Cap nhat p_hold cho zone.
        Tu dong cap nhat sorted set score = p_hold * w_zone.
        Tra ve True neu cap nhat thanh cong.
        """
        key = self._build_key(symbol, timeframe, zone_type, formed_time)
        existing = await self._redis.get_zone(key)
        if not existing:
            logger.warning(f"ZoneRegistry: zone {zone_id} not found for p_hold update")
            return False

        now_ms = int(datetime.utcnow().timestamp() * 1000)
        await self._redis.client.hset(key, "p_hold", str(p_hold))
        await self._redis.client.hset(key, "p_hold_updated", str(now_ms))

        if w_zone is None:
            w_zone_str = existing.get("w_zone", "1.0")
            try:
                w_zone = float(w_zone_str)
            except (ValueError, TypeError):
                w_zone = 1.0

        # Cap nhat sorted set score
        await self._redis.update_zone_rank(symbol, key, p_hold * w_zone, zone_type)

        logger.debug(
            f"ZoneRegistry: zone {zone_id} p_hold updated to {p_hold:.3f} "
            f"(score={p_hold * w_zone:.3f})"
        )
        return True

    # =========================================================================
    # Touch Tracking
    # =========================================================================
    async def record_zone_touch(
        self,
        symbol: str,
        timeframe: str,
        formed_time: int,
        zone_type: str,
        touch_time: int,
    ) -> None:
        """Ghi nhan zone bi cham (touch). Tang touch_count va cap nhat last_touch_time."""
        key = self._build_key(symbol, timeframe, zone_type, formed_time)
        existing = await self._redis.get_zone(key)
        if not existing:
            return

        old_count = int(existing.get("touch_count", 0))
        await self._redis.client.hset(key, "touch_count", str(old_count + 1))
        await self._redis.client.hset(key, "last_touch_time", str(touch_time))

    # =========================================================================
    # Bulk Operations
    # =========================================================================
    async def get_all_active_zones(self, symbol: str) -> list[Zone]:
        """Lay tat ca zones active cho symbol (khong phan bi zone_type)."""
        return await self.get_zones_near_price(symbol, price=0.0, window_pips=float("inf"))

    async def get_active_count(self, symbol: str) -> int:
        """Dem so zones active cho symbol."""
        zones = await self.get_all_active_zones(symbol)
        return len(zones)

    # =========================================================================
    # Internal Helpers
    # =========================================================================
    def _zone_key(self, zone: Zone) -> str:
        """Build Redis key tu Zone object."""
        tf = zone.timeframe.value if hasattr(zone.timeframe, "value") else str(zone.timeframe)
        zt = zone.zone_type.value if hasattr(zone.zone_type, "value") else str(zone.zone_type)
        return self._build_key(zone.symbol, tf, zt, zone.formed_time)

    @staticmethod
    def _build_key(symbol: str, timeframe: str, zone_type: str, formed_time: int) -> str:
        """Build zone key: zone:{symbol}:{tf}:{type}:{formed}"""
        return f"{RedisNamespace.ZONE}:{symbol}:{timeframe}:{zone_type}:{formed_time}"

    def _dict_to_zone(self, data: dict[str, Any]) -> Zone | None:
        """Chuyen dict thanh Zone object."""
        try:
            return Zone.from_dict(data)
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"ZoneRegistry: failed to deserialize zone: {e}")
            return None
