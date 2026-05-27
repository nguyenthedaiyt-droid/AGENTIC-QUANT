# =============================================================================
# AGENTIC-QUANT — Redis Connection Manager
# Async Redis pool voi retry logic, 8 namespaces, JSON/MessagePack serialization
# =============================================================================

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncGenerator

import msgpack
import redis.asyncio as redis
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Key Namespace Constants
# =============================================================================
class RedisNamespace:
    """
    8 namespace pattern cho Redis keys.
    Dung lam prefix cho tat ca Redis keys trong he thong.
    """

    # Zone Registry - luu zone dang active
    # Pattern: zone:{symbol}:{tf}:{type}:{formed_unix_ms}
    ZONE = "zone"

    # AI Output Cache - luu prediction moi nhat
    # Pattern: ai:output:{symbol}:latest
    AI_OUTPUT = "ai:output"

    # Macro State - trang thai macro hien tai
    # Pattern: macro:state:{currency}
    MACRO_STATE = "macro:state"

    # Macro Events - danh sach su kien sap toi
    # Pattern: macro:events:{currency}:upcoming
    MACRO_EVENTS = "macro:events"

    # Debate Log - debate record cho bar hien tai
    # Pattern: debate:{symbol}:{bar_close_ts}
    DEBATE = "debate"

    # Feature Cache - feature vector da tinh toan
    # Pattern: features:{symbol}:{bar_close_ts}:{model_name}
    FEATURES = "features"

    # Latent Vector Cache - latent vector tu LSTM
    # Pattern: latent:{symbol}:{bar_close_ts}
    LATENT = "latent"

    # System Metrics - metrics hien tai
    # Pattern: metrics:{component}:latest
    METRICS = "metrics"


# =============================================================================
# TTL Constants (seconds)
# =============================================================================
class RedisTTL:
    """TTL cho tung namespace, theo spec V4.5."""

    ZONE = 86400  # 24h
    AI_OUTPUT = 120  # 2 min
    MACRO_STATE = 60  # 1 min
    MACRO_EVENTS = 21600  # 6h
    DEBATE = 3600  # 1h
    FEATURES = 3600  # 1h
    LATENT = 3600  # 1h
    METRICS = 300  # 5 min


# =============================================================================
# Default Redis Settings
# =============================================================================
DEFAULT_REDIS_HOST = "127.0.0.1"
DEFAULT_REDIS_PORT = 6379
DEFAULT_REDIS_DB = 0
DEFAULT_REDIS_MAXmemory_MB = 512
DEFAULT_POOL_SIZE = 20
DEFAULT_POOL_TIMEOUT = 5.0
RETRY_MAX_ATTEMPTS = 3


# =============================================================================
# Custom Exceptions
# =============================================================================
class RedisCacheError(Exception):
    """Base exception cho Redis cache operations."""
    pass


class RedisConnectionError(RedisCacheError):
    """Khi khong the ket noi den Redis."""
    pass


class RedisSerializationError(RedisCacheError):
    """Khi serialization/deserialization that bai."""
    pass


# =============================================================================
# Retry Strategy
# =============================================================================
def _retry_strategy() -> retry:
    """Retry strategy: 3 attempts, exponential backoff 2s/4s/8s."""
    return retry(
        retry=retry_if_exception_type((redis.ConnectionError, redis.TimeoutError, OSError)),
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            f"Redis connection failed, attempt {retry_state.attempt_number}/{RETRY_MAX_ATTEMPTS}, "
            f"retrying in {retry_state.next_action.sleep}s..."
        ),
    )


# =============================================================================
# JSON Encoder/Decoder
# =============================================================================
class _DecimalSafeEncoder(json.JSONEncoder):
    """JSON encoder xu ly float/int cua Redis values."""

    def default(self, obj: Any) -> Any:
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


def _json_dumps(obj: dict) -> bytes:
    """Serialize dict sang JSON bytes (cho Redis String)."""
    try:
        return json.dumps(obj, cls=_DecimalSafeEncoder).encode("utf-8")
    except (TypeError, ValueError) as e:
        raise RedisSerializationError(f"JSON dumps failed: {e}") from e


def _json_loads(data: bytes | str) -> dict:
    """Deserialize JSON bytes/str thanh dict."""
    try:
        if isinstance(data, bytes):
            return json.loads(data.decode("utf-8"))
        return json.loads(data)
    except (TypeError, ValueError) as e:
        raise RedisSerializationError(f"JSON loads failed: {e}") from e


def _msgpack_dumps(obj: list) -> bytes:
    """Serialize list[float] (vector) sang MessagePack bytes."""
    try:
        return msgpack.packb(obj, use_bin_type=True)
    except Exception as e:
        raise RedisSerializationError(f"MessagePack dumps failed: {e}") from e


def _msgpack_loads(data: bytes) -> list:
    """Deserialize MessagePack bytes thanh list[float]."""
    try:
        return msgpack.unpackb(data, raw=False)
    except Exception as e:
        raise RedisSerializationError(f"MessagePack loads failed: {e}") from e


# =============================================================================
# Redis Cache Manager
# =============================================================================
@dataclass
class RedisCacheManager:
    """
    Async Redis connection manager voi connection pool.

    Features:
    - Connection pooling voi redis.asyncio
    - Retry logic: 3 attempts, exponential backoff 2s/4s/8s
    - 8 key namespaces voi TTL chinh xac theo spec
    - JSON serialization cho dicts
    - MessagePack serialization cho vectors
    - Health check & metrics
    """

    host: str = DEFAULT_REDIS_HOST
    port: int = DEFAULT_REDIS_PORT
    db: int = DEFAULT_REDIS_DB
    password: str | None = None
    maxmemory_mb: int = DEFAULT_REDIS_MAXmemory_MB
    ssl: bool = False
    pool_size: int = DEFAULT_POOL_SIZE
    pool_timeout: float = DEFAULT_POOL_TIMEOUT

    _pool: redis.ConnectionPool | None = field(default=None, init=False, repr=False)
    _client: redis.Redis | None = field(default=None, init=False, repr=False)
    _connected: bool = field(default=False, init=False, repr=False)

    # =============================================================================
    # Lifecycle
    # =============================================================================
    async def connect(self) -> None:
        """Thiet lap connection pool va ping Redis."""
        if self._connected:
            logger.debug("Redis already connected")
            return

        try:
            self._pool = redis.ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                ssl=self.ssl,
                max_connections=self.pool_size,
                socket_timeout=self.pool_timeout,
                socket_connect_timeout=self.pool_timeout,
                decode_responses=False,
            )
            self._client = redis.Redis(connection_pool=self._pool)

            await self._client.ping()
            await self._configure_server()
            self._connected = True
            logger.info(
                f"Redis connected: {self.host}:{self.port}/{self.db} "
                f"(pool_size={self.pool_size}, maxmemory={self.maxmemory_mb}MB)"
            )
        except Exception as e:
            await self._cleanup()
            raise RedisConnectionError(f"Failed to connect to Redis: {e}") from e

    @_retry_strategy()
    async def _configure_server(self) -> None:
        """Cau hinh Redis server (maxmemory, eviction policy)."""
        if not self._client:
            raise RedisConnectionError("Client not initialized")

        await self._client.config_set("maxmemory", f"{self.maxmemory_mb}mb")
        await self._client.config_set("maxmemory-policy", "allkeys-lru")
        logger.debug(f"Redis server configured: maxmemory={self.maxmemory_mb}MB, policy=allkeys-lru")

    async def disconnect(self) -> None:
        """Dong ket noi pool."""
        await self._cleanup()
        logger.info("Redis disconnected")

    async def _cleanup(self) -> None:
        """Cleanup internal resources."""
        self._connected = False
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._pool:
            await self._pool.disconnect()
            self._pool = None

    @property
    def client(self) -> redis.Redis:
        """Tra ve Redis client, raise neu chua connect."""
        if not self._client:
            raise RedisConnectionError("Redis not connected. Call connect() first.")
        return self._client

    @property
    def is_connected(self) -> bool:
        """Kiem tra trang thai ket noi."""
        return self._connected

    # =============================================================================
    # Key Building Helpers
    # =============================================================================
    @staticmethod
    def zone_key(symbol: str, tf: str, zone_type: str, formed_unix_ms: int) -> str:
        """Build key: zone:{symbol}:{tf}:{type}:{formed}"""
        return f"{RedisNamespace.ZONE}:{symbol}:{tf}:{zone_type}:{formed_unix_ms}"

    @staticmethod
    def ai_output_key(symbol: str) -> str:
        """Build key: ai:output:{symbol}:latest"""
        return f"{RedisNamespace.AI_OUTPUT}:{symbol}:latest"

    @staticmethod
    def macro_state_key(currency: str) -> str:
        """Build key: macro:state:{currency}"""
        return f"{RedisNamespace.MACRO_STATE}:{currency}"

    @staticmethod
    def macro_events_key(currency: str) -> str:
        """Build key: macro:events:{currency}:upcoming"""
        return f"{RedisNamespace.MACRO_EVENTS}:{currency}:upcoming"

    @staticmethod
    def debate_key(symbol: str, bar_close_ts: int) -> str:
        """Build key: debate:{symbol}:{bar_close_ts}"""
        return f"{RedisNamespace.DEBATE}:{symbol}:{bar_close_ts}"

    @staticmethod
    def features_key(symbol: str, bar_close_ts: int, model_name: str) -> str:
        """Build key: features:{symbol}:{bar_close_ts}:{model_name}"""
        return f"{RedisNamespace.FEATURES}:{symbol}:{bar_close_ts}:{model_name}"

    @staticmethod
    def latent_key(symbol: str, bar_close_ts: int) -> str:
        """Build key: latent:{symbol}:{bar_close_ts}"""
        return f"{RedisNamespace.LATENT}:{symbol}:{bar_close_ts}"

    @staticmethod
    def metrics_key(component: str) -> str:
        """Build key: metrics:{component}:latest"""
        return f"{RedisNamespace.METRICS}:{component}:latest"

    # =============================================================================
    # Zone Registry Namespace
    # =============================================================================
    async def set_zone(self, key: str, zone_data: dict) -> None:
        """Luu zone vao Redis Hash voi TTL 24h."""
        processed = self._prepare_for_hset(zone_data)
        await self.client.hset(key, mapping=processed)
        await self.client.expire(key, RedisTTL.ZONE)

    async def get_zone(self, key: str) -> dict | None:
        """Doc zone tu Redis Hash."""
        data = await self.client.hgetall(key)
        if not data:
            return None
        return {k.decode(): v.decode() if isinstance(v, bytes) else v for k, v in data.items()}

    async def delete_zone(self, key: str) -> None:
        """Xoa zone khoi Redis."""
        await self.client.delete(key)

    async def get_all_zone_keys(self, symbol: str | None = None) -> list[str]:
        """
        Lay tat ca zone keys.
        Neu symbol duoc cung cap, chi tra ve zone cua symbol do.
        """
        if symbol:
            pattern = f"{RedisNamespace.ZONE}:{symbol}:*"
        else:
            pattern = f"{RedisNamespace.ZONE}:*"
        keys: list[str] = []
        async for key in self.client.scan_iter(match=pattern, count=1000):
            keys.append(key.decode() if isinstance(key, bytes) else key)
        return keys

    # =============================================================================
    # AI Output Cache Namespace
    # =============================================================================
    async def set_ai_output(self, symbol: str, output_data: dict) -> None:
        """Luu AI output moi nhat voi TTL 120s."""
        key = self.ai_output_key(symbol)
        await self.client.hset(key, mapping=self._prepare_for_hset(output_data))
        await self.client.expire(key, RedisTTL.AI_OUTPUT)

    async def get_ai_output(self, symbol: str) -> dict | None:
        """Doc AI output moi nhat."""
        key = self.ai_output_key(symbol)
        data = await self.client.hgetall(key)
        if not data:
            return None
        return {k.decode(): self._decode_hgetall_value(v) for k, v in data.items()}

    # =============================================================================
    # Macro State Namespace
    # =============================================================================
    async def set_macro_state(self, currency: str, state_data: dict) -> None:
        """Luu macro state voi TTL 60s."""
        key = self.macro_state_key(currency)
        await self.client.hset(key, mapping=self._prepare_for_hset(state_data))
        await self.client.expire(key, RedisTTL.MACRO_STATE)

    async def get_macro_state(self, currency: str) -> dict | None:
        """Doc macro state."""
        key = self.macro_state_key(currency)
        data = await self.client.hgetall(key)
        if not data:
            return None
        return {k.decode(): self._decode_hgetall_value(v) for k, v in data.items()}

    # =============================================================================
    # Macro Events Namespace
    # =============================================================================
    async def add_macro_event(self, currency: str, event_data: dict) -> None:
        """Them event vao danh sach upcoming events (List)."""
        key = self.macro_events_key(currency)
        packed = _json_dumps(event_data)
        await self.client.rpush(key, packed)
        await self.client.expire(key, RedisTTL.MACRO_EVENTS)

    async def get_macro_events(self, currency: str) -> list[dict]:
        """Doc tat ca upcoming events."""
        key = self.macro_events_key(currency)
        raw_list = await self.client.lrange(key, 0, -1)
        return [_json_loads(item) for item in raw_list]

    async def clear_macro_events(self, currency: str) -> None:
        """Xoa tat ca upcoming events."""
        await self.client.delete(self.macro_events_key(currency))

    # =============================================================================
    # Debate Log Namespace
    # =============================================================================
    async def set_debate(self, symbol: str, bar_close_ts: int, debate_data: dict) -> None:
        """Luu debate record voi TTL 1h."""
        key = self.debate_key(symbol, bar_close_ts)
        await self.client.hset(key, mapping=self._prepare_for_hset(debate_data))
        await self.client.expire(key, RedisTTL.DEBATE)

    async def get_debate(self, symbol: str, bar_close_ts: int) -> dict | None:
        """Doc debate record."""
        key = self.debate_key(symbol, bar_close_ts)
        data = await self.client.hgetall(key)
        if not data:
            return None
        return {k.decode(): self._decode_hgetall_value(v) for k, v in data.items()}

    async def get_pending_debates_for_archive(self, symbol: str | None = None) -> list[tuple[str, int, dict]]:
        """
        Tim cac debate can archive (archived != True AND TTL < 600s).
        Tra ve list of (symbol, bar_close_ts, debate_data).
        """
        results: list[tuple[str, int, dict]] = []
        pattern = f"{RedisNamespace.DEBATE}:{symbol + ':' if symbol else ''}*"
        async for key_bytes in self.client.scan_iter(match=pattern, count=100):
            key = key_bytes.decode() if isinstance(key_bytes, bytes) else key_bytes

            # Bo qua key 'latest'
            if key.endswith(":latest"):
                continue

            # Kiem tra archived flag
            archived = await self.client.hget(key, "archived")
            if archived and self._decode_val(archived) == "True":
                continue

            # Kiem tra TTL
            ttl = await self.client.ttl(key)
            if ttl > 600 or ttl < 0:
                continue

            data = await self.client.hgetall(key)
            if data:
                decoded = {k2.decode(): self._decode_hgetall_value(v2) for k2, v2 in data.items()}
                parts = key.split(":")
                bar_ts = int(parts[-1]) if parts[-1].isdigit() else 0
                sym = parts[1] if len(parts) > 1 else symbol or ""
                results.append((sym, bar_ts, decoded))

        return results

    async def mark_debate_archived(self, symbol: str, bar_close_ts: int) -> None:
        """Danh dau debate da duoc archive."""
        key = self.debate_key(symbol, bar_close_ts)
        await self.client.hset(key, "archived", "True")

    # =============================================================================
    # Feature Cache Namespace
    # =============================================================================
    async def set_features(
        self,
        symbol: str,
        bar_close_ts: int,
        model_name: str,
        features_data: dict,
    ) -> None:
        """Luu feature vector dict voi TTL 1h."""
        key = self.features_key(symbol, bar_close_ts, model_name)
        await self.client.hset(key, mapping=self._prepare_for_hset(features_data))
        await self.client.expire(key, RedisTTL.FEATURES)

    async def get_features(
        self,
        symbol: str,
        bar_close_ts: int,
        model_name: str,
    ) -> dict | None:
        """Doc feature vector."""
        key = self.features_key(symbol, bar_close_ts, model_name)
        data = await self.client.hgetall(key)
        if not data:
            return None
        return {k.decode(): self._decode_hgetall_value(v) for k, v in data.items()}

    # =============================================================================
    # Latent Vector Namespace
    # =============================================================================
    async def set_latent_vector(self, symbol: str, bar_close_ts: int, vector: list[float]) -> None:
        """Luu latent vector (MessagePack) voi TTL 1h."""
        key = self.latent_key(symbol, bar_close_ts)
        packed = _msgpack_dumps(vector)
        await self.client.set(key, packed, ex=RedisTTL.LATENT)

    async def get_latent_vector(self, symbol: str, bar_close_ts: int) -> list[float] | None:
        """Doc latent vector (MessagePack)."""
        key = self.latent_key(symbol, bar_close_ts)
        data = await self.client.get(key)
        if not data:
            return None
        return _msgpack_loads(data)

    # =============================================================================
    # System Metrics Namespace
    # =============================================================================
    async def set_metrics(self, component: str, metrics_data: dict) -> None:
        """Luu system metrics voi TTL 5 min."""
        key = self.metrics_key(component)
        await self.client.hset(key, mapping=self._prepare_for_hset(metrics_data))
        await self.client.expire(key, RedisTTL.METRICS)

    async def get_metrics(self, component: str) -> dict | None:
        """Doc system metrics."""
        key = self.metrics_key(component)
        data = await self.client.hgetall(key)
        if not data:
            return None
        return {k.decode(): self._decode_hgetall_value(v) for k, v in data.items()}

    # =============================================================================
    # Sorted Set - Zone Ranking
    # =============================================================================
    async def update_zone_rank(
        self,
        symbol: str,
        zone_key: str,
        score: float,
        zone_type: str | None = None,
    ) -> None:
        """
        Cap nhat sorted set cho zone ranking.
        Score = p_hold * w_zone.
        """
        set_key = f"{RedisNamespace.ZONE}:rank:{symbol}"
        if zone_type:
            set_key = f"{set_key}:{zone_type}"
        await self.client.zadd(set_key, {zone_key: score})

    async def get_top_zones_by_rank(
        self,
        symbol: str,
        zone_type: str | None = None,
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Lay top-k zones theo rank score (p_hold * w_zone).
        Tra ve list of (zone_key, score).
        """
        set_key = f"{RedisNamespace.ZONE}:rank:{symbol}"
        if zone_type:
            set_key = f"{set_key}:{zone_type}"
        results = await self.client.zrevrange(set_key, 0, k - 1, withscores=True)
        return [(k.decode() if isinstance(k, bytes) else k, s) for k, s in results]

    # =============================================================================
    # Health & Diagnostics
    # =============================================================================
    @_retry_strategy()
    async def ping(self) -> bool:
        """Ping Redis - kiem tra ket noi."""
        return await self.client.ping()

    async def info(self) -> dict:
        """Tra ve Redis INFO."""
        raw = await self.client.info("memory")
        if isinstance(raw, bytes):
            raw = raw.decode()
        return raw if isinstance(raw, dict) else {}

    async def get_memory_usage(self) -> dict[str, Any]:
        """Tra ve memory usage statistics."""
        info = await self.info()
        used = info.get("used_memory", 0)
        maxmem = info.get("maxmemory", self.maxmemory_mb * 1024 * 1024)
        return {
            "used_bytes": used,
            "used_mb": round(used / (1024 * 1024), 2),
            "max_bytes": maxmem,
            "max_mb": round(maxmem / (1024 * 1024), 2),
            "pct": round(used / maxmem * 100, 2) if maxmem > 0 else 0,
        }

    async def get_key_counts(self) -> dict[str, int]:
        """Dem so key theo namespace."""
        counts: dict[str, int] = {}
        namespaces = [
            RedisNamespace.ZONE,
            RedisNamespace.AI_OUTPUT,
            RedisNamespace.MACRO_STATE,
            RedisNamespace.MACRO_EVENTS,
            RedisNamespace.DEBATE,
            RedisNamespace.FEATURES,
            RedisNamespace.LATENT,
            RedisNamespace.METRICS,
        ]
        for ns in namespaces:
            count = 0
            async for _ in self.client.scan_iter(match=f"{ns}:*", count=1000):
                count += 1
            counts[ns] = count
        return counts

    # =============================================================================
    # Internal Helpers
    # =============================================================================
    @staticmethod
    def _prepare_for_hset(data: dict) -> dict:
        """Chuan bi dict cho HSET - loc None values va convert sang string."""
        result: dict = {}
        for k, v in data.items():
            if v is None:
                v_str = "null"
            elif isinstance(v, (dict, list)):
                v_bytes = _json_dumps(v)
                v_str = v_bytes.decode("utf-8") if isinstance(v_bytes, bytes) else v_bytes
            elif isinstance(v, bool):
                v_str = "True" if v else "False"
            elif isinstance(v, (int, float)):
                v_str = str(v)
            else:
                v_str = str(v)
            result[str(k)] = v_str
        return result

    @staticmethod
    def _decode_val(value: bytes | str | int | float | None) -> Any:
        """Decode mot gia tri don le tu Redis."""
        if value is None:
            return None
        if isinstance(value, bytes):
            s = value.decode("utf-8")
            if s == "True":
                return True
            if s == "False":
                return False
            try:
                if "." in s:
                    return float(s)
                return int(s)
            except ValueError:
                return s
        return value

    @staticmethod
    def _decode_hgetall_value(value: bytes) -> Any:
        """Decode gia tri tu HGETALL."""
        if value is None:
            return None
        if isinstance(value, bytes):
            s = value.decode("utf-8")
            if s.startswith("{"):
                try:
                    return _json_loads(s)
                except Exception:
                    pass
            if s == "True":
                return True
            if s == "False":
                return False
            try:
                if "." in s:
                    return float(s)
                return int(s)
            except ValueError:
                return s
        return value


# =============================================================================
# Global Singleton
# =============================================================================
_redis_manager: RedisCacheManager | None = None


def get_redis_cache_manager() -> RedisCacheManager:
    """Tra ve singleton RedisCacheManager instance."""
    global _redis_manager
    if _redis_manager is None:
        _redis_manager = RedisCacheManager()
    return _redis_manager


async def init_redis(
    host: str = DEFAULT_REDIS_HOST,
    port: int = DEFAULT_REDIS_PORT,
    db: int = DEFAULT_REDIS_DB,
    password: str | None = None,
    maxmemory_mb: int = DEFAULT_REDIS_MAXmemory_MB,
) -> RedisCacheManager:
    """Khoi tao va ket noi Redis, tra ve singleton."""
    manager = get_redis_cache_manager()
    manager.host = host
    manager.port = port
    manager.db = db
    manager.password = password
    manager.maxmemory_mb = maxmemory_mb
    await manager.connect()
    return manager
