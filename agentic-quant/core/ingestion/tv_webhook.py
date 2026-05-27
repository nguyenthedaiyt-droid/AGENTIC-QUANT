# =============================================================================
# AGENTIC-QUANT — TradingView Webhook Receiver
# HTTP server nhan alert tu TradingView
# =============================================================================

from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import aiohttp
from aiohttp import web
from loguru import logger

from core.utils.events import EventBus, TickReceivedEvent

if TYPE_CHECKING:
    pass


# =============================================================================
# Rate Limiter (Token Bucket)
# =============================================================================
class TokenBucketRateLimiter:
    """
    Rate limiter dung token bucket algorithm.

    Cho phep burst nhung gioi han toc do trung binh.
    """

    def __init__(
        self,
        rate: float,  # tokens per second
        capacity: int,
    ) -> None:
        self._rate = rate
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Thu hoc token. Tra ve True neu duoc phep, False neu bi reject."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update
            self._tokens = min(
                self._capacity,
                self._tokens + elapsed * self._rate,
            )
            self._last_update = now

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False


# =============================================================================
# TV Webhook Handler
# =============================================================================
@dataclass
class TVAlertPayload:
    """Du lieu tu TradingView webhook."""

    symbol: str
    timestamp_us: int
    action: str  # "buy" | "sell" | "close" | "ohlcv_update"
    price: float | None = None
    volume: float | None = None
    ohlcv: dict | None = None  # {open, high, low, close, volume}


class TVWebhookHandler:
    """
    HTTP handler nhan TradingView webhook.

    Tich hop voi EventBus de publish tick/alert events.

    Args:
        port: Port HTTP (default: 8080)
        hmac_secret: Secret key de xac thuc HMAC-SHA256
        rate_limit: So request toi da moi phut (default: 100)
        event_bus: EventBus instance
    """

    def __init__(
        self,
        port: int = 8080,
        hmac_secret: str | None = None,
        rate_limit_rpm: int = 100,
        event_bus: EventBus | None = None,
    ) -> None:
        self.port = port
        self.hmac_secret = hmac_secret or ""
        self.event_bus = event_bus

        # Rate limiter: 100 requests / 60 seconds = 1.67 rps
        self._rate_limiter = TokenBucketRateLimiter(
            rate=rate_limit_rpm / 60.0,
            capacity=rate_limit_rpm,
        )

        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._request_count: int = 0
        self._rejected_count: int = 0

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    async def start(self) -> None:
        """Khoi dong HTTP server."""
        self._app = web.Application()
        self._app.router.add_post("/webhook/tv", self._handle_webhook)
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/metrics", self._handle_metrics)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        self._site = self._runner.make_site(host="0.0.0.0", port=self.port)
        await self._site.start()

        logger.info(
            "TVWebhook server bat dau tren port {port}",
            port=self.port,
        )

    async def stop(self) -> None:
        """Dung HTTP server."""
        if self._runner:
            await self._runner.cleanup()
        logger.info("TVWebhook server da dung")

    # -------------------------------------------------------------------------
    # Handlers
    # -------------------------------------------------------------------------
    async def _handle_webhook(self, request: web.Request) -> web.Response:
        """Xu ly TradingView webhook POST."""
        self._request_count += 1

        # Rate limiting
        if not await self._rate_limiter.acquire():
            self._rejected_count += 1
            logger.warning("Webhook rejected: rate limit exceeded")
            return web.Response(
                status=429,
                text="Rate limit exceeded (100 requests/minute)",
            )

        # HMAC validation
        if self.hmac_secret:
            signature = request.headers.get("X-TV-Signature", "")
            body = await request.read()

            if not self._validate_signature(body, signature):
                logger.warning("Webhook rejected: invalid HMAC signature")
                return web.Response(status=401, text="Invalid signature")
        else:
            body = await request.read()

        # Parse JSON
        try:
            import json
            data = json.loads(body)
        except Exception:
            logger.warning("Webhook rejected: malformed JSON")
            return web.Response(status=400, text="Malformed JSON")

        # Validate payload
        try:
            payload = self._parse_payload(data)
        except Exception:
            logger.warning("Webhook rejected: invalid payload schema")
            return web.Response(status=400, text="Invalid payload schema")

        # Convert to TickReceivedEvent
        event = TickReceivedEvent(
            symbol=payload.symbol,
            timestamp_us=payload.timestamp_us,
            bid=payload.price or 0.0,
            ask=payload.price or 0.0,
            last=payload.price or 0.0,
            volume=payload.volume or 0.0,
            flags=0,
            aggressor=self._action_to_aggressor(payload.action),
            spread_pips=0.0,
            mid_price=payload.price or 0.0,
        )

        if self.event_bus:
            self.event_bus.publish(event)

        logger.debug(
            "TV Webhook: {symbol} {action} @ {price}",
            symbol=payload.symbol,
            action=payload.action,
            price=payload.price,
        )

        return web.Response(status=200, text="OK")

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            "status": "healthy",
            "port": self.port,
        })

    async def _handle_metrics(self, request: web.Request) -> web.Response:
        """Metrics endpoint."""
        return web.json_response({
            "requests_total": self._request_count,
            "requests_rejected": self._rejected_count,
        })

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _validate_signature(self, body: bytes, signature: str) -> bool:
        """Validate HMAC-SHA256 signature."""
        if not signature:
            return False

        expected = hmac.new(
            self.hmac_secret.encode(),
            body,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(expected, signature)

    def _parse_payload(self, data: dict) -> TVAlertPayload:
        """Parse TradingView payload."""
        # Support nhieu format
        symbol = data.get("symbol", data.get("ticker", "UNKNOWN"))

        # Timestamp
        ts = data.get("timestamp_us")
        if ts is None:
            ts = int(time.time() * 1_000_000)

        action = data.get("action", data.get("strategy", "ohlcv_update")).lower()

        price = None
        volume = None
        if action in ("buy", "sell"):
            price = float(data.get("price", 0.0))
            volume = float(data.get("volume", data.get("qty", 0.0)))
        elif action == "ohlcv_update":
            ohlcv = data.get("ohlcv", {})
            price = float(ohlcv.get("close", 0.0))
            volume = float(ohlcv.get("volume", 0.0))

        return TVAlertPayload(
            symbol=symbol,
            timestamp_us=ts,
            action=action,
            price=price,
            volume=volume,
        )

    def _action_to_aggressor(self, action: str) -> str:
        """Chuyen TradingView action thanh aggressor side."""
        mapping = {
            "buy": "BUY",
            "sell": "SELL",
            "close": "UNKNOWN",
            "ohlcv_update": "UNKNOWN",
        }
        return mapping.get(action.lower(), "UNKNOWN")

    @property
    def request_count(self) -> int:
        return self._request_count

    @property
    def rejected_count(self) -> int:
        return self._rejected_count
