# =============================================================================
# AGENTIC-QUANT — Broadcast Dispatcher
# Subscribe vao EventBus, chuyen doi event thanh IPC message,
# throttle va broadcast toi WebSocket clients
# =============================================================================

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from core.ipc.message_schema import (
    BarUpdateMessage,
    ConsensusReadyMessage,
    CountdownUpdateMessage,
    FullStateSnapshotMessage,
    NewsReleaseMessage,
    PredictionUpdateMessage,
    SystemStatusMessage,
    ZoneUpdateMessage,
)
from core.utils.events import EventBus
from core.utils.events.types import (
    BarUpdateEvent,
    ConsensusReadyEvent,
    EventType,
    GuardrailActivatedEvent,
    GuardrailDeactivatedEvent,
    ModelDegradedEvent,
    NewsAlertEvent,
    PredictionReadyEvent,
    SystemReadyEvent,
    ZoneUpdateEvent,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from core.utils.events.types import AllEvents


# =============================================================================
# BroadcastDispatcher
# =============================================================================

class BroadcastDispatcher:
    """Nhan events tu EventBus, chuyen thanh IPC message va broadcast.

    Tich hop throttle logic:
      - bar_update: max 10 fps (0.1s min interval)
      - countdown_update: 1s khi guardrail active, 10s khi khong
    """

    def __init__(self, event_bus: EventBus) -> None:
        self._bus = event_bus
        self._broadcast_fn: Callable[[dict[str, Any]], None] | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self._msgpack_available: bool = False

        # Thu import msgpack, fallback ve JSON
        try:
            import msgpack  # noqa: F401
            self._msgpack_available = True
        except ImportError:
            self._msgpack_available = False
            logger.warning(
                "msgpack khong co san — fallback ve JSON serialization"
            )

        # --- Throttle state ---
        self._last_bar_update_ms: int = 0
        self._last_countdown_ms: int = 0
        self._bar_min_interval_ms: int = 100  # 10 fps = 100ms
        self._countdown_guardrail_interval_ms: int = 1000  # 1s
        self._countdown_normal_interval_ms: int = 10000  # 10s

        logger.info("BroadcastDispatcher khoi tao")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_broadcast_fn(
        self, fn: Callable[[dict[str, Any]], None]
    ) -> None:
        """Gan ham broadcast (duoc WebSocketServer goi)."""
        self._broadcast_fn = fn
        logger.debug("Broadcast function da duoc gan")

    def start(self) -> None:
        """Bat dau subscribe vao EventBus."""
        if self._broadcast_fn is None:
            logger.warning(
                "BroadcastDispatcher chua co broadcast_fn, "
                "subscriber se duoc kich hoat sau"
            )
            return

        handlers: dict[EventType, Any] = {
            # Data
            EventType.BAR_UPDATE: self._on_bar_update,
            # Zones
            EventType.ZONE_UPDATE: self._on_zone_update,
            # Predictions
            EventType.PREDICTION_READY: self._on_prediction_ready,
            # Countdown / Guardrail
            EventType.GUARDRAIL_ACTIVATED: self._on_guardrail_activated,
            EventType.GUARDRAIL_DEACTIVATED: self._on_guardrail_deactivated,
            # Consensus
            EventType.CONSENSUS_READY: self._on_consensus_ready,
            # System
            EventType.SYSTEM_READY: self._on_system_ready,
            EventType.MODEL_DEGRADED: self._on_model_degraded,
            EventType.FEED_FAILURE: self._on_feed_failure,
            EventType.FEED_RECONNECTED: self._on_feed_reconnected,
            # News
            EventType.NEWS_ALERT: self._on_news_alert,
        }

        self._unsubscribe = self._bus.subscribe_many(handlers)
        logger.info(
            f"BroadcastDispatcher subscribed vao "
            f"{len(handlers)} event types"
        )

    def stop(self) -> None:
        """Huy subscribe."""
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
            logger.info("BroadcastDispatcher da unsubscribe")

    # ------------------------------------------------------------------
    # Internal: throttle helpers
    # ------------------------------------------------------------------

    def _should_throttle(self, last_ms: int, min_interval_ms: int) -> bool:
        """Kiem tra co nen throttle message hay khong."""
        now_ms = int(time.time() * 1000)
        if now_ms - last_ms < min_interval_ms:
            return True
        return False

    def _update_throttle(self, attr_name: str) -> int:
        """Cap nhat last throttle timestamp, tra ve current ms."""
        now_ms = int(time.time() * 1000)
        setattr(self, attr_name, now_ms)
        return now_ms

    # ------------------------------------------------------------------
    # Event Handlers
    # ------------------------------------------------------------------

    async def _on_bar_update(self, event: AllEvents) -> None:
        assert isinstance(event, BarUpdateEvent)
        # Throttle bar_update: max 10 fps
        if self._should_throttle(
            self._last_bar_update_ms, self._bar_min_interval_ms
        ):
            return
        self._update_throttle("_last_bar_update_ms")

        msg = BarUpdateMessage(
            symbol=event.symbol,
            timeframe=event.timeframe,
            bar_open=event.bar_open,
            bar_high=event.bar_high,
            bar_low=event.bar_low,
            bar_close=event.bar_close,
            bar_volume=event.bar_volume,
            bucket_time=event.bucket_time,
            is_closed=event.is_closed,
        )
        self._broadcast_fn(msg.model_dump())

    async def _on_zone_update(self, event: AllEvents) -> None:
        assert isinstance(event, ZoneUpdateEvent)
        msg = ZoneUpdateMessage(
            zone_id=event.zone_id,
            old_status=event.old_status,
            new_status=event.new_status,
            p_hold=event.new_p_hold if event.new_p_hold is not None else 0.0,
        )
        self._broadcast_fn(msg.model_dump())

    async def _on_prediction_ready(self, event: AllEvents) -> None:
        assert isinstance(event, PredictionReadyEvent)
        msg = PredictionUpdateMessage(
            symbol=event.symbol,
            timeframe=event.timeframe,
            timestamp=event.timestamp,
            p_bsl=event.p_bsl,
            p_ssl=event.p_ssl,
            p_lateral=event.p_lateral,
            bsl_target=event.bsl_target,
            ssl_target=event.ssl_target,
            zones_predicted=event.zones_predicted,
            session_id=event.session_id,
            active_guardrail=event.active_guardrail,
            macro_regime=event.macro_regime,
            confidence_qualifier=event.confidence_qualifier,
        )
        self._broadcast_fn(msg.model_dump())

    async def _on_guardrail_activated(self, event: AllEvents) -> None:
        assert isinstance(event, GuardrailActivatedEvent)
        # Throttle: 1s khi guardrail active
        if self._should_throttle(
            self._last_countdown_ms,
            self._countdown_guardrail_interval_ms,
        ):
            return
        self._update_throttle("_last_countdown_ms")

        msg = CountdownUpdateMessage(
            guardrail_type=event.guardrail_type,
            seconds_remaining=event.seconds_remaining,
            is_active=True,
            event_id=event.event_id,
            dampening_factor=event.dampening_factor,
        )
        self._broadcast_fn(msg.model_dump())

    async def _on_guardrail_deactivated(self, event: AllEvents) -> None:
        assert isinstance(event, GuardrailDeactivatedEvent)
        # Throttle: 10s khi khong guardrail (gui ngay khi deactivate)
        # Cho deactivate => broadcast ngay de frontend biet
        msg = CountdownUpdateMessage(
            guardrail_type=event.guardrail_type,
            seconds_remaining=0,
            is_active=False,
        )
        self._broadcast_fn(msg.model_dump())

    async def _on_consensus_ready(self, event: AllEvents) -> None:
        assert isinstance(event, ConsensusReadyEvent)
        msg = ConsensusReadyMessage(
            symbol=event.symbol,
            timestamp=event.timestamp,
            rating=event.rating,
            direction=event.direction,
            confidence_qualifier=event.confidence_qualifier,
            agreement_score=event.agreement_score,
            bull_thesis=event.bull_thesis,
            bear_thesis=event.bear_thesis,
            reasoning=event.reasoning,
            bull_evidence=event.bull_evidence,
            bear_evidence=event.bear_evidence,
            debate_used_fallback=event.debate_used_fallback,
        )
        self._broadcast_fn(msg.model_dump())

    async def _on_system_ready(self, event: AllEvents) -> None:
        assert isinstance(event, SystemReadyEvent)
        msg = SystemStatusMessage(
            component="system",
            status="ready",
            message="He thong da san sang",
            details={
                "startup_duration_ms": event.startup_duration_ms,
                "components_ready": event.components_ready,
            },
        )
        self._broadcast_fn(msg.model_dump())

    async def _on_model_degraded(self, event: AllEvents) -> None:
        assert isinstance(event, ModelDegradedEvent)
        msg = SystemStatusMessage(
            component="model",
            status="degraded",
            message=event.trigger_reason,
            details={
                "ic_value": event.ic_value,
                "brier_value": event.brier_value,
                "f1_value": event.f1_value,
                "drifted_features": event.drifted_features,
            },
        )
        self._broadcast_fn(msg.model_dump())

    async def _on_feed_failure(self, event: AllEvents) -> None:
        # FeedFailureEvent
        msg = SystemStatusMessage(
            component="feed",
            status="failure",
            message=getattr(event, "error_message", ""),
            details={
                "feed_name": getattr(event, "feed_name", ""),
                "reconnect_attempt": getattr(event, "reconnect_attempt", 0),
            },
        )
        self._broadcast_fn(msg.model_dump())

    async def _on_feed_reconnected(self, event: AllEvents) -> None:
        # FeedReconnectedEvent
        msg = SystemStatusMessage(
            component="feed",
            status="reconnected",
            message="Feed da ket noi lai",
            details={
                "feed_name": getattr(event, "feed_name", ""),
                "attempts": getattr(event, "attempts", 0),
            },
        )
        self._broadcast_fn(msg.model_dump())

    async def _on_news_alert(self, event: AllEvents) -> None:
        assert isinstance(event, NewsAlertEvent)
        msg = NewsReleaseMessage(
            event_id=event.event_id,
            title=event.title,
            currency=event.currency,
            impact=event.impact.value
            if hasattr(event.impact, "value")
            else str(event.impact),
            scheduled_time=int(event.scheduled_time.timestamp()),
            forecast=event.forecast,
            previous=event.previous,
            actual=event.actual,
            seconds_to_event=event.seconds_to_event,
            state=event.state,
            i_news=event.i_news,
            surprise_z=event.surprise_z,
            surprise_direction=event.surprise_direction,
        )
        self._broadcast_fn(msg.model_dump())

    # ------------------------------------------------------------------
    # Serialization helpers (MessagePack fallback)
    # ------------------------------------------------------------------

    def serialize_message(self, message: dict[str, Any]) -> tuple[bytes, bool]:
        """Serialize dict message, thu MessagePack truoc, fallback JSON.

        Args:
            message: Dict can serialize

        Returns:
            Tuple (serialized_bytes, is_msgpack)
        """
        if self._msgpack_available:
            try:
                import msgpack

                packed: bytes = msgpack.packb(message, use_bin_type=True)  # type: ignore[assignment]
                return packed, True
            except Exception:
                logger.debug(
                    "MessagePack serialization failed, fallback to JSON"
                )

        # JSON fallback
        raw_str = json.dumps(message, default=str)
        return raw_str.encode("utf-8"), False

    # ------------------------------------------------------------------
    # Broadcast (goi tu WebSocketServer)
    # ------------------------------------------------------------------

    def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast truc tiep mot dict message (goi tu ben ngoai).

        Tu dong dung MessagePack neu co san, fallback ve JSON.
        Phat hien client disconnect: broadcast_fn xu ly cleanup
        (WebSocketServer _broadcast_to_all discard client khi loi).
        """
        if self._broadcast_fn is not None:
            self._broadcast_fn(message)
        else:
            logger.warning("BroadcastDispatcher: chua co broadcast_fn")
