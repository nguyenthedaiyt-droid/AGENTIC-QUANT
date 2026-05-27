# =============================================================================
# AGENTIC-QUANT — IPC Message Schemas (Backend -> Frontend)
# Dinh nghia 8 loai message Pydantic gui qua WebSocket
# Tat ca message deu co type: str va emit_time_ms: int
# =============================================================================

from __future__ import annotations

import time
from typing import Any, Literal

import msgpack
from pydantic import BaseModel, Field


# =============================================================================
# Base Message
# =============================================================================

class BaseIPCMessage(BaseModel):
    """Base cho tat ca IPC messages Backend->Frontend.

    Tat ca message deu co:
      - type: str — dinh danh loai message
      - emit_time_ms: int — unix timestamp (ms) dat luc serialize
    """

    type: str = Field(..., description="Loai message (dinh danh)")
    emit_time_ms: int = Field(
        default_factory=lambda: int(time.time() * 1000),
        description="Thoi gian phat (ms since epoch)",
    )

    def model_dump_json(self, **kwargs: Any) -> str:
        """Serialize JSON, tu dong set emit_time_ms truoc khi dump."""
        self.emit_time_ms = int(time.time() * 1000)
        return super().model_dump_json(**kwargs)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize dict, tu dong set emit_time_ms truoc khi dump."""
        self.emit_time_ms = int(time.time() * 1000)
        return super().model_dump(**kwargs)


# =============================================================================
# 1. BarUpdateMessage
# =============================================================================

class BarUpdateMessage(BaseIPCMessage):
    """Cap nhat bar dang hinh thanh (live)."""

    type: Literal["bar_update"] = "bar_update"
    symbol: str = ""
    timeframe: str = ""
    bar_open: float = 0.0
    bar_high: float = 0.0
    bar_low: float = 0.0
    bar_close: float = 0.0
    bar_volume: float = 0.0
    bucket_time: int = 0
    is_closed: bool = False

    def serialize_msgpack(self) -> bytes:
        """Serialize sang MessagePack binary (optional, cho bar_update)."""
        self.emit_time_ms = int(time.time() * 1000)
        data = self.model_dump()
        return msgpack.packb(data, use_bin_type=True)


# =============================================================================
# 2. ZoneUpdateMessage
# =============================================================================

class ZoneUpdateMessage(BaseIPCMessage):
    """Trang thai zone thay doi (mitigated, claimed, ...)."""

    type: Literal["zone_update"] = "zone_update"
    zone_id: str = ""
    zone_type: str = ""
    symbol: str = ""
    timeframe: str = ""
    old_status: str = ""
    new_status: str = ""
    price_top: float = 0.0
    price_bottom: float = 0.0
    premium_discount: str = ""
    p_hold: float = 0.0


# =============================================================================
# 3. PredictionUpdateMessage
# =============================================================================

class PredictionUpdateMessage(BaseIPCMessage):
    """Ket qua du doan tu Model A/B."""

    type: Literal["prediction_update"] = "prediction_update"
    symbol: str = ""
    timeframe: str = ""
    timestamp: int = 0

    # Model A
    p_bsl: float = 0.0
    p_ssl: float = 0.0
    p_lateral: float = 0.0
    bsl_target: float = 0.0
    ssl_target: float = 0.0

    # Model B
    zones_predicted: list[dict] = Field(default_factory=list)

    # Context
    session_id: str = ""
    active_guardrail: bool = False
    macro_regime: str = ""
    confidence_qualifier: str = ""


# =============================================================================
# 4. CountdownUpdateMessage
# =============================================================================

class CountdownUpdateMessage(BaseIPCMessage):
    """Cap nhat countdown cho guardrail / news event.

    Khi guardrail active: throttle 1s
    Khi khong guardrail: throttle 10s
    """

    type: Literal["countdown_update"] = "countdown_update"
    guardrail_type: str = ""  # "PRE_NEWS" | "NEWS_WINDOW" | "SPIKE" | ""
    seconds_remaining: int = 0
    is_active: bool = False
    event_id: str | None = None
    dampening_factor: float = 1.0


# =============================================================================
# 5. ConsensusReadyMessage
# =============================================================================

class ConsensusReadyMessage(BaseIPCMessage):
    """Ket qua debate tu Multi-Agent (Bull/Bear/Critic)."""

    type: Literal["consensus_ready"] = "consensus_ready"
    symbol: str = ""
    timestamp: int = 0
    rating: int = 0  # -4 to +4
    direction: str = ""  # "BULL" | "BEAR" | "NEUTRAL"
    confidence_qualifier: str = ""
    agreement_score: float = 0.0
    bull_thesis: str = ""
    bear_thesis: str = ""
    reasoning: str = ""
    bull_evidence: list[str] = Field(default_factory=list)
    bear_evidence: list[str] = Field(default_factory=list)
    debate_used_fallback: bool = False


# =============================================================================
# 6. SystemStatusMessage
# =============================================================================

class SystemStatusMessage(BaseIPCMessage):
    """Cap nhat trang thai he thong (health, feed, model, ...)."""

    type: Literal["system_status"] = "system_status"
    component: str = ""  # "feed" | "model" | "redis" | "calendar" | "system"
    status: str = ""  # "healthy" | "degraded" | "failure" | "reconnected" | "ready"
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# 7. FullStateSnapshotMessage
# =============================================================================

class FullStateSnapshotMessage(BaseIPCMessage):
    """Snapshot toan bo trang thai hien tai (gui khi frontend request)."""

    type: Literal["full_state_snapshot"] = "full_state_snapshot"
    symbol: str = ""
    bars: list[dict[str, Any]] = Field(default_factory=list)
    zones: list[dict[str, Any]] = Field(default_factory=list)
    predictions: dict[str, Any] = Field(default_factory=dict)
    consensus: dict[str, Any] = Field(default_factory=dict)
    system: dict[str, Any] = Field(default_factory=dict)
    guardrail_active: bool = False
    seconds_remaining: int = 0
    active_news_events: list[dict[str, Any]] = Field(default_factory=list)
    timestamp: int = 0


# =============================================================================
# 8. NewsReleaseMessage
# =============================================================================

class NewsReleaseMessage(BaseIPCMessage):
    """Su kien kinh te sap/giong release."""

    type: Literal["news_release"] = "news_release"
    event_id: str = ""
    title: str = ""
    currency: str = ""
    impact: str = "MEDIUM"  # "LOW" | "MEDIUM" | "HIGH"
    scheduled_time: int = 0  # unix timestamp
    forecast: float | None = None
    previous: float | None = None
    actual: float | None = None
    seconds_to_event: int = 0
    state: str = "SCHEDULED"  # SCHEDULED | PRE_NEWS | NEWS_WINDOW | POST_NEWS | COMPLETED
    i_news: float = 0.0
    surprise_z: float | None = None
    surprise_direction: str = "NEUTRAL"


# =============================================================================
# Union type cho message routing
# =============================================================================

AllIPCMessages = (
    BarUpdateMessage
    | ZoneUpdateMessage
    | PredictionUpdateMessage
    | CountdownUpdateMessage
    | ConsensusReadyMessage
    | SystemStatusMessage
    | FullStateSnapshotMessage
    | NewsReleaseMessage
)
