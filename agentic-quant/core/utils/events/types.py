# =============================================================================
# AGENTIC-QUANT — Din nghia tat ca cac loai Event
# Event Bus la noi trung tam ket noi cac module voi nhau
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Coroutine

if TYPE_CHECKING:
    pass


# =============================================================================
# Enum: Loai Event
# =============================================================================
class EventType(str, Enum):
    """Tat ca cac loai event trong he thong."""

    # --- Data Ingestion ---
    TICK_RECEIVED = "TICK_RECEIVED"
    BAR_UPDATE = "BAR_UPDATE"
    BAR_CLOSE = "BAR_CLOSE"

    # --- Volumetrics ---
    VOLUMETRICS_UPDATE = "VOLUMETRICS_UPDATE"

    # --- Macro / News ---
    NEWS_ALERT = "NEWS_ALERT"
    CALENDAR_FETCHED = "CALENDAR_FETCHED"
    REGIME_CHANGE = "REGIME_CHANGE"

    # --- Guardrails ---
    GUARDRAIL_ACTIVATED = "GUARDRAIL_ACTIVATED"
    GUARDRAIL_DEACTIVATED = "GUARDRAIL_DEACTIVATED"
    SPIKE_REGIME = "SPIKE_REGIME"

    # --- AI / Predictions ---
    PREDICTION_READY = "PREDICTION_READY"
    PREDICTION_BATCH = "PREDICTION_BATCH"
    CONSENSUS_READY = "CONSENSUS_READY"

    # --- Zones ---
    ZONE_CREATED = "ZONE_CREATED"
    ZONE_UPDATE = "ZONE_UPDATE"
    ZONE_CLAIMED = "ZONE_CLAIMED"

    # --- Outcomes ---
    OUTCOME_CONFIRMED = "OUTCOME_CONFIRMED"
    OUTCOME_TIMEOUT = "OUTCOME_TIMEOUT"

    # --- System Health ---
    MODEL_DEGRADED = "MODEL_DEGRADED"
    FEED_FAILURE = "FEED_FAILURE"
    FEED_RECONNECTED = "FEED_RECONNECTED"
    STALENESS_ALERT = "STALENESS_ALERT"
    CALENDAR_STALE = "CALENDAR_STALE"
    SYSTEM_READY = "SYSTEM_READY"

    # --- Backtest ---
    BACKTEST_TICK = "BACKTEST_TICK"
    BACKTEST_BAR = "BACKTEST_BAR"


# =============================================================================
# Base Event
# =============================================================================
@dataclass
class BaseEvent:
    """Base class cho tat ca events."""

    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "system"

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, EventType):
            self.event_type = EventType(self.event_type)


# =============================================================================
# Data Ingestion Events
# =============================================================================
@dataclass
class TickReceivedEvent(BaseEvent):
    """Tick moi tu ZeroMQ MT5."""
    event_type: EventType = field(default=EventType.TICK_RECEIVED, init=False)

    symbol: str = ""
    timestamp_us: int = 0  # microseconds since epoch
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: float = 0.0
    flags: int = 0
    is_abnormal_spread: bool = False
    aggressor: str = ""  # "BUY" | "SELL" | "UNKNOWN"

    spread_pips: float = 0.0
    mid_price: float = 0.0


@dataclass
class BarUpdateEvent(BaseEvent):
    """Cap nhat bar dang hinh thanh (real-time update)."""
    event_type: EventType = field(default=EventType.BAR_UPDATE, init=False)

    symbol: str = ""
    timeframe: str = ""  # M1, M5, M15, H1, H4, D1
    bar_open: float = 0.0
    bar_high: float = 0.0
    bar_low: float = 0.0
    bar_close: float = 0.0
    bar_volume: float = 0.0
    bucket_time: int = 0  # unix timestamp (second)
    is_closed: bool = False


@dataclass
class BarCloseEvent(BaseEvent):
    """Bar da dong — day la trigger chinh cho AI Engine."""
    event_type: EventType = field(default=EventType.BAR_CLOSE, init=False)

    symbol: str = ""
    timeframe: str = ""
    bar_open: float = 0.0
    bar_high: float = 0.0
    bar_low: float = 0.0
    bar_close: float = 0.0
    bar_volume: float = 0.0
    bucket_time: int = 0
    tick_count: int = 0  # so tick trong bar nay


# =============================================================================
# Volumetrics Events
# =============================================================================
@dataclass
class VolumetricsUpdateEvent(BaseEvent):
    """Cap nhat volumetrics (CVD, OBI, III) sau moi tick."""
    event_type: EventType = field(default=EventType.VOLUMETRICS_UPDATE, init=False)

    symbol: str = ""
    timeframe: str = ""

    # CVD (Cumulative Volume Delta)
    cvd: float = 0.0
    cvd_norm: float = 0.0  # CVD / V_total

    # CVD rolling windows
    cvd_ma5: float = 0.0
    cvd_ma10: float = 0.0
    cvd_ma20: float = 0.0
    cvd_ma50: float = 0.0

    # Order Book Imbalance
    obi: float = 0.0  # [-1, 1]

    # Institutional Intensity Index
    iii: float = 0.0

    # Divergence Score
    divergence_score: float = 0.0  # -1=sell, 0=neutral, 1=buy divergence

    bucket_time: int = 0


# =============================================================================
# Macro / News Events
# =============================================================================
class NewsImpact(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class NewsAlertEvent(BaseEvent):
    """Su kien kinh te sap dien ra."""
    event_type: EventType = field(default=EventType.NEWS_ALERT, init=False)

    event_id: str = ""
    title: str = ""
    currency: str = ""  # USD, EUR, GBP...
    impact: NewsImpact = NewsImpact.MEDIUM
    scheduled_time: datetime = field(default_factory=datetime.utcnow)
    forecast: float | None = None
    previous: float | None = None
    actual: float | None = None
    seconds_to_event: int = 0
    state: str = "SCHEDULED"  # SCHEDULED | PRE_NEWS | NEWS_WINDOW | POST_NEWS | COMPLETED
    i_news: float = 0.0
    surprise_z: float | None = None
    surprise_direction: str = "NEUTRAL"


@dataclass
class CalendarFetchedEvent(BaseEvent):
    """Lich kinh te da duoc fetch tu ForexFactory."""
    event_type: EventType = field(default=EventType.CALENDAR_FETCHED, init=False)

    events_count: int = 0
    fetched_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RegimeChangeEvent(BaseEvent):
    """Thi truong chuyen regime."""
    event_type: EventType = field(default=EventType.REGIME_CHANGE, init=False)

    previous_regime: str = ""
    new_regime: str = ""
    trigger_reason: str = ""


# =============================================================================
# Guardrail Events
# =============================================================================
@dataclass
class GuardrailActivatedEvent(BaseEvent):
    """Guardrail duoc kich hoat (pre-news, news window)."""
    event_type: EventType = field(default=EventType.GUARDRAIL_ACTIVATED, init=False)

    guardrail_type: str = ""  # "PRE_NEWS" | "NEWS_WINDOW" | "SPIKE"
    event_id: str | None = None
    dampening_factor: float = 1.0
    seconds_remaining: int = 0


@dataclass
class GuardrailDeactivatedEvent(BaseEvent):
    """Guardrail bi tat."""
    event_type: EventType = field(default=EventType.GUARDRAIL_DEACTIVATED, init=False)

    guardrail_type: str = ""


# =============================================================================
# Zone Events
# =============================================================================
@dataclass
class ZoneCreatedEvent(BaseEvent):
    """Zone moi duoc tao (FVG, OB, Swing Point)."""
    event_type: EventType = field(default=EventType.ZONE_CREATED, init=False)

    zone_id: str = ""
    zone_type: str = ""  # "FVG_BULL" | "FVG_BEAR" | "OB_BULL" | "OB_BEAR" | "BSL" | "SSL"
    symbol: str = ""
    timeframe: str = ""
    price_top: float = 0.0
    price_bottom: float = 0.0
    formed_time: int = 0
    premium_discount: str = ""  # "PREMIUM" | "DISCOUNT" | "MID"
    p_hold: float = 0.0
    w_zone: float = 0.0


@dataclass
class ZoneUpdateEvent(BaseEvent):
    """Zone bi thay doi trang thai (mitigated, claimed)."""
    event_type: EventType = field(default=EventType.ZONE_UPDATE, init=False)

    zone_id: str = ""
    old_status: str = ""
    new_status: str = ""  # "UNMITIGATED" | "PARTIALLY_MITIGATED" | "MITIGATED" | "CLAIMED"
    new_p_hold: float | None = None


# =============================================================================
# Prediction Events
# =============================================================================
@dataclass
class PredictionReadyEvent(BaseEvent):
    """Ket qua du doan tu XGBoost Model A/B."""
    event_type: EventType = field(default=EventType.PREDICTION_READY, init=False)

    symbol: str = ""
    timeframe: str = ""
    timestamp: int = 0

    # Model A outputs
    p_bsl: float = 0.0
    p_ssl: float = 0.0
    p_lateral: float = 0.0
    bsl_target: float = 0.0
    ssl_target: float = 0.0

    # Model B outputs
    zones_predicted: list[dict] = field(default_factory=list)

    # Session / macro context
    session_id: str = ""
    active_guardrail: bool = False
    macro_regime: str = ""

    confidence_qualifier: str = ""  # "HIGH" | "MEDIUM" | "LOW"


@dataclass
class ConsensusReadyEvent(BaseEvent):
    """Ket qua debate tu Multi-Agent (Bull/Bear/Critic)."""
    event_type: EventType = field(default=EventType.CONSENSUS_READY, init=False)

    symbol: str = ""
    timestamp: int = 0

    rating: int = 0  # -4 to +4
    direction: str = ""  # "BULL" | "BEAR" | "NEUTRAL"
    confidence_qualifier: str = ""
    agreement_score: float = 0.0

    bull_thesis: str = ""
    bear_thesis: str = ""
    reasoning: str = ""

    bull_evidence: list[str] = field(default_factory=list)
    bear_evidence: list[str] = field(default_factory=list)

    debate_used_fallback: bool = False


# =============================================================================
# Outcome Events
# =============================================================================
@dataclass
class OutcomeConfirmedEvent(BaseEvent):
    """Ket qua du doan da duoc xac nhan (BSL hit, SSL hit, timeout)."""
    event_type: EventType = field(default=EventType.OUTCOME_CONFIRMED, init=False)

    prediction_id: str = ""
    symbol: str = ""
    outcome: str = ""  # "BSL_HIT" | "SSL_HIT" | "LATERAL" | "TIMEOUT" | "ZONE_HOLD"
    outcome_timestamp: datetime = field(default_factory=datetime.utcnow)

    actual_bsl_hit: bool = False
    actual_ssl_hit: bool = False
    actual_zone_hold: bool = False

    elapsed_minutes: float = 0.0
    max_horizon_minutes: float = 240.0  # 4 hours


# =============================================================================
# System Health Events
# =============================================================================
@dataclass
class ModelDegradedEvent(BaseEvent):
    """Model hieu suat giam."""
    event_type: EventType = field(default=EventType.MODEL_DEGRADED, init=False)

    trigger_reason: str = ""
    ic_value: float | None = None
    brier_value: float | None = None
    f1_value: float | None = None
    drifted_features: list[str] = field(default_factory=list)


@dataclass
class FeedFailureEvent(BaseEvent):
    """Mat ket noi nguon du lieu."""
    event_type: EventType = field(default=EventType.FEED_FAILURE, init=False)

    feed_name: str = ""  # "MT5_ZMQ" | "TV_WEBHOOK" | "CALENDAR"
    error_message: str = ""
    reconnect_attempt: int = 0


@dataclass
class FeedReconnectedEvent(BaseEvent):
    """Nguon du lieu da ket noi lai."""
    event_type: EventType = field(default=EventType.FEED_RECONNECTED, init=False)

    feed_name: str = ""
    attempts: int = 0


@dataclass
class StalenessAlertEvent(BaseEvent):
    """Du lieu cu or隔时."""
    event_type: EventType = field(default=EventType.STALENESS_ALERT, init=False)

    feed_name: str = ""
    last_update_age_seconds: int = 0
    threshold_seconds: int = 0


@dataclass
class SystemReadyEvent(BaseEvent):
    """He thong da san sang (sau cold start)."""
    event_type: EventType = field(default=EventType.SYSTEM_READY, init=False)

    startup_duration_ms: int = 0
    components_ready: dict[str, bool] = field(default_factory=dict)


# =============================================================================
# Union type cho tat ca events
# =============================================================================
AllEvents = (
    TickReceivedEvent
    | BarUpdateEvent
    | BarCloseEvent
    | VolumetricsUpdateEvent
    | NewsAlertEvent
    | CalendarFetchedEvent
    | RegimeChangeEvent
    | GuardrailActivatedEvent
    | GuardrailDeactivatedEvent
    | ZoneCreatedEvent
    | ZoneUpdateEvent
    | PredictionReadyEvent
    | ConsensusReadyEvent
    | OutcomeConfirmedEvent
    | ModelDegradedEvent
    | FeedFailureEvent
    | FeedReconnectedEvent
    | StalenessAlertEvent
    | SystemReadyEvent
    | BaseEvent
)

# Handler type alias
EventHandler = Callable[[AllEvents], Coroutine[Any, Any, None]]
