# =============================================================================
# AGENTIC-QUANT — Events Module
# =============================================================================
"""
Event Bus va dinh nghia cac loai Event.

Su dung::

    from core.utils.events import EventBus, get_event_bus
    from core.utils.events.types import TickReceivedEvent, EventType

    bus = EventBus()

    async def on_tick(event: TickReceivedEvent):
        print(f"Tick nhan: {event.last}")

    bus.subscribe(EventType.TICK_RECEIVED, on_tick)
    bus.publish(TickReceivedEvent(symbol="XAUUSD", last=2500.0))
"""

from .bus import EventBus, get_event_bus, get_event_bus_sync
from .types import (
    AllEvents,
    BarCloseEvent,
    BarUpdateEvent,
    CalendarFetchedEvent,
    ConsensusReadyEvent,
    EventHandler,
    EventType,
    FeedFailureEvent,
    FeedReconnectedEvent,
    GuardrailActivatedEvent,
    GuardrailDeactivatedEvent,
    ModelDegradedEvent,
    NewsAlertEvent,
    NewsImpact,
    OutcomeConfirmedEvent,
    PredictionReadyEvent,
    RegimeChangeEvent,
    StalenessAlertEvent,
    SystemReadyEvent,
    TickReceivedEvent,
    VolumetricsUpdateEvent,
    ZoneCreatedEvent,
    ZoneUpdateEvent,
)

__all__ = [
    "EventBus",
    "get_event_bus",
    "get_event_bus_sync",
    "EventType",
    "AllEvents",
    "EventHandler",
    # Events
    "TickReceivedEvent",
    "BarUpdateEvent",
    "BarCloseEvent",
    "VolumetricsUpdateEvent",
    "NewsAlertEvent",
    "NewsImpact",
    "CalendarFetchedEvent",
    "RegimeChangeEvent",
    "GuardrailActivatedEvent",
    "GuardrailDeactivatedEvent",
    "ZoneCreatedEvent",
    "ZoneUpdateEvent",
    "PredictionReadyEvent",
    "ConsensusReadyEvent",
    "OutcomeConfirmedEvent",
    "ModelDegradedEvent",
    "FeedFailureEvent",
    "FeedReconnectedEvent",
    "StalenessAlertEvent",
    "SystemReadyEvent",
]
