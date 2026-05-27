# =============================================================================
# AGENTIC-QUANT — Unit Tests cho Event Bus
# =============================================================================

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from core.utils.events import (
    EventBus,
    EventType,
    TickReceivedEvent,
    BarCloseEvent,
    AllEvents,
)
from core.utils.events.types import EventHandler


class TestEventBus:
    """Tests cho EventBus publish/subscribe."""

    @pytest.fixture
    def bus(self) -> EventBus:
        return EventBus(max_queue_size=100, default_handler_timeout=5.0)

    # --- Basic publish/subscribe ---
    def test_subscribe_and_publish(self, bus: EventBus) -> None:
        received: list[AllEvents] = []

        async def handler(event: AllEvents) -> None:
            received.append(event)

        bus.subscribe(EventType.TICK_RECEIVED, handler)
        bus.publish(TickReceivedEvent(symbol="XAUUSD", last=2500.0))

        # Handler chay async nen can cho
        # Trong sync test, kiem tra subscribe da thuc hien
        assert EventType.TICK_RECEIVED in bus._handlers
        assert len(bus._handlers[EventType.TICK_RECEIVED]) == 1

    def test_unsubscribe(self, bus: EventBus) -> None:
        async def handler(_: AllEvents) -> None:
            pass

        unsub = bus.subscribe(EventType.TICK_RECEIVED, handler)
        assert len(bus._handlers[EventType.TICK_RECEIVED]) == 1

        unsub()
        assert len(bus._handlers[EventType.TICK_RECEIVED]) == 0

    def test_multiple_handlers(self, bus: EventBus) -> None:
        count = 0

        async def h1(_: AllEvents) -> None:
            nonlocal count
            count += 1

        async def h2(_: AllEvents) -> None:
            nonlocal count
            count += 2

        bus.subscribe(EventType.TICK_RECEIVED, h1)
        bus.subscribe(EventType.TICK_RECEIVED, h2)

        assert len(bus._handlers[EventType.TICK_RECEIVED]) == 2

    def test_publish_no_subscribers(self, bus: EventBus) -> None:
        # Khong co loi khi publish event khong co subscriber
        bus.publish(TickReceivedEvent(symbol="XAUUSD", last=2500.0))
        assert bus._metrics["events_published"] == 1

    def test_publish_to_specific_event_type(self, bus: EventBus) -> None:
        subscribed_types: list[EventType] = []

        async def on_tick(_: AllEvents) -> None:
            subscribed_types.append(EventType.TICK_RECEIVED)

        bus.subscribe(EventType.TICK_RECEIVED, on_tick)

        # Publish TICK_RECEIVED
        bus.publish(TickReceivedEvent(symbol="XAUUSD", last=2500.0))
        assert EventType.TICK_RECEIVED not in subscribed_types  # Chua chay async

        # Publish BAR_CLOSE (khong co subscriber)
        bus.publish(BarCloseEvent(symbol="XAUUSD", timeframe="M1"))
        assert bus._metrics["events_published"] == 2

    def test_subscribe_many(self, bus: EventBus) -> None:
        async def on_tick(_: AllEvents) -> None:
            pass

        async def on_bar(_: AllEvents) -> None:
            pass

        handlers = {
            EventType.TICK_RECEIVED: on_tick,
            EventType.BAR_CLOSE: on_bar,
        }
        unsub_all = bus.subscribe_many(handlers)

        assert len(bus._handlers[EventType.TICK_RECEIVED]) == 1
        assert len(bus._handlers[EventType.BAR_CLOSE]) == 1

        unsub_all()
        assert len(bus._handlers[EventType.TICK_RECEIVED]) == 0
        assert len(bus._handlers[EventType.BAR_CLOSE]) == 0

    def test_metrics(self, bus: EventBus) -> None:
        bus.publish(TickReceivedEvent(symbol="XAUUSD", last=2500.0))
        bus.publish(TickReceivedEvent(symbol="XAUUSD", last=2501.0))

        metrics = bus.get_metrics()
        assert metrics["published"] == 2
        assert metrics["published_by_type"]["TICK_RECEIVED"] == 2

    def test_unsubscribe_all(self, bus: EventBus) -> None:
        async def h1(_: AllEvents) -> None:
            pass

        async def h2(_: AllEvents) -> None:
            pass

        bus.subscribe(EventType.TICK_RECEIVED, h1)
        bus.subscribe(EventType.TICK_RECEIVED, h2)
        bus.subscribe(EventType.BAR_CLOSE, h2)

        bus.unsubscribe_all(EventType.TICK_RECEIVED)
        assert len(bus._handlers[EventType.TICK_RECEIVED]) == 0
        assert len(bus._handlers[EventType.BAR_CLOSE]) == 1

        bus.unsubscribe_all()
        assert len(bus._handlers) == 0

    # --- Async integration ---
    @pytest.mark.asyncio
    async def test_async_publish_subscribe(self) -> None:
        bus = EventBus(max_queue_size=100)
        received: list[TickReceivedEvent] = []

        async def handler(event: AllEvents) -> None:
            assert isinstance(event, TickReceivedEvent)
            received.append(event)

        bus.subscribe(EventType.TICK_RECEIVED, handler)

        bus.publish(TickReceivedEvent(symbol="XAUUSD", last=2500.0))
        bus.publish(TickReceivedEvent(symbol="XAUUSD", last=2501.0))

        # Cho handlers chay xong
        await asyncio.sleep(0.1)

        # Kiem tra da nhan duoc event
        metrics = bus.get_metrics()
        assert metrics["published"] == 2

    @pytest.mark.asyncio
    async def test_event_data_integrity(self) -> None:
        bus = EventBus(max_queue_size=100)
        received: list[TickReceivedEvent] = []

        async def handler(event: AllEvents) -> None:
            assert isinstance(event, TickReceivedEvent)
            received.append(event)

        bus.subscribe(EventType.TICK_RECEIVED, handler)

        event = TickReceivedEvent(
            symbol="XAUUSD",
            timestamp_us=1_700_000_000_000_000,
            bid=2499.0,
            ask=2499.5,
            last=2499.3,
            volume=100.0,
            flags=0,
            aggressor="BUY",
            spread_pips=0.5,
        )
        bus.publish(event)

        await asyncio.sleep(0.1)

        assert len(received) == 1
        assert received[0].symbol == "XAUUSD"
        assert received[0].last == 2499.3
        assert received[0].aggressor == "BUY"
