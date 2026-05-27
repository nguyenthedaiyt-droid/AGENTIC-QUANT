# =============================================================================
# AGENTIC-QUANT — Asyncio-based Central Event Bus
# Pattern: pub/sub voi asyncio queues, non-blocking
# =============================================================================

from __future__ import annotations

import asyncio
import weakref
from collections import defaultdict
from collections.abc import Awaitable
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from loguru import logger

from .types import AllEvents, EventHandler, EventType

if TYPE_CHECKING:
    pass


# =============================================================================
# EventBus: Central pub/sub hub
# =============================================================================
class EventBus:
    """
    Central Event Bus - asyncio-based publish/subscribe.

    Su dung asyncio.Queue de dam bao non-blocking.
    Cac handlers chay trong background task, khong block publisher.

    Vi du su dung::

        bus = EventBus()

        async def on_tick(event):
            print(f"Tick: {event.last}")

        bus.subscribe(EventType.TICK_RECEIVED, on_tick)
        bus.publish(TickReceivedEvent(symbol="XAUUSD", last=2500.0))
    """

    def __init__(
        self,
        max_queue_size: int = 1000,
        default_handler_timeout: float = 5.0,
    ) -> None:
        self._queues: dict[EventType, list[asyncio.Queue[AllEvents]]] = (
            defaultdict(list)
        )
        self._handlers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._running_tasks: set[asyncio.Task[None]] = set()
        self._shutdown = asyncio.Event()
        self._lock = asyncio.Lock()
        self._max_queue_size = max_queue_size
        self._default_handler_timeout = default_handler_timeout
        self._metrics: dict[str, int] = defaultdict(int)
        self._event_counts: dict[str, int] = defaultdict(int)
        self._dropped_counts: dict[str, int] = defaultdict(int)

    # -------------------------------------------------------------------------
    # Publish
    # -------------------------------------------------------------------------
    def publish(self, event: AllEvents) -> None:
        """
        Phat event den tat ca subscribers.

        Non-blocking - chi day event vao queue cua moi handler.
        Neu queue day, event bi drop va dem dropped_counts tang.
        """
        event_type = event.event_type
        self._metrics["events_published"] += 1
        self._event_counts[str(event_type)] += 1

        if event_type not in self._handlers:
            return

        for handler in self._handlers[event_type]:
            self._queue_event(handler, event)

    def _queue_event(self, handler: EventHandler, event: AllEvents) -> None:
        """Day event vao handler queue, tao task neu can."""
        # Tao mot queue rieng cho handler nay
        q: asyncio.Queue[AllEvents] = asyncio.Queue(
            maxsize=self._max_queue_size
        )
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            self._dropped_counts[str(event.event_type)] += 1
            logger.warning(
                "Event dropped (queue full): {type}",
                type=str(event.event_type),
            )
            return

        # Tao hoac reuse task cho handler
        task = asyncio.create_task(self._run_handler(handler, q))
        self._running_tasks.add(task)
        task.add_done_callback(self._running_tasks.discard)

    async def _run_handler(
        self,
        handler: EventHandler,
        queue: asyncio.Queue[AllEvents],
    ) -> None:
        """Chay handler voi timeout, xu ly exceptions."""
        try:
            event = await asyncio.wait_for(
                queue.get(),
                timeout=self._default_handler_timeout,
            )
            await handler(event)
            self._metrics["events_processed"] += 1
        except asyncio.TimeoutError:
            self._metrics["handler_timeouts"] += 1
            logger.warning(
                "Handler timeout after {t}s: {handler}",
                t=self._default_handler_timeout,
                handler=handler,
            )
        except Exception:
            logger.exception(
                "Handler error for event: {handler}",
                handler=handler,
            )
            self._metrics["handler_errors"] += 1

    # -------------------------------------------------------------------------
    # Subscribe
    # -------------------------------------------------------------------------
    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> Callable[[], None]:
        """
        Dang ky handler cho event type.

        Tra ve callable de unsubscribe.
        """
        self._handlers[event_type].append(handler)

        logger.debug(
            "Subscribed {handler} to {type}",
            handler=handler,
            type=event_type.value,
        )

        def unsubscribe() -> None:
            try:
                self._handlers[event_type].remove(handler)
                logger.debug(
                    "Unsubscribed {handler} from {type}",
                    handler=handler,
                    type=event_type.value,
                )
            except ValueError:
                pass

        return unsubscribe

    def subscribe_many(
        self,
        handlers: dict[EventType, EventHandler],
    ) -> Callable[[], None]:
        """
        Dang ky nhieu handlers cung luc.

        Tra ve callable de unsubscribe tat ca.
        """
        unsubscribes: list[Callable[[], None]] = []
        for event_type, handler in handlers.items():
            unsubscribes.append(self.subscribe(event_type, handler))

        def unsubscribe_all() -> None:
            for unsub in unsubscribes:
                unsub()

        return unsubscribe_all

    # -------------------------------------------------------------------------
    # Once (subscribe chi mot lan, tu dong unsubscribe)
    # -------------------------------------------------------------------------
    def once(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> None:
        """Dang ky handler chi chay mot lan, tu dong unsubscribes sau khi chay."""

        async def wrapper(event: AllEvents) -> None:
            await handler(event)
            self.unsubscribe(event_type, wrapper)

        self.subscribe(event_type, wrapper)

    # -------------------------------------------------------------------------
    # Unsubscribe
    # -------------------------------------------------------------------------
    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Huy dang ky handler."""
        try:
            self._handlers[event_type].remove(handler)
        except ValueError:
            pass

    def unsubscribe_all(self, event_type: EventType | None = None) -> None:
        """Huy tat ca handlers cho mot event type, hoac tat ca neu None."""
        if event_type is None:
            self._handlers.clear()
        else:
            self._handlers[event_type].clear()

    # -------------------------------------------------------------------------
    # Metrics & Health
    # -------------------------------------------------------------------------
    def get_metrics(self) -> dict[str, Any]:
        """Tra ve metrics de监控."""
        return {
            "published": self._metrics["events_published"],
            "processed": self._metrics["events_processed"],
            "handler_errors": self._metrics["handler_errors"],
            "handler_timeouts": self._metrics["handler_timeouts"],
            "dropped_by_type": dict(self._dropped_counts),
            "published_by_type": dict(self._event_counts),
            "active_tasks": len(self._running_tasks),
            "subscribers": {
                et.value: len(handlers)
                for et, handlers in self._handlers.items()
            },
        }

    async def wait_until_empty(self, timeout: float = 5.0) -> bool:
        """Cho den khi tat ca queued events duoc xu ly."""
        start = asyncio.get_event_loop().time()
        while self._running_tasks:
            if asyncio.get_event_loop().time() - start > timeout:
                return False
            await asyncio.sleep(0.05)
        return True

    async def shutdown(self) -> None:
        """Huy tat ca running tasks."""
        self._shutdown.set()
        for task in self._running_tasks:
            task.cancel()
        if self._running_tasks:
            await asyncio.gather(*self._running_tasks, return_exceptions=True)
        self._running_tasks.clear()
        logger.info("EventBus shut down")


# =============================================================================
# Singleton instance
# =============================================================================
_global_bus: EventBus | None = None
_bus_lock = asyncio.Lock()


async def get_event_bus() -> EventBus:
    """Lay singleton EventBus instance."""
    global _global_bus
    async with _bus_lock:
        if _global_bus is None:
            _global_bus = EventBus()
        return _global_bus


def get_event_bus_sync() -> EventBus:
    """Lay singleton EventBus instance (sync, chi dung trong running event loop)."""
    global _global_bus
    if _global_bus is None:
        raise RuntimeError(
            "EventBus chua duoc khoi tao. Goi await get_event_bus() truoc."
        )
    return _global_bus
