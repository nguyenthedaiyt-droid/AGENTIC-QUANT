# =============================================================================
# AGENTIC-QUANT — ZeroMQ Tick Receiver
# Nhan tick tu MetaTrader 5 Expert Advisor
# =============================================================================

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import zmq
import zmq.asyncio
from loguru import logger

from core.utils.events import EventBus, EventType, TickReceivedEvent

from .tick_frame import TickFrame

if TYPE_CHECKING:
    pass


# =============================================================================
# Tick Receiver
# =============================================================================
class TickReceiver:
    """
    ZeroMQ PULL socket receiver — nhan tick tu MT5 EA.

    Chay trong asyncio loop rieng, phat TickReceivedEvent len EventBus.

    Thu tu xu ly mot tick:
    1. Nhan binary message tu ZeroMQ
    2. Deserialize thanh TickFrame
    3. Validate spread
    4. Xac dinh aggressor side
    5. Phat event len EventBus

    Args:
        port: ZeroMQ PULL port (default: 5556)
        event_bus: EventBus instance de publish events
        abnormal_spread_threshold: Nguong spread bat thuong (pips)
    """

    def __init__(
        self,
        port: int = 5556,
        event_bus: EventBus | None = None,
        abnormal_spread_threshold: float = 0.5,
    ) -> None:
        self.port = port
        self.event_bus = event_bus
        self.spread_threshold = abnormal_spread_threshold

        self._ctx: zmq.asyncio.Context | None = None
        self._socket: zmq.asyncio.Socket | None = None
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._prev_price: float | None = None
        self._last_success_time: float = 0.0
        self._reconnect_delay: float = 1.0
        self._max_reconnect_delay: float = 60.0
        self._tick_count: int = 0
        self._last_log_time: float = 0.0

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    async def start(self) -> None:
        """Bat dau receiver (goi mot lan)."""
        if self._running:
            logger.warning("TickReceiver da dang chay")
            return

        self._running = True
        self._ctx = zmq.asyncio.Context.instance()
        self._task = asyncio.create_task(self._receive_loop())
        logger.info("TickReceiver bat dau tren port {port}", port=self.port)

    async def stop(self) -> None:
        """Dung receiver."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._close_socket()
        logger.info("TickReceiver da dung")

    def _close_socket(self) -> None:
        """Dong ZeroMQ socket."""
        if self._socket:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
            self._socket = None

    # -------------------------------------------------------------------------
    # Receive loop
    # -------------------------------------------------------------------------
    async def _receive_loop(self) -> None:
        """Main loop — connect, nhan, va xu ly tick."""
        while self._running:
            try:
                await self._connect()
                await self._recv_forever()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("TickReceiver gap loi, se reconnect...")
                self._close_socket()
                await self._wait_before_reconnect()

    async def _connect(self) -> None:
        """Ket noi den ZeroMQ PULL socket."""
        if self._ctx is None:
            self._ctx = zmq.asyncio.Context.instance()

        self._socket = self._ctx.socket(zmq.PULL)
        self._socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5s timeout
        self._socket.setsockopt(zmq.LINGER, 0)  # Dont wait on close

        connect_url = f"tcp://127.0.0.1:{self.port}"
        self._socket.connect(connect_url)
        logger.info("TickReceiver da ket noi den {url}", url=connect_url)

        # Reset reconnect state
        self._reconnect_delay = 1.0

    async def _recv_forever(self) -> None:
        """Vong lap nhan tick."""
        while self._running:
            try:
                data = await self._socket.recv()
                await self._process_tick(data)
            except zmq.Again:
                # Timeout — kiem tra running state
                if not self._running:
                    break
                await self._check_staleness()
            except zmq.ZMQError as e:
                if not self._running:
                    break
                logger.error("ZeroMQ error: {e}", e=e)
                raise

    async def _process_tick(self, data: bytes) -> None:
        """Xu ly mot tick message."""
        try:
            tick = TickFrame.from_binary(data)
        except Exception:
            logger.warning("Khong the parse tick binary: {data!r}", data=data[:20])
            return

        self._tick_count += 1
        self._last_success_time = time.time()

        # Kiem tra spread bat thuong
        is_abnormal = tick.is_abnormal_spread(self.spread_threshold)

        # Xac dinh aggressor
        aggressor = tick.aggressor_side(self._prev_price)
        self._prev_price = tick.last

        # Tao event
        event = TickReceivedEvent(
            symbol=tick.symbol,
            timestamp_us=tick.timestamp_us,
            bid=tick.bid,
            ask=tick.ask,
            last=tick.last,
            volume=tick.volume,
            flags=tick.flags,
            is_abnormal_spread=is_abnormal,
            aggressor=aggressor,
            spread_pips=tick.spread_pips,
            mid_price=tick.mid_price,
        )

        # Publish len EventBus
        if self.event_bus:
            self.event_bus.publish(event)

        # Log thong tin tick moi 5s
        await self._maybe_log_tick(tick, aggressor, is_abnormal)

    async def _maybe_log_tick(
        self,
        tick: TickFrame,
        aggressor: str,
        is_abnormal: bool,
    ) -> None:
        """Log thong tin tick moi 5s (tranh spam)."""
        now = time.time()
        if now - self._last_log_time >= 5.0:
            self._last_log_time = now
            logger.debug(
                "Tick: {symbol} bid={bid:.2f} ask={ask:.2f} "
                "last={last:.2f} vol={vol:.0f} "
                "spread={spread:.1f}pips aggressor={agg}",
                symbol=tick.symbol,
                bid=tick.bid,
                ask=tick.ask,
                last=tick.last,
                vol=tick.volume,
                spread=tick.spread_pips,
                agg=aggressor,
            )
            if is_abnormal:
                logger.warning(
                    "Abnormal spread detected: {spread:.1f} pips on {symbol}",
                    spread=tick.spread_pips,
                    symbol=tick.symbol,
                )

    async def _check_staleness(self) -> None:
        """Kiem tra tick data co bi stale khong."""
        if self._last_success_time > 0:
            age = time.time() - self._last_success_time
            if age > 30.0:
                logger.warning(
                    "Khong nhan tick trong {age:.0f}s", age=age
                )
                if self.event_bus:
                    from core.utils.events import StalenessAlertEvent
                    self.event_bus.publish(StalenessAlertEvent(
                        feed_name="MT5_ZMQ",
                        last_update_age_seconds=int(age),
                        threshold_seconds=30,
                    ))

    async def _wait_before_reconnect(self) -> None:
        """Cho truoc khi reconnect voi exponential backoff."""
        logger.info(
            "Doi {delay:.1f}s truoc khi reconnect...",
            delay=self._reconnect_delay,
        )
        await asyncio.sleep(self._reconnect_delay)
        self._reconnect_delay = min(
            self._reconnect_delay * 2,
            self._max_reconnect_delay,
        )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    @property
    def tick_count(self) -> int:
        """Tong so tick da nhan."""
        return self._tick_count

    @property
    def is_connected(self) -> bool:
        """Kiem tra co dang nhan tick hay khong."""
        return self._running and self._socket is not None
