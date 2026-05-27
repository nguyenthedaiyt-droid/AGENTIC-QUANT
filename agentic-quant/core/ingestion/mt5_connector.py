# =============================================================================
# AGENTIC-QUANT — MT5 Connector
# Ket noi truc tiep den MT5 Terminal qua official Python API
# Khong can EA, khong can ZeroMQ PUSH ben MT5
# =============================================================================

from __future__ import annotations

import asyncio
import struct
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import zmq
import zmq.asyncio
from loguru import logger

from core.ingestion.tick_frame import TickFrame
from core.utils.events import EventBus, TickReceivedEvent

if TYPE_CHECKING:
    pass


# =============================================================================
# MT5 Connection Config
# =============================================================================
@dataclass
class MT5ConnectionConfig:
    """Cau hinh ket noi MT5."""

    terminal_path: str = ""  # Duong dan den MT5 terminal .exe (neu can khoi dong)
    login: int = 0  # MT5 account login
    password: str = ""  # MT5 password
    server: str = ""  # MT5 server (broker)
    uuid: str = ""  # UUID de nhan dien

    zmq_push_address: str = "tcp://127.0.0.1:5556"  # Day tick den day
    tick_publish_interval_ms: int = 0  # 0 = tat ca tick, >0 = throttle

    reconnect_delay_sec: float = 5.0
    max_reconnect_delay: float = 60.0
    connection_timeout_sec: int = 30


class MT5Connector:
    """
    Ket noi MT5 Terminal bang official MetaTrader5 Python API.

    Hoat dong:
    1. Khoi tao ket noi MT5
    2. Dang ky nhan tick real-time
    3. Dong goi thanh TickFrame binary
    4. Push qua ZeroMQ PUSH socket (de TickReceiver nhan)

    Chu y: MT5 Python API chi hoat dong khi MT5 Terminal dang mo.

    Args:
        config: MT5ConnectionConfig
        event_bus: EventBus (optional)
    """

    def __init__(
        self,
        config: MT5ConnectionConfig | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.config = config or MT5ConnectionConfig()
        self.event_bus = event_bus

        self._mt5 = None  # MetaTrader5 module
        self._connected = False
        self._running = False
        self._task: asyncio.Task[None] | None = None

        # ZeroMQ PUSH socket
        self._zmq_ctx: zmq.asyncio.Context | None = None
        self._zmq_socket: zmq.asyncio.Socket | None = None

        # Rate limiting
        self._last_publish_time: float = 0.0
        self._tick_count: int = 0
        self._last_log_time: float = 0.0

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    async def connect(self) -> bool:
        """
        Ket noi den MT5 Terminal.

        Tra ve True neu thanh cong, False neu that bai.
        """
        try:
            import MetaTrader5 as mt5

            self._mt5 = mt5

            # Khoi tao MT5
            if not mt5.initialize():
                error = mt5.last_error()
                logger.error(
                    "MT5 initialize that bai: error={error}",
                    error=error,
                )
                return False

            logger.info(
                "MT5 Terminal connected: version={version}, build={build}",
                version=mt5.terminal_info().version,
                build=mt5.terminal_info().build,
            )

            # Login neu co thong tin
            if self.config.login > 0:
                if not mt5.login(
                    self.config.login,
                    password=self.config.password,
                    server=self.config.server,
                ):
                    logger.error("MT5 login that bai: {error}", error=mt5.last_error())
                    return False
                logger.info("MT5 logged in as account {login}", login=self.config.login)

            self._connected = True
            return True

        except ImportError:
            logger.error(
                "MetaTrader5 package chua duoc cai dat. "
                "Chay: pip install MetaTrader5"
            )
            return False
        except Exception:
            logger.exception("MT5 connect that bai")
            return False

    async def disconnect(self) -> None:
        """Ngat ket noi MT5."""
        self._running = False

        if self._zmq_socket:
            self._zmq_socket.close(linger=0)
            self._zmq_socket = None
        if self._zmq_ctx:
            self._zmq_ctx.term()
            self._zmq_ctx = None

        if self._mt5:
            self._mt5.shutdown()
            self._mt5 = None

        self._connected = False
        logger.info("MT5 disconnected")

    async def start(self) -> None:
        """
        Bat dau nhan tick va push qua ZeroMQ.

        Tao ZeroMQ PUSH socket, lay symbol info, bat dau tick loop.
        """
        if not self._connected:
            raise RuntimeError("MT5 chua duoc ket noi. Goi connect() truoc.")

        self._running = True

        # Setup ZeroMQ PUSH
        self._zmq_ctx = zmq.asyncio.Context()
        self._zmq_socket = self._zmq_ctx.socket(zmq.PUSH)
        self._zmq_socket.setsockopt(zmq.LINGER, 1000)
        self._zmq_socket.bind(self.config.zmq_push_address)

        logger.info(
            "MT5 tick stream started, pushing to {addr}",
            addr=self.config.zmq_push_address,
        )

        # Chay tick loop
        self._task = asyncio.create_task(self._tick_loop())

    async def stop(self) -> None:
        """Dung tick stream."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.disconnect()

    # -------------------------------------------------------------------------
    # Tick Loop
    # -------------------------------------------------------------------------
    async def _tick_loop(self) -> None:
        """Vong lap nhan tick tu MT5."""
        import MetaTrader5 as mt5

        while self._running:
            try:
                # Lay tick moi nhat
                symbol = self.config.uuid or "XAUUSD"
                tick = mt5.symbol_info_tick(symbol)

                if tick is None:
                    await asyncio.sleep(0.001)
                    continue

                # Throttle
                now = time.time()
                if self.config.tick_publish_interval_ms > 0:
                    interval = self.config.tick_publish_interval_ms / 1000.0
                    if now - self._last_publish_time < interval:
                        await asyncio.sleep(0.001)
                        continue

                self._last_publish_time = now

                # Chuyen thanh TickFrame
                frame = self._mt5_tick_to_frame(symbol, tick)

                # Push qua ZeroMQ
                if self._zmq_socket:
                    await self._zmq_socket.send(frame.to_binary())

                # Publish len EventBus
                if self.event_bus:
                    event = self._frame_to_event(frame)
                    self.event_bus.publish(event)

                self._tick_count += 1
                await self._maybe_log_stats(frame)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Tick loop error")
                await asyncio.sleep(1.0)

    async def _maybe_log_stats(self, frame: TickFrame) -> None:
        """Log thong ke moi 5s."""
        now = time.time()
        if now - self._last_log_time >= 5.0:
            self._last_log_time = now
            logger.debug(
                "MT5 Ticks sent: {count} @ {price}",
                count=self._tick_count,
                price=frame.last,
            )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _mt5_tick_to_frame(self, symbol: str, tick) -> TickFrame:
        """Chuyen MT5 MqlTick thanh TickFrame."""
        # timestamp_ms = miliseconds, chuyen thanh microseconds
        ts_us = tick.time_msc * 1000

        return TickFrame(
            symbol=symbol,
            timestamp_us=ts_us,
            bid=float(tick.bid),
            ask=float(tick.ask),
            last=float(tick.last),
            volume=float(tick.volume),
            flags=int(tick.flags),
        )

    def _frame_to_event(self, frame: TickFrame) -> TickReceivedEvent:
        """Chuyen TickFrame thanh TickReceivedEvent."""
        is_abnormal = frame.is_abnormal_spread(threshold_pips=0.5)
        aggressor = frame.aggressor_side()

        return TickReceivedEvent(
            symbol=frame.symbol,
            timestamp_us=frame.timestamp_us,
            bid=frame.bid,
            ask=frame.ask,
            last=frame.last,
            volume=frame.volume,
            flags=frame.flags,
            is_abnormal_spread=is_abnormal,
            aggressor=aggressor,
            spread_pips=frame.spread_pips,
            mid_price=frame.mid_price,
        )

    # -------------------------------------------------------------------------
    # Symbols
    # -------------------------------------------------------------------------
    def get_symbols(self) -> list[str]:
        """Lay danh sach symbols co san trong MT5."""
        import MetaTrader5 as mt5

        symbols = mt5.symbols_get()
        return [s.name for s in symbols] if symbols else []

    def get_symbol_info(self, symbol: str) -> dict | None:
        """Lay thong tin symbol."""
        import MetaTrader5 as mt5

        info = mt5.symbol_info(symbol)
        if info is None:
            return None

        return {
            "name": info.name,
            "description": info.description,
            "bid": info.bid,
            "ask": info.ask,
            "last": info.last,
            "volume": info.volume,
            "digits": info.digits,
            "spread": info.spread,
            "point": info.point,
        }

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def tick_count(self) -> int:
        return self._tick_count
