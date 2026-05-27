# =============================================================================
# AGENTIC-QUANT — MT5 Tick Simulator
# Gia lap tick MT5 cho backtesting va integration test
# Khong can MT5 Terminal
# =============================================================================

from __future__ import annotations

import asyncio
import random
import struct
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import zmq
import zmq.asyncio
from loguru import logger

from core.ingestion.tick_frame import TickFrame

if TYPE_CHECKING:
    pass


@dataclass
class SimulatedMarketState:
    """Trang thai thi truong gia lap."""

    symbol: str
    base_price: float
    volatility: float  # Std dev per tick
    spread: float  # Bid-ask spread
    tick_volume: float

    # Trend
    trend_direction: int = 0  # -1, 0, 1
    trend_strength: float = 0.0
    trend_remaining: int = 0  # Ticks remaining in trend

    # Momentum
    momentum: float = 0.0
    bid: float = 0.0
    ask: float = 0.0


class MT5TickSimulator:
    """
    Gia lap tick MT5 real-time.

    Tao tick nhu MT5 thuc te:
    - Price co tinh xu huong (momentum)
    - Spread bat dong thuong
    - Volume thay doi

    Co the push truc tiep qua ZeroMQ PUSH (de TickReceiver nhan).

    Args:
        symbol: Symbol giao dich (default: XAUUSD)
        base_price: Gia bat dau
        volatility: Do biến động mỗi tick
        push_address: ZeroMQ address de push (tcp://127.0.0.1:5556)
        tick_interval_ms: Khoang cach giua cac tick (0 = max speed)
    """

    def __init__(
        self,
        symbol: str = "XAUUSD",
        base_price: float = 4450.0,
        volatility: float = 0.05,  # ~0.5 pip per tick cho XAUUSD
        spread: float = 0.5,  # ~5 pips (XAUUSD default)
        push_address: str = "tcp://127.0.0.1:5556",
        tick_interval_ms: int = 100,
    ) -> None:
        self.symbol = symbol
        self.push_address = push_address
        self.tick_interval_ms = tick_interval_ms

        # spread: price units (~0.5 = 5 pips for XAUUSD)
        # volatility: price units per tick (~0.05 = 0.5 pip per tick)
        self._state = SimulatedMarketState(
            symbol=symbol,
            base_price=base_price,
            volatility=volatility,
            spread=spread,
            tick_volume=10.0,
        )

        self._zmq_ctx: zmq.asyncio.Context | None = None
        self._zmq_socket: zmq.asyncio.Socket | None = None
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._tick_count: int = 0
        self._start_time: float = 0.0

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    async def start(self) -> None:
        """Bat dau simulator."""
        self._zmq_ctx = zmq.asyncio.Context()
        self._zmq_socket = self._zmq_ctx.socket(zmq.PUSH)
        self._zmq_socket.setsockopt(zmq.LINGER, 1000)
        self._zmq_socket.bind(self.push_address)

        self._running = True
        self._start_time = time.time()
        self._task = asyncio.create_task(self._sim_loop())

        logger.info(
            "MT5 Simulator started: {symbol} @ {price}, push to {addr}",
            symbol=self.symbol,
            price=self._state.base_price,
            addr=self.push_address,
        )

    async def stop(self) -> None:
        """Dung simulator."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        if self._zmq_socket:
            self._zmq_socket.close(linger=0)
            self._zmq_socket = None
        if self._zmq_ctx:
            self._zmq_ctx.term()
            self._zmq_ctx = None

        elapsed = time.time() - self._start_time
        rate = self._tick_count / max(elapsed, 1)
        logger.info(
            "MT5 Simulator stopped: {count} ticks in {elapsed:.1f}s ({rate:.1f}/s)",
            count=self._tick_count,
            elapsed=elapsed,
            rate=rate,
        )

    # -------------------------------------------------------------------------
    # Simulation loop
    # -------------------------------------------------------------------------
    async def _sim_loop(self) -> None:
        """Vong lap chinh - tao tick."""
        ts_offset = int(time.time() * 1_000_000)  # Microseconds

        while self._running:
            try:
                tick = self._generate_tick(ts_offset)
                ts_offset += self.tick_interval_ms * 1000

                # Push to ZeroMQ
                if self._zmq_socket:
                    await self._zmq_socket.send(tick.to_binary())

                self._tick_count += 1

                # Throttle
                if self.tick_interval_ms > 0:
                    await asyncio.sleep(self.tick_interval_ms / 1000.0)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Simulator error")
                break

    def _generate_tick(self, ts_offset_us: int) -> TickFrame:
        """Tao mot tick gia lap."""
        state = self._state

        # Cap nhat xu huong
        self._update_trend()

        # Tinh price change
        random_component = random.gauss(0, state.volatility)
        trend_component = state.trend_direction * state.trend_strength * state.volatility
        momentum_component = state.momentum * 0.3

        price_delta = random_component + trend_component + momentum_component
        state.base_price += price_delta

        # Momentum decay
        state.momentum *= 0.95
        state.momentum += price_delta * 0.1

        # Spread co the thay doi
        spread_variation = random.uniform(-0.1, 0.1)
        actual_spread = max(0.1, state.spread + spread_variation)

        bid = round(state.base_price - actual_spread / 2, 2)
        ask = round(state.base_price + actual_spread / 2, 2)

        # Last = mid hoac gap
        if random.random() < 0.9:
            last = round((bid + ask) / 2, 2)
        elif random.random() < 0.5:
            last = ask  # Buy aggressor
        else:
            last = bid  # Sell aggressor

        # Volume co hieu ung
        volume_multiplier = 1.0 + abs(state.momentum) / state.volatility
        base_vol = random.uniform(5.0, 20.0)
        volume = round(base_vol * volume_multiplier, 2)

        state.bid = bid
        state.ask = ask

        # Flags (MT5 tick flags)
        flags = 0
        if price_delta > state.volatility * 0.5:
            flags |= 1  # Bid changed
        if volume > 15.0:
            flags |= 2  # High volume

        # Neu last vuot khoi bid-ask -> set flags
        if last > ask:
            flags |= 4  # Ask touched
        if last < bid:
            flags |= 8  # Bid touched

        return TickFrame(
            symbol=self.symbol,
            timestamp_us=ts_offset_us,
            bid=bid,
            ask=ask,
            last=last,
            volume=volume,
            flags=flags,
        )

    def _update_trend(self) -> None:
        """Cap nhat xu huong thi truong."""
        state = self._state

        # Neu trend con lai
        if state.trend_remaining > 0:
            state.trend_remaining -= 1
            return

        # Random tao trend moi
        if random.random() < 0.05:  # 5% chance moi tick
            state.trend_direction = random.choice([-1, 1])
            state.trend_strength = random.uniform(0.5, 2.0)
            state.trend_remaining = random.randint(5, 30)  # 5-30 ticks

    # -------------------------------------------------------------------------
    # Batch generation (cho backtest)
    # -------------------------------------------------------------------------
    def generate_batch(self, count: int) -> list[TickFrame]:
        """
        Tao N ticks cho backtest (sync, khong can asyncio).

        Args:
            count: So tick can tao

        Returns:
            List TickFrame
        """
        ticks = []
        base_ts = int(time.time() * 1_000_000)

        for i in range(count):
            tick = self._generate_tick(base_ts + i * 100_000)
            ticks.append(tick)

        return ticks

    def generate_bar_ticks(
        self,
        bar_count: int,
        ticks_per_bar: int = 60,
        timeframe: str = "M1",
    ) -> list[TickFrame]:
        """
        Tao ticks cho N bars.

        Args:
            bar_count: So bar can tao
            ticks_per_bar: So tick trong moi bar
            timeframe: Khung thoi gian (M1, M5, etc.)

        Returns:
            List TickFrame
        """
        ticks = []
        total_ticks = bar_count * ticks_per_bar

        tf_seconds = {
            "M1": 60, "M5": 300, "M15": 900,
            "H1": 3600, "H4": 14400, "D1": 86400,
        }
        tf_sec = tf_seconds.get(timeframe, 60)

        base_ts = int(time.time() * 1_000_000)

        for i in range(total_ticks):
            bucket_start = (i // ticks_per_bar) * tf_sec * 1_000_000
            ts = bucket_start + (i % ticks_per_bar) * 10_000_000 // ticks_per_bar
            tick = self._generate_tick(ts)
            tick.timestamp_us = ts
            ticks.append(tick)

        return ticks

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def current_price(self) -> float:
        return self._state.base_price
