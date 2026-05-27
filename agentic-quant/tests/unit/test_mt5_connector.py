# =============================================================================
# AGENTIC-QUANT — Unit Tests cho MT5 Connector va Simulator
# =============================================================================

from __future__ import annotations

import asyncio
import struct
import time

import pytest

from core.ingestion.mt5_simulator import MT5TickSimulator
from core.ingestion.tick_frame import TickFrame


class TestMT5TickSimulator:
    """Tests cho MT5TickSimulator."""

    @pytest.fixture
    def sim(self) -> MT5TickSimulator:
        return MT5TickSimulator(
            symbol="XAUUSD",
            base_price=2500.0,
            volatility=0.5,
            spread=0.5,
            push_address="tcp://127.0.0.1:5558",
            tick_interval_ms=0,
        )

    def test_generate_single_tick(self, sim: MT5TickSimulator) -> None:
        """Tao mot tick va kiem tra format."""
        tick = sim._generate_tick(int(time.time() * 1_000_000))

        assert tick.symbol == "XAUUSD"
        assert tick.bid > 0
        assert tick.ask > tick.bid
        assert tick.last > 0
        assert tick.volume >= 0

    def test_binary_roundtrip(self, sim: MT5TickSimulator) -> None:
        """Tick -> Binary -> Parse = same data."""
        tick = sim._generate_tick(1_700_000_000_000_000)
        data = tick.to_binary()
        restored = TickFrame.from_binary(data)

        assert restored.symbol == tick.symbol
        assert restored.timestamp_us == tick.timestamp_us
        assert restored.bid == tick.bid
        assert restored.ask == tick.ask
        assert restored.last == tick.last
        assert restored.volume == tick.volume

    def test_spread_calculation(self, sim: MT5TickSimulator) -> None:
        """Spread duoc tinh dung."""
        tick = sim._generate_tick(0)
        spread = tick.ask - tick.bid
        assert spread > 0
        assert spread < 5.0  # Khong qua 5 pip (XAUUSD)

    def test_aggressor_detection(self, sim: MT5TickSimulator) -> None:
        """Aggressor duoc xac dinh dung."""
        tick = sim._generate_tick(0)
        agg = tick.aggressor_side()

        assert agg in ("BUY", "SELL", "UNKNOWN")
        if agg == "BUY":
            assert tick.last >= tick.ask
        elif agg == "SELL":
            assert tick.last <= tick.bid

    def test_generate_batch(self, sim: MT5TickSimulator) -> None:
        """Generate 100 ticks."""
        ticks = sim.generate_batch(100)
        assert len(ticks) == 100

        for i, tick in enumerate(ticks):
            assert tick.symbol == "XAUUSD"
            assert tick.timestamp_us > 0

    def test_generate_bar_ticks(self, sim: MT5TickSimulator) -> None:
        """Tao ticks cho 10 bars M1."""
        ticks = sim.generate_bar_ticks(bar_count=10, ticks_per_bar=60)
        assert len(ticks) == 600  # 10 bars * 60 ticks

        # Kiem tra timestamp tang dan
        for i in range(1, len(ticks)):
            assert ticks[i].timestamp_us > ticks[i - 1].timestamp_us

    def test_generate_multiple_timeframes(self, sim: MT5TickSimulator) -> None:
        """Tao ticks cho nhieu timeframe."""
        for tf in ["M1", "M5", "M15", "H1"]:
            ticks = sim.generate_bar_ticks(bar_count=5, ticks_per_bar=30, timeframe=tf)
            assert len(ticks) == 150

    def test_price_momentum(self, sim: MT5TickSimulator) -> None:
        """Gia co momentum (changes smoothly)."""
        ticks = sim.generate_batch(50)

        # Tinh tong thay doi
        total_change = sum(
            abs(ticks[i].last - ticks[i - 1].last)
            for i in range(1, len(ticks))
        )

        # Khong nhay qua nhieu
        assert total_change < 100.0  # 50 ticks * vol=0.5, 95% trong khoang nay

    def test_start_stop(self, sim: MT5TickSimulator) -> None:
        """Start va stop simulator."""
        asyncio.run(sim.start())
        assert sim.is_running is True
        assert sim.tick_count == 0

        time.sleep(0.1)  # Mot so tick

        asyncio.run(sim.stop())
        assert sim.is_running is False
        assert sim.tick_count > 0


class TestMT5TickSimulatorReconnect:
    """Tests cho MT5 Simulator voi reconnect."""

    @pytest.mark.asyncio
    async def test_simulator_zmq_publish(self) -> None:
        """Simulator push tick qua ZeroMQ."""
        import zmq.asyncio

        sim = MT5TickSimulator(
            symbol="XAUUSD",
            base_price=2500.0,
            tick_interval_ms=0,
            push_address="tcp://127.0.0.1:5559",
        )

        # ZeroMQ subscriber
        ctx = zmq.asyncio.Context()
        socket = ctx.socket(zmq.PULL)
        socket.setsockopt(zmq.RCVTIMEO, 1000)
        socket.connect("tcp://127.0.0.1:5559")

        await sim.start()

        # Nhan mot so ticks
        received = []
        for _ in range(5):
            try:
                data = await socket.recv()
                tick = TickFrame.from_binary(data)
                received.append(tick)
            except zmq.Again:
                break

        await sim.stop()
        ctx.term()

        assert len(received) > 0
        assert all(t.symbol == "XAUUSD" for t in received)


class TestMT5TickFrame:
    """Tests cho TickFrame voi MT5 flags."""

    def test_mt5_tick_flags(self) -> None:
        """MT5 tick flags."""
        # Bit 0: bid changed
        tick_bid_change = TickFrame(
            symbol="XAUUSD",
            timestamp_us=0,
            bid=2500.0,
            ask=2500.5,
            last=2500.2,
            volume=10.0,
            flags=1,
        )
        assert tick_bid_change.flags & 1

        # Bit 1: high volume
        tick_vol = TickFrame(
            symbol="XAUUSD",
            timestamp_us=0,
            bid=2500.0,
            ask=2500.5,
            last=2500.2,
            volume=100.0,
            flags=2,
        )
        assert tick_vol.flags & 2

    def test_binary_size_56_bytes(self) -> None:
        """Binary luon dung 56 bytes."""
        tick = TickFrame(
            symbol="XAUUSD",
            timestamp_us=1_700_000_000_000_000,
            bid=2499.0,
            ask=2499.5,
            last=2499.3,
            volume=100.0,
            flags=0,
        )
        data = tick.to_binary()
        assert len(data) == 56

        # Deconstructed: 12 + 8 + 1 + 1 + 1 + 8 + 4 = 35 bytes header
        # Nhung price scale them 8 bytes moi -> 56
        assert len(data) == struct.calcsize("<12s Q d d d Q I")
