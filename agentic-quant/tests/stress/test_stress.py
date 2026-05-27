# =============================================================================
# AGENTIC-QUANT — Stress Tests
#
# NFP Spike simulation: 10x tick rate trong 30s
# Verify queue khong overflow (backpressure)
# Verify guardrail kich hoat dung
# 100,000 ticks trong 5s -> verify throughput > 20k ticks/sec
# =============================================================================

from __future__ import annotations

import asyncio
import statistics
import time
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np
import pytest
import pytest_asyncio
from loguru import logger


# =============================================================================
# Constants
# =============================================================================

TICK_RATE_NORMAL = 10  # ticks/sec (normal)
TICK_RATE_SPIKE = 100  # ticks/sec (NFP spike: 10x)
SPIKE_DURATION_SEC = 30
HIGH_THROUGHPUT_TICKS = 100_000
HIGH_THROUGHPUT_WINDOW_SEC = 5.0
MIN_THROUGHPUT_TPS = 20_000  # minimum 20k ticks/sec
GUARDRAIL_Q_SIZE = 5000  # max queue size before backpressure


# =============================================================================
# Mock Components
# =============================================================================


class MockGuardrail:
    """Mock guardrail component cho stress testing.

    Mo phong guardrail mechanism:
      - kich hoat khi queue vuot qua threshold
      - ap dung dampening factor
    """

    def __init__(self, threshold: int = 4000, dampening_factor: float = 0.5) -> None:
        self.threshold = threshold
        self.dampening_factor = dampening_factor
        self.activation_count = 0
        self.active = False

    def evaluate(self, queue_size: int) -> bool:
        """Evaluate guardrail condition.

        Args:
            queue_size: Current queue size.

        Returns:
            True neu guardrail kich hoat (dampening), False neu normal.
        """
        if queue_size >= self.threshold and not self.active:
            self.activation_count += 1
            self.active = True
            logger.warning(
                f"Guardrail KICH HOAT | queue={queue_size} >= threshold={self.threshold}"
            )
        elif queue_size < self.threshold // 2 and self.active:
            self.active = False
            logger.info(
                f"Guardrail TAT | queue={queue_size} < {self.threshold // 2}"
            )

        return self.active

    def apply_dampening(self, tick_rate: float) -> float:
        """Ap dung dampening factor khi guardrail active.

        Args:
            tick_rate: Current tick rate.

        Returns:
            Dampened tick rate.
        """
        if self.active:
            return tick_rate * self.dampening_factor
        return tick_rate


class MockTickQueue:
    """Mock tick queue voi backpressure detection."""

    def __init__(self, max_size: int = GUARDRAIL_Q_SIZE) -> None:
        self.max_size = max_size
        self._queue: list[dict[str, Any]] = []
        self.overflow_count = 0
        self.max_reached = 0

    def push(self, tick: dict[str, Any]) -> bool:
        """Push tick vao queue.

        Args:
            tick: Tick data.

        Returns:
            True neu push thanh cong, False neu overflow (backpressure).
        """
        if len(self._queue) >= self.max_size:
            self.overflow_count += 1
            return False
        self._queue.append(tick)
        self.max_reached = max(self.max_reached, len(self._queue))
        return True

    def pop(self) -> dict[str, Any] | None:
        """Pop tick tu queue."""
        if not self._queue:
            return None
        return self._queue.pop(0)

    @property
    def size(self) -> int:
        """Current queue size."""
        return len(self._queue)

    def clear(self) -> None:
        """Clear queue."""
        self._queue.clear()
        self.overflow_count = 0
        self.max_reached = 0


# =============================================================================
# Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def mock_guardrail() -> MockGuardrail:
    """Tao MockGuardrail instance cho stress tests."""
    return MockGuardrail()


@pytest_asyncio.fixture
async def mock_tick_queue() -> MockTickQueue:
    """Tao MockTickQueue instance cho stress tests."""
    return MockTickQueue()


def _generate_ticks(
    count: int,
    symbol: str = "XAUUSD",
    price_base: float = 2500.0,
) -> list[dict[str, Any]]:
    """Sinh ticks voi so luong xac dinh.

    Args:
        count: So luong ticks can sinh.
        symbol: Trading symbol.
        price_base: Base price.

    Returns:
        List ticks.
    """
    rng = np.random.default_rng(42)
    ticks: list[dict[str, Any]] = []
    for i in range(count):
        price = price_base + rng.normal(0, 0.5)
        ticks.append(
            {
                "symbol": symbol,
                "bid": round(price - 0.5, 2),
                "ask": round(price + 0.5, 2),
                "last": round(price, 2),
                "volume": round(abs(rng.normal(10, 5)), 1),
                "timestamp_us": int(time.time() * 1_000_000) + i * 10_000,
                "flags": 0,
            }
        )
    return ticks


# =============================================================================
# Stress Tests
# =============================================================================


@pytest.mark.stress
class TestNfpSpikeStress:
    """Stress test: NFP Spike simulation.

    Mo phong 10x tick rate trong 30s (NFP news event).
    Verify queue khong overflow, guardrail kich hoat dung.
    """

    @pytest.mark.asyncio
    async def test_nfp_spike_backpressure(
        self,
        mock_tick_queue: MockTickQueue,
    ) -> None:
        """Test backpressure duoi NFP spike.

        Mo phong 10x tick rate (100 ticks/sec) trong 30s.
        Total ticks = 100 * 30 = 3000 ticks.
        Simulate xu ly cham (only pop 1 tick per 5 push) de queue
        tich tu va guardrail kich hoat.
        Verify queue khong overflow (backpressure) khi guardrail active.
        """
        total_ticks = TICK_RATE_SPIKE * SPIKE_DURATION_SEC
        ticks = _generate_ticks(total_ticks)
        logger.info(
            f"NFP Spike: {total_ticks} ticks @ {TICK_RATE_SPIKE} ticks/sec"
        )

        # Use lower threshold so guardrail activates within 3000 ticks
        guardrail = MockGuardrail(threshold=500, dampening_factor=0.5)
        queue = MockTickQueue(max_size=GUARDRAIL_Q_SIZE)

        # Simulate pipeline processing — xu ly cham de queue tich tu
        guardrail_log: list[bool] = []
        dropped = 0
        processed = 0
        push_count = 0

        for tick in ticks:
            success = queue.push(tick)
            push_count += 1
            if not success:
                dropped += 1
                continue

            # Evaluate guardrail sau moi push
            guardrail_active = guardrail.evaluate(queue.size)
            guardrail_log.append(guardrail_active)

            # Simulate xu ly cham: chi pop 1 tick per 5 push
            if push_count % 5 == 0:
                if guardrail_active:
                    _ = guardrail.apply_dampening(1.0)
                _ = queue.pop()
                processed += 1

        # Pop remaining ticks
        while queue.size > 0:
            _ = queue.pop()
            processed += 1

        # Assertions
        assert dropped == 0, (
            f"Queue overflow! {dropped} ticks bi drop (queue max={queue.max_size})"
        )
        assert guardrail.activation_count > 0, (
            "Guardrail khong kich hoat duoi NFP spike!"
        )
        assert processed == total_ticks, (
            f"Chi xu ly {processed}/{total_ticks} ticks"
        )

        logger.info(
            f"  Result: processed={processed}, dropped={dropped}, "
            f"guardrail_activations={guardrail.activation_count}, "
            f"max_queue={queue.max_reached}"
        )

    @pytest.mark.asyncio
    async def test_nfp_spike_guardrail_dampening(
        self,
    ) -> None:
        """Test guardrail dampening factor duoi NFP spike.

        Verify dampening factor duoc ap dung dung khi guardrail active.
        """
        guardrail = MockGuardrail(threshold=200, dampening_factor=0.5)
        ticks = _generate_ticks(500)
        active_count = 0
        dampened_values: list[float] = []

        for i, tick in enumerate(ticks):
            # Simulate increasing queue size vuot qua threshold
            queue_size = i * 2  # 0, 2, 4, ... 998

            guardrail_active = guardrail.evaluate(queue_size)
            if guardrail_active:
                active_count += 1
                dampened = guardrail.apply_dampening(TICK_RATE_SPIKE)
                dampened_values.append(dampened)
            else:
                dampened_values.append(float(TICK_RATE_SPIKE))

        assert active_count > 0, "Guardrail khong active!"
        assert any(v < TICK_RATE_SPIKE for v in dampened_values), (
            "Dampening khong duoc ap dung!"
        )

        # Verify dampening factor
        dampened_tick_rates = [v for v in dampened_values if v < TICK_RATE_SPIKE]
        if dampened_tick_rates:
            avg_dampened = statistics.mean(dampened_tick_rates)
            expected_dampened = TICK_RATE_SPIKE * guardrail.dampening_factor
            assert abs(avg_dampened - expected_dampened) < 0.01, (
                f"Dampened rate {avg_dampened:.2f} != expected {expected_dampened:.2f}"
            )

        logger.info(
            f"Guardrail dampening: {active_count} activations, "
            f"dampened_rate={TICK_RATE_SPIKE * guardrail.dampening_factor:.0f} ticks/sec"
        )


@pytest.mark.stress
class TestHighThroughputStress:
    """Stress test: High throughput.

    100,000 ticks trong 5s -> verify throughput > 20k ticks/sec.
    """

    @pytest.mark.asyncio
    async def test_high_throughput(
        self,
        mock_guardrail: MockGuardrail,
        mock_tick_queue: MockTickQueue,
    ) -> None:
        """Test throughput: 100k ticks trong 5s.

        Verify:
          - Throughput > 20k ticks/sec
          - Queue khong overflow
          - Processing hoan thanh trong window
        """
        ticks = _generate_ticks(HIGH_THROUGHPUT_TICKS)
        logger.info(
            f"High throughput: {HIGH_THROUGHPUT_TICKS} ticks "
            f"trong {HIGH_THROUGHPUT_WINDOW_SEC}s window"
        )

        t0 = time.perf_counter()
        processed = 0
        dropped = 0

        for tick in ticks:
            success = mock_tick_queue.push(tick)
            if not success:
                dropped += 1
                if dropped > 100:  # Neu drop nhieu, stop som
                    break
                continue

            # Check guardrail
            _ = mock_guardrail.evaluate(mock_tick_queue.size)

            # Process
            _ = mock_tick_queue.pop()
            processed += 1

        t1 = time.perf_counter()
        elapsed = t1 - t0
        throughput = processed / elapsed if elapsed > 0 else 0

        logger.info(
            f"  Results: processed={processed}, dropped={dropped}, "
            f"elapsed={elapsed:.3f}s, throughput={throughput:.0f} ticks/sec"
        )

        assert dropped == 0, (
            f"Queue overflow! {dropped} ticks bi drop"
        )
        assert throughput >= MIN_THROUGHPUT_TPS, (
            f"Throughput {throughput:.0f} ticks/sec < {MIN_THROUGHPUT_TPS} target"
        )

    @pytest.mark.asyncio
    async def test_throughput_with_guardrail_active(self) -> None:
        """Test throughput khi guardrail active (dampened).

        Verify van dam bao throughput > 20k ticks/sec khi guardrail active.
        """
        guardrail = MockGuardrail(threshold=100, dampening_factor=0.5)
        queue = MockTickQueue(max_size=5000)
        ticks = _generate_ticks(50_000)

        t0 = time.perf_counter()
        processed = 0
        dropped = 0

        # Pre-fill queue de guardrail kich hoat ngay
        for tick in ticks[:200]:
            success = queue.push(tick)
            if not success:
                break

        # Now process — guardrail should be active due to queue size
        for tick in ticks[200:]:
            success = queue.push(tick)
            if not success:
                dropped += 1
                continue

            guardrail_active = guardrail.evaluate(queue.size)

            # Pop moi lan de khong overflow
            _ = queue.pop()
            processed += 1

            # Mo phong dampening bang 10 iteration no-op loop
            # khi guardrail active (khong dung sleep de tranh latency)
            if guardrail_active:
                _ = sum(i * i for i in range(10))

        # Drain remaining
        while queue.size > 0:
            _ = queue.pop()
            processed += 1

        t1 = time.perf_counter()
        elapsed = t1 - t0
        throughput = processed / elapsed if elapsed > 0 else 0

        logger.info(
            f"  Guardrail throughput: processed={processed}, dropped={dropped}, "
            f"elapsed={elapsed:.3f}s, throughput={throughput:.0f} ticks/sec"
        )

        assert dropped == 0, f"Queue overflow! {dropped} ticks bi drop"
        assert throughput >= MIN_THROUGHPUT_TPS, (
            f"Throughput {throughput:.0f} ticks/sec < {MIN_THROUGHPUT_TPS} target"
        )
        assert guardrail.activation_count > 0, (
            "Guardrail khong kich hoat!"
        )
