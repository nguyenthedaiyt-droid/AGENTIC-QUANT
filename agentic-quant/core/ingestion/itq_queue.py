# =============================================================================
# AGENTIC-QUANT — Incoming Tick Queue (ITQ)
# Xu ly queue overflow voi Dynamic K sampling
# =============================================================================

from __future__ import annotations

import asyncio
import random
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

from loguru import logger

from core.ingestion.tick_frame import TickFrame
from core.utils.events import TickReceivedEvent
from core.utils.metrics import ITQ_QUEUE_DEPTH

if TYPE_CHECKING:
    pass


@dataclass
class QueuedTick:
    """Tick trong queue chua duoc xu ly."""

    tick: TickFrame
    received_at_us: int
    sequence: int


class IncomingTickQueue:
    """
    ITQ - Incoming Tick Queue voi Dynamic K sampling.

    Co che hoat dong:
    - Normal mode: K=1, moi tick deu duoc xu ly
    - Overflow mode: K > 1, chi xu ly 1/K tick
    - K tinh toan dong dua tren queue depth

    Args:
        max_size: Dung luong toi da cua queue (default: 10000)
        sample_threshold: Muc ma tai do K bat dau tang (default: 8000)
        k_dynamic: Cho phep K dong (default: True)
        max_k: Gia tri K toi da (default: 10)
    """

    def __init__(
        self,
        max_size: int = 10000,
        sample_threshold: int = 8000,
        k_dynamic: bool = True,
        max_k: int = 10,
    ) -> None:
        self._max_size = max_size
        self._sample_threshold = sample_threshold
        self._k_dynamic = k_dynamic
        self._max_k = max_k

        self._queue: deque[QueuedTick] = deque(maxlen=max_size)
        self._sequence: int = 0
        self._total_enqueued: int = 0
        self._total_dequeued: int = 0
        self._total_sampled: int = 0  # Tick bi skip
        self._is_overflow: bool = False
        self._current_k: int = 1

    # -------------------------------------------------------------------------
    # Enqueue
    # -------------------------------------------------------------------------
    def enqueue(self, tick: TickFrame, received_at_us: int) -> bool:
        """
        Day tick vao queue.

        Tra ve True neu duoc enqueue, False neu bi drop vi queue day.
        Neu queue full va khoong chan, tick bi drop.
        """
        self._total_enqueued += 1

        if len(self._queue) >= self._max_size:
            # Queue day - drop oldest tick
            try:
                self._queue.popleft()
            except IndexError:
                pass
            self._total_sampled += 1
            logger.warning(
                "ITQ overflow: dropped oldest tick. "
                "queue_size={size}, total_dropped={dropped}",
                size=len(self._queue),
                dropped=self._total_sampled,
            )

        self._queue.append(QueuedTick(
            tick=tick,
            received_at_us=received_at_us,
            sequence=self._sequence,
        ))
        self._sequence += 1

        # Cap nhat metrics
        ITQ_QUEUE_DEPTH.set(len(self._queue))

        # Tinh K moi khi co 100 ticks moi
        if self._k_dynamic and self._total_enqueued % 100 == 0:
            self._update_k()

        return True

    # -------------------------------------------------------------------------
    # Dequeue with Dynamic K sampling
    # -------------------------------------------------------------------------
    def dequeue(self) -> QueuedTick | None:
        """
        Lay tick tu queue voi Dynamic K sampling.

        Neu K > 1, chi tra ve 1/K ticks (randomly sampled).
        Tra ve None neu queue rong.
        """
        if not self._queue:
            return None

        if self._current_k == 1:
            # Normal mode: lay tick dau tien
            qt = self._queue.popleft()
            self._total_dequeued += 1
            return qt

        # Overflow mode: randomly sample 1/K ticks
        if random.randint(1, self._current_k) != 1:
            # Skip this tick
            qt = self._queue.popleft()
            self._total_sampled += 1
            return None

        qt = self._queue.popleft()
        self._total_dequeued += 1
        return qt

    def dequeue_all(self) -> list[QueuedTick]:
        """
        Lay tat ca ticks, ap dung K sampling.

        Dung khi can xu ly batch.
        """
        result = []
        while self._queue:
            tick = self.dequeue()
            if tick:
                result.append(tick)
        return result

    def peek(self) -> QueuedTick | None:
        """Xem tick dau tien nhung khong remove."""
        if self._queue:
            return self._queue[0]
        return None

    # -------------------------------------------------------------------------
    # Flush (for backtest)
    # -------------------------------------------------------------------------
    def flush(self) -> list[QueuedTick]:
        """Xoa het queue, tra ve tat ca ticks."""
        result = list(self._queue)
        self._queue.clear()
        self._total_dequeued += len(result)
        ITQ_QUEUE_DEPTH.set(0)
        return result

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------
    def _update_k(self) -> None:
        """Tinh K dong dua tren queue depth."""
        depth = len(self._queue)
        was_overflow = self._is_overflow

        if depth >= self._max_size:
            self._current_k = self._max_k
            self._is_overflow = True
        elif depth >= self._sample_threshold:
            # Linear interpolation: [threshold, max] -> [1, max_k]
            excess = depth - self._sample_threshold
            range_size = self._max_size - self._sample_threshold
            ratio = excess / max(range_size, 1)
            self._current_k = max(1, int(1 + ratio * (self._max_k - 1)))
            self._is_overflow = True
        else:
            self._current_k = 1
            self._is_overflow = False

        if self._is_overflow and not was_overflow:
            logger.warning(
                "ITQ entered overflow mode: K={k}, queue_depth={depth}",
                k=self._current_k,
                depth=depth,
            )
        elif not self._is_overflow and was_overflow:
            logger.info("ITQ returned to normal mode")

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    @property
    def size(self) -> int:
        """So tick hien tai trong queue."""
        return len(self._queue)

    @property
    def is_overflow(self) -> bool:
        """Queue co dang o overflow mode khong."""
        return self._is_overflow

    @property
    def current_k(self) -> int:
        """Gia tri K hien tai."""
        return self._current_k

    @property
    def stats(self) -> dict:
        """Thong ke queue."""
        return {
            "size": len(self._queue),
            "max_size": self._max_size,
            "total_enqueued": self._total_enqueued,
            "total_dequeued": self._total_dequeued,
            "total_sampled": self._total_sampled,
            "current_k": self._current_k,
            "is_overflow": self._is_overflow,
            "utilization_pct": len(self._queue) / max(self._max_size, 1) * 100,
        }
