# =============================================================================
# AGENTIC-QUANT — Timing Utilities
# =============================================================================

from __future__ import annotations

import time
import functools
from typing import TYPE_CHECKING, Callable, TypeVar

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger

try:
    from prometheus_client import Histogram
    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False

F = TypeVar("F", bound=Callable[..., object])


# =============================================================================
# Decorator do luong
# =============================================================================

def measure_latency(
    metric_name: str = "default",
    histogram: "Histogram | None" = None,
    log_level: str = "DEBUG",
) -> Callable[[F], F]:
    """Decorator do thoi gian thuc thi mot ham.

    Su dung:
        @measure_latency("lstm_inference")
        def encode(tick_seq, bar_seqs):
            ...

        @measure_latency("model_a", histogram=INFERENCE_LATENCY_MS)
        def predict(x):
            ...

    Args:
        metric_name: Ten de hien thi trong log
        histogram: Prometheus Histogram de record
        log_level: Muc log (DEBUG, INFO, WARNING)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                elapsed_s = elapsed_ms / 1000

                # Log
                log_fn = getattr(logger, log_level.lower(), logger.debug)
                if elapsed_ms > 100:
                    log_fn(f"[TIMING] {metric_name} = {elapsed_ms:.1f}ms ({elapsed_s:.3f}s)")
                else:
                    log_fn(f"[TIMING] {metric_name} = {elapsed_ms:.2f}ms")

                # Prometheus histogram
                if histogram is not None and _HAS_PROMETHEUS:
                    histogram.observe(elapsed_ms)

        return wrapper  # type: ignore[return-value]
    return decorator


class LatencyTracker:
    """Context manager de do thoi gian thuc thi mot khoi code.

    Su dung:
        tracker = LatencyTracker("build_usv")
        with tracker:
            # ... code ...
        print(f"Elapsed: {tracker.elapsed_ms:.2f}ms")
    """

    def __init__(
        self,
        name: str = "unnamed",
        histogram: "Histogram | None" = None,
        log: bool = True,
    ):
        self.name = name
        self.histogram = histogram
        self.log = log
        self.start: float = 0
        self.elapsed_ms: float = 0

    def __enter__(self) -> "LatencyTracker":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000
        if self.log:
            if self.elapsed_ms > 100:
                logger.warning(f"[TIMING] {self.name} = {self.elapsed_ms:.1f}ms")
            else:
                logger.debug(f"[TIMING] {self.name} = {self.elapsed_ms:.2f}ms")
        if self.histogram is not None and _HAS_PROMETHEUS:
            self.histogram.observe(self.elapsed_ms)


class MovingAverageLatency:
    """Tinh trung binh di chuyen cua do tre.

    Su dung:
        tracker = MovingAverageLatency(window=100)
        tracker.add(5.2)
        tracker.add(3.8)
        print(tracker.avg)  # Trung binh
        print(tracker.p95) # Phan tram 95
    """

    def __init__(self, window: int = 100):
        self.window = window
        self._values: list[float] = []

    def add(self, value: float) -> None:
        self._values.append(value)
        if len(self._values) > self.window:
            self._values.pop(0)

    @property
    def avg(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    @property
    def p95(self) -> float:
        if not self._values:
            return 0.0
        sorted_vals = sorted(self._values)
        idx = int(len(sorted_vals) * 0.95)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    @property
    def p99(self) -> float:
        if not self._values:
            return 0.0
        sorted_vals = sorted(self._values)
        idx = int(len(sorted_vals) * 0.99)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    @property
    def max(self) -> float:
        return max(self._values) if self._values else 0.0
