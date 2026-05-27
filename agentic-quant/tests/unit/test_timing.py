# =============================================================================
# AGENTIC-QUANT — Unit Tests: Timing Utilities
# =============================================================================

from __future__ import annotations

import pytest
import time

from core.utils.timing import measure_latency, LatencyTracker, MovingAverageLatency


class TestMeasureLatency:
    """Test decorator measure_latency."""

    def test_basic_timing(self):
        @measure_latency("test_op")
        def fast_operation():
            return 42

        result = fast_operation()
        assert result == 42

    def test_slow_operation_logged(self):
        @measure_latency("slow_op")
        def slow_operation():
            time.sleep(0.05)  # 50ms
            return "done"

        result = slow_operation()
        assert result == "done"

    def test_nested_timing(self):
        @measure_latency("outer")
        def outer():
            @measure_latency("inner")
            def inner():
                time.sleep(0.01)
                return 1

            return inner() + 1

        assert outer() == 2


class TestLatencyTracker:
    """Test LatencyTracker context manager."""

    def test_basic_tracking(self):
        tracker = LatencyTracker("test", log=False)
        with tracker:
            time.sleep(0.01)
        assert tracker.elapsed_ms >= 10
        assert tracker.elapsed_ms < 100

    def test_with_histogram(self):
        from unittest.mock import MagicMock

        mock_histogram = MagicMock()
        tracker = LatencyTracker("test_with_hist", histogram=mock_histogram, log=False)
        with tracker:
            time.sleep(0.01)
        mock_histogram.observe.assert_called_once()


class TestMovingAverageLatency:
    """Test MovingAverageLatency."""

    def test_empty_state(self):
        ma = MovingAverageLatency(window=10)
        assert ma.avg == 0.0
        assert ma.p95 == 0.0
        assert ma.p99 == 0.0

    def test_single_value(self):
        ma = MovingAverageLatency(window=10)
        ma.add(5.0)
        assert ma.avg == 5.0
        assert ma.p95 == 5.0

    def test_multiple_values(self):
        ma = MovingAverageLatency(window=10)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            ma.add(v)
        assert 2.9 < ma.avg < 3.1

    def test_window_enforced(self):
        ma = MovingAverageLatency(window=3)
        for v in range(1, 6):
            ma.add(float(v))
        assert len(ma._values) == 3
        assert ma.avg == 4.0  # (3+4+5)/3

    def test_percentiles(self):
        ma = MovingAverageLatency(window=100)
        for i in range(1, 101):
            ma.add(float(i))
        assert ma.avg == 50.5
        # p95 of 1..100 is approximately 95 (due to binning, may be 95 or 96)
        assert 94 < ma.p95 < 97
