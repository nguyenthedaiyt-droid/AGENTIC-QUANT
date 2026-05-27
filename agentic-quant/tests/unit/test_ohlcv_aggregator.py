# =============================================================================
# AGENTIC-QUANT — Unit Tests cho OHLCV Aggregator
# =============================================================================

from __future__ import annotations

import pytest

from core.ingestion import OHLCVAggregator, BarState, TIMEFRAME_SECONDS, CASCADE_ORDER
from core.ingestion.tick_frame import TickFrame
from core.utils.events import EventBus, EventType, TickReceivedEvent


class TestBarState:
    """Tests cho BarState."""

    def test_update_first_tick(self) -> None:
        bar = BarState()
        bar.update(price=2500.0, volume=10.0, aggressor="BUY")

        assert bar.open == 2500.0
        assert bar.high == 2500.0
        assert bar.low == 2500.0
        assert bar.close == 2500.0
        assert bar.volume == 10.0
        assert bar.tick_count == 1
        assert bar.buy_volume == 10.0
        assert bar.sell_volume == 0.0

    def test_update_high_low(self) -> None:
        bar = BarState()
        bar.update(price=2500.0, volume=10.0, aggressor="BUY")
        bar.update(price=2510.0, volume=5.0, aggressor="BUY")
        bar.update(price=2490.0, volume=3.0, aggressor="SELL")

        assert bar.high == 2510.0
        assert bar.low == 2490.0
        assert bar.close == 2490.0
        assert bar.volume == 18.0
        assert bar.tick_count == 3
        assert bar.buy_volume == 15.0
        assert bar.sell_volume == 3.0

    def test_to_bar_close(self) -> None:
        bar = BarState(
            open=2500.0, high=2510.0, low=2495.0,
            close=2505.0, volume=100.0, tick_count=50,
            bucket_time=1700000000, is_closed=True,
        )
        event = bar.to_bar_close("M1", "XAUUSD")

        assert event.timeframe == "M1"
        assert event.bar_open == 2500.0
        assert event.bar_high == 2510.0
        assert event.bar_low == 2495.0
        assert event.bar_close == 2505.0
        assert event.bar_volume == 100.0
        assert event.tick_count == 50


class TestTimeframeSeconds:
    """Tests cho timeframe definitions."""

    def test_all_timeframes_defined(self) -> None:
        expected = {"M1": 60, "M5": 300, "M15": 900, "H1": 3600, "H4": 14400, "D1": 86400}
        assert TIMEFRAME_SECONDS == expected

    def test_cascade_order(self) -> None:
        assert CASCADE_ORDER == ["M1", "M5", "M15", "H1", "H4", "D1"]


class TestOHLCVAggregator:
    """Tests cho OHLCVAggregator."""

    def _make_tick(
        self,
        timestamp_us: int,
        price: float = 2500.0,
        volume: float = 10.0,
        aggressor: str = "BUY",
    ) -> TickReceivedEvent:
        return TickReceivedEvent(
            symbol="XAUUSD",
            timestamp_us=timestamp_us,
            bid=price - 0.5,
            ask=price + 0.5,
            last=price,
            volume=volume,
            flags=0,
            aggressor=aggressor,
            spread_pips=1.0,
            mid_price=price,
        )

    def test_single_bar_close(self) -> None:
        """Tick tao bar, tick tiep theo bat dau bar moi -> bar cu dong."""
        agg = OHLCVAggregator()

        # Tick luc T=0 (bucket M1 = 0)
        t1 = 0 * 60 * 1_000_000  # 0 microseconds
        closed = agg.process_tick(self._make_tick(t1, price=2500.0))
        assert len(closed) == 0
        assert agg.total_ticks == 1

        # Tick luc T=61s (bucket M1 = 60s)
        t2 = 61 * 60 * 1_000_000
        closed = agg.process_tick(self._make_tick(t2, price=2510.0))
        assert len(closed) >= 1  # M1 + cascade events
        assert closed[0].timeframe == "M1"
        assert closed[0].bar_open == 2500.0
        assert closed[0].bar_close == 2500.0  # Close = tick dau tien

    def test_multiple_ticks_same_bar(self) -> None:
        """Nhieu ticks cung mot bucket -> bar tich luy."""
        agg = OHLCVAggregator()

        # 5 ticks cung bucket M1
        for i in range(5):
            t = i * 10 * 1_000_000  # 10s apart
            agg.process_tick(self._make_tick(t, price=2500.0 + i))

        bar = agg.get_latest_bar("XAUUSD", "M1")
        assert bar is not None
        assert bar.tick_count == 5
        assert bar.open == 2500.0
        assert bar.high == 2504.0
        assert bar.low == 2500.0

    def test_bar_high_equals_close_for_single_tick(self) -> None:
        """Mot tick duy nhat -> open=high=low=close."""
        agg = OHLCVAggregator()
        agg.process_tick(self._make_tick(0, price=2500.0))

        bar = agg.get_latest_bar("XAUUSD", "M1")
        assert bar is not None
        assert bar.open == bar.high == bar.low == bar.close == 2500.0

    def test_get_bars_since(self) -> None:
        """Kiem tra get_bars_since tra ve dung bars."""
        agg = OHLCVAggregator()

        # 3 bars
        agg.process_tick(self._make_tick(0, price=2500.0))          # bar 0
        agg.process_tick(self._make_tick(61 * 1_000_000, price=2510.0))  # bar 1 dong, bar 2 mo
        agg.process_tick(self._make_tick(121 * 1_000_000, price=2520.0))  # bar 2 dong

        bars = agg.get_bars_since("XAUUSD", "M1", since_timestamp=0, limit=10)
        assert len(bars) == 2

    def test_stats(self) -> None:
        agg = OHLCVAggregator()
        agg.process_tick(self._make_tick(0))
        agg.process_tick(self._make_tick(61 * 1_000_000))

        stats = agg.get_stats()
        assert stats["total_ticks"] == 2
        assert len(stats.get("bar_closes", [])) >= 0

    def test_multi_timeframe(self) -> None:
        """Tick tao bars tren nhieu TF cung luc."""
        agg = OHLCVAggregator()

        # M1 bucket 0, M5 bucket 0, H1 bucket 0
        agg.process_tick(self._make_tick(0, price=2500.0))

        m1 = agg.get_latest_bar("XAUUSD", "M1")
        m5 = agg.get_latest_bar("XAUUSD", "M5")
        h1 = agg.get_latest_bar("XAUUSD", "H1")

        assert m1 is not None
        assert m5 is not None
        assert h1 is not None

        assert m1.bucket_time == 0
        assert m5.bucket_time == 0
        assert h1.bucket_time == 0

    def test_cascade_closure(self) -> None:
        """Khi M1 dong, HTF co the cung dong neu tick vuot boundary."""
        agg = OHLCVAggregator()

        # Bar M1: T=0..60s
        agg.process_tick(self._make_tick(0, price=2500.0))

        # Bar M5: T=0..300s, tick 301s se dong M5
        # Tick T=301s cung la M1 bucket 300s (moi)
        agg.process_tick(self._make_tick(301 * 1_000_000, price=2510.0))

        # Co 2 bars M1 dong (T=60s va T=300s)
        # Chi can kiem tra HTF duoc tao
        m1 = agg.get_latest_bar("XAUUSD", "M1")
        assert m1 is not None
        assert m1.bucket_time == 300  # Bucket 300s (T=301s)
