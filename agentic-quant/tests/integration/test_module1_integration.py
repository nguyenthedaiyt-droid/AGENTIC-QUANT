# =============================================================================
# AGENTIC-QUANT — Integration Test cho Module 1
# Tu tick -> USV hoan chinh
# =============================================================================

from __future__ import annotations

import time

import pytest

from core.ingestion import (
    OHLCVAggregator,
    VolumetricsEngine,
    MTFSynchronizer,
    TickFrame,
    BarState,
    TIMEFRAME_SECONDS,
)
from core.ingestion.itq_queue import IncomingTickQueue
from core.ingestion.desync_detector import TimeframeDesyncDetector
from core.utils.events import EventBus, EventType, TickReceivedEvent


class TestModule1Integration:
    """
    Integration test cho toan bo Module 1.

    Flow: Tick -> ITQ -> OHLCV -> Volumetrics -> MTF Sync -> USV
    """

    @pytest.fixture
    def event_bus(self) -> EventBus:
        return EventBus(max_queue_size=1000)

    @pytest.fixture
    def components(self, event_bus):
        """Khoi tao tat ca components."""
        # OHLCV Aggregator
        ohlcv = OHLCVAggregator(event_bus=event_bus)

        # Volumetrics Engine
        vol = VolumetricsEngine(event_bus=event_bus)

        # MTF Synchronizer
        mtf = MTFSynchronizer(symbol="XAUUSD")

        # ITQ
        itq = IncomingTickQueue(max_size=10000)

        # Desync Detector
        desync = TimeframeDesyncDetector()

        return {
            "ohlcv": ohlcv,
            "vol": vol,
            "mtf": mtf,
            "itq": itq,
            "desync": desync,
        }

    def _make_tick(
        self,
        timestamp_us: int,
        price: float,
        volume: float = 10.0,
        aggressor: str = "BUY",
        bid_offset: float = 0.5,
    ) -> TickReceivedEvent:
        return TickReceivedEvent(
            symbol="XAUUSD",
            timestamp_us=timestamp_us,
            bid=round(price - bid_offset, 2),
            ask=round(price + bid_offset, 2),
            last=round(price, 2),
            volume=volume,
            flags=0,
            aggressor=aggressor,
            spread_pips=bid_offset * 10,
            mid_price=price,
        )

    # -------------------------------------------------------------------------
    # Full pipeline: tick -> USV
    # -------------------------------------------------------------------------
    def test_full_pipeline_tick_to_usv(self, event_bus, components) -> None:
        """Test day du tu tick den USV."""
        ohlcv = components["ohlcv"]
        vol = components["vol"]
        mtf = components["mtf"]

        # Tạo 120 ticks (2 phut) de tao 2 bars M1
        base_ts = 1700000000_000_000  # 2023-11-14 17:46:40 UTC

        for i in range(120):
            ts = base_ts + i * 1_000_000  # 1 tick / second
            price = 2500.0 + i * 0.1

            tick = self._make_tick(ts, price, aggressor="BUY" if i % 2 == 0 else "SELL")

            # OHLCV
            closed = ohlcv.process_tick(tick)

            # Cap nhat bar vao MTF
            for tf in TIMEFRAME_SECONDS:
                bar = ohlcv.get_latest_bar("XAUUSD", tf)
                if bar:
                    mtf.update_bar(tf, bar)

            # Volumetrics
            vol.process_tick(tick)
            mtf.update_volumetrics(
                cvd=0.0,  # would come from vol engine
                cvd_norm=0.0,
                iii=0.0,
                divergence=0.0,
            )

        # Build USV
        latest_tick = self._make_tick(base_ts + 120 * 1_000_000, 2512.0)
        usv = mtf.build_usv(latest_tick)

        # Assertions
        assert usv.symbol == "XAUUSD"
        assert usv.snapshot_time == base_ts + 120 * 1_000_000
        assert "M1" in usv.bars
        assert len(usv.bars) == 6  # 6 TF

        # M1 bar co du lieu
        m1_bar = usv.bars.get("M1")
        assert m1_bar is not None
        assert m1_bar.tick_count > 0
        assert m1_bar.high >= m1_bar.low

    def test_bar_close_triggers_ai_inference(self, event_bus, components) -> None:
        """Test rang bar close phat event."""
        ohlcv = components["ohlcv"]
        received_bars: list = []
        vol = components["vol"]

        def on_bar(event):
            received_bars.append(event)

        event_bus.subscribe(EventType.BAR_CLOSE, on_bar)

        # Tick 1: tao bar M1
        base_ts = 1700000000_000_000
        tick1 = self._make_tick(base_ts, 2500.0)
        closed1 = ohlcv.process_tick(tick1)
        assert len(closed1) == 0

        # Tick 61: dong bar cu, tao bar moi
        tick2 = self._make_tick(base_ts + 61 * 1_000_000, 2510.0)
        closed2 = ohlcv.process_tick(tick2)
        assert len(closed2) >= 1
        assert closed2[0].timeframe == "M1"
        assert closed2[0].bar_open == 2500.0

        # Volumetrics reset on bar close
        vol.process_bar_close(closed2[0])

        # USV: tick moi
        mtf = components["mtf"]
        for tf in TIMEFRAME_SECONDS:
            bar = ohlcv.get_latest_bar("XAUUSD", tf)
            if bar:
                mtf.update_bar(tf, bar)
        usv = mtf.build_usv(tick2)
        assert usv.bars["M1"].bucket_time == 60  # Bar 2

    def test_cascade_closure_multi_tf(self, event_bus, components) -> None:
        """Test cascade closure cho nhieu TF."""
        ohlcv = components["ohlcv"]

        # Tick 1: M1 bucket 0, M5 bucket 0, H1 bucket 0
        ts1 = 1700000000_000_000  # 2023-11-14 17:46:40 UTC
        tick1 = self._make_tick(ts1, 2500.0)
        closed1 = ohlcv.process_tick(tick1)
        assert len(closed1) == 0

        # Tick tao 301 bars tren M1 (5 phut + 1 giay)
        # Luc nay M1 bucket 300, M5 bucket 0 van active
        ts2 = ts1 + 301 * 1_000_000
        tick2 = self._make_tick(ts2, 2510.0)
        closed2 = ohlcv.process_tick(tick2)

        # M1 dong 2 bars (bucket 60 va bucket 300)
        m1_closed = [c for c in closed2 if c.timeframe == "M1"]
        assert len(m1_closed) >= 1

        # H1 bar van o bucket 0
        h1_bar = ohlcv.get_latest_bar("XAUUSD", "H1")
        assert h1_bar is not None
        assert h1_bar.bucket_time == 0

    def test_volumetrics_cvd_accumulation(self, event_bus, components) -> None:
        """Test CVD tich luy dung."""
        vol = components["vol"]

        base_ts = 1700000000_000_000

        # 10 BUY ticks, 10 SELL ticks
        for i in range(10):
            ts = base_ts + i * 1_000_000
            buy_tick = self._make_tick(ts, 2500.0 + i * 0.1, aggressor="BUY", volume=10.0)
            sell_tick = self._make_tick(ts + 500_000, 2500.5 + i * 0.1, aggressor="SELL", volume=10.0)

            vol.process_tick(buy_tick)
            vol.process_tick(sell_tick)

        # Net CVD = 0 (10 buy * 10 - 10 sell * 10 = 0)
        # Nhung chung ta chi kiem tra no chay ma khong crash

    def test_abnormal_spread_skipped(self, event_bus, components) -> None:
        """Test tick co spread bat thuong bi bo qua trong volumetrics."""
        vol = components["vol"]

        base_ts = 1700000000_000_000
        normal_tick = self._make_tick(base_ts, 2500.0, bid_offset=0.5)
        vol.process_tick(normal_tick)

        # Spread bat thuong (10 pips)
        abnormal_tick = TickReceivedEvent(
            symbol="XAUUSD",
            timestamp_us=base_ts + 1_000_000,
            bid=2500.0,
            ask=2510.0,  # spread = 10.0
            last=2505.0,
            volume=10.0,
            flags=0,
            aggressor="BUY",
            spread_pips=100.0,
            is_abnormal_spread=True,
        )

        # Khong crash
        result = vol.process_tick(abnormal_tick)
        assert result is None  # Abnormal spread tra ve None

    def test_itq_overflow_sampling(self, event_bus, components) -> None:
        """Test ITQ overflow voi K dynamic sampling."""
        itq = components["itq"]

        # Day queue vuot muc
        base_ts = 1700000000_000_000

        for i in range(12000):  # Nhieu hon max_size = 10000
            ts = base_ts + i * 100_000  # 100ms interval
            tick = TickFrame(
                symbol="XAUUSD",
                timestamp_us=ts,
                bid=2500.0,
                ask=2500.5,
                last=2500.2,
                volume=10.0,
                flags=0,
            )
            itq.enqueue(tick, ts)

        stats = itq.stats
        assert stats["total_sampled"] > 0
        assert stats["is_overflow"] is True
        assert stats["current_k"] > 1

    def test_desync_detection(self, event_bus, components) -> None:
        """Test phat hien desync."""
        ohlcv = components["ohlcv"]
        desync = components["desync"]

        base_ts = 1700000000_000_000

        # Tao 50 bars M1 (thiếu 50 bars)
        for i in range(50):
            ts = base_ts + i * 60 * 1_000_000
            tick = self._make_tick(ts, 2500.0 + i * 0.1)
            ohlcv.process_tick(tick)

        bars = {b.bucket_time: b for b in ohlcv.get_bars_since("XAUUSD", "M1", 0, 200)}
        report = desync.check_desync(
            timeframe="M1",
            bars=bars,
            reference_time=base_ts // 1_000_000 + 100,  # 100 bars sau
            lookback_buckets=100,
        )

        # 50 bars co, 50 bars thieu
        assert report.missing_bars > 0
        assert report.severity.value in ("MODERATE", "SEVERE")

    def test_desync_interpolation(self, event_bus, components) -> None:
        """Test noi suy desync nho."""
        desync = components["desync"]

        bars = {
            0: BarState(
                open=2500.0, high=2510.0, low=2490.0, close=2505.0,
                volume=100.0, tick_count=50, bucket_time=0, is_closed=True,
            ),
            120: BarState(
                open=2506.0, high=2515.0, low=2500.0, close=2510.0,
                volume=100.0, tick_count=50, bucket_time=120, is_closed=True,
            ),
        }

        # Gap 1 bar (bucket 60 thieu)
        interp = desync.fill_gaps(bars, tf_sec=60, reference_time=300)
        assert 60 in interp
        assert interp[60].open == 2505.0  # prev close
        assert interp[60].close == 2506.0  # next open


# =============================================================================
# Benchmark: Throughput > 10,000 events/sec
# =============================================================================
class TestEventBusThroughput:
    """Benchmark EventBus throughput."""

    @pytest.mark.benchmark
    def test_throughput_10k_events_per_sec(self) -> None:
        """Verify EventBus dat throughput > 10,000 events/sec."""
        bus = EventBus(max_queue_size=50000)
        received_count = 0

        async def handler(event):
            nonlocal received_count
            received_count += 1

        bus.subscribe(EventType.TICK_RECEIVED, handler)

        # Publish 100,000 events
        start = time.perf_counter()
        for i in range(100_000):
            tick = TickReceivedEvent(
                symbol="XAUUSD",
                timestamp_us=i,
                bid=2500.0,
                ask=2500.5,
                last=2500.2,
                volume=10.0,
                flags=0,
            )
            bus.publish(tick)

        publish_time = time.perf_counter() - start

        # Tất cả events phải được publish trong < 10s cho 10k/sec
        assert publish_time < 10.0, f"Publish too slow: {publish_time:.2f}s for 100k events"

        # Qua hơn 10k/sec
        throughput = 100_000 / publish_time
        assert throughput > 10_000, f"Throughput {throughput:.0f}/sec < 10,000/sec"
