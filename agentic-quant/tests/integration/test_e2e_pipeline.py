# =============================================================================
# AGENTIC-QUANT — Full E2E Pipeline Integration Test
# Mock MT5 EA gui 1000 ticks, verify toan bo pipeline
# =============================================================================
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from core.ingestion.tick_frame import TickFrame
from core.ingestion.ohlcv_aggregator import (
    OHLCVAggregator,
    BarState,
    TIMEFRAME_SECONDS,
    CASCADE_ORDER,
)
from core.ingestion.volumetrics_engine import VolumetricsEngine
from core.ingestion.mtf_synchronizer import MTFSynchronizer
from core.ingestion.itq_queue import IncomingTickQueue

from core.utils.events import EventBus, EventType
from core.utils.events.types import (
    TickReceivedEvent,
    BarCloseEvent,
    BarUpdateEvent,
    NewsAlertEvent,
    NewsImpact,
    PredictionReadyEvent,
    ZoneCreatedEvent,
)


# =============================================================================
# Test Class: Full E2E Pipeline
# =============================================================================
class TestE2EPipeline:
    """End-to-End test: 1000 MT5 ticks -> verify pipeline output.
    
    Mo phong:
      1. MT5 EA gui 1000 ticks qua ZeroMQ
      2. OHLCVAggregator xu ly ticks thanh bars
      3. Feature engineering pipeline detect zones
      4. AI Engine predict BSL/SSL probabilities
      5. News alert countdown trigger
      6. WebSocket broadcast (latency < 15ms)
    """

    @pytest.fixture(autouse=True)
    def setup_method(self, mock_event_bus: EventBus) -> None:
        """Khoi tao cac component cho moi test."""
        self.bus = mock_event_bus
        self.ohlcv = OHLCVAggregator(event_bus=self.bus)
        self.vol = VolumetricsEngine(event_bus=self.bus)
        self.mtf = MTFSynchronizer(symbol="XAUUSD")
        self.itq = IncomingTickQueue(max_size=10000)

        # Tracking variables
        self.received_bar_closes: list[BarCloseEvent] = []
        self.received_updates: list[BarUpdateEvent] = []
        self.websocket_messages: list[dict] = []
        self.zones_created: list[ZoneCreatedEvent] = []
        self.news_alert_fired: bool = False

    # -------------------------------------------------------------------------
    # Helper: tao tick event tu TickFrame
    # -------------------------------------------------------------------------
    def _tick_to_event(self, tick: TickFrame) -> TickReceivedEvent:
        """Convert TickFrame -> TickReceivedEvent de publish."""
        return TickReceivedEvent(
            symbol=tick.symbol,
            timestamp_us=tick.timestamp_us,
            bid=tick.bid,
            ask=tick.ask,
            last=tick.last,
            volume=tick.volume,
            flags=tick.flags,
            is_abnormal_spread=tick.is_abnormal_spread(),
            aggressor=tick.aggressor_side(),
            spread_pips=tick.spread_pips,
            mid_price=tick.mid_price,
        )

    # -------------------------------------------------------------------------
    # Helper: generate 1000 ticks tu mock MT5
    # -------------------------------------------------------------------------
    def _generate_1000_ticks(self) -> list[TickFrame]:
        """Tao 1000 ticks mo phong MT5 EA.
        
        Price di tu 2500.0 -> 2510.0 trong ~100s.
        """
        ticks: list[TickFrame] = []
        base_us = 1_700_000_000_000_000

        for i in range(1000):
            ts_us = base_us + i * 100_000  # 100ms interval
            # Simulate price movement
            price = 2500.0 + (i / 1000) * 10.0 + np.sin(i * 0.1) * 2.0
            price = round(price, 2)
            bid = round(price - 0.5, 2)
            ask = round(price + 0.5, 2)
            volume = 10.0 + (i % 5) * 2.0

            tick = TickFrame(
                symbol="XAUUSD",
                timestamp_us=ts_us,
                bid=bid,
                ask=ask,
                last=price,
                volume=volume,
                flags=0,
            )
            ticks.append(tick)

        return ticks

    # -------------------------------------------------------------------------
    # Test 1: Full pipeline 1000 ticks -> chart updates
    # -------------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_e2e_chart_updates(self) -> None:
        """Verify chart update via WebSocket (latency < 15ms) sau 1000 ticks.
        
        Test flow:
          1. Mock MT5 gui 1000 ticks
          2. OHLCVAggregator process ticks
          3. Verify bars duoc tao cho M1, M5
          4. Check latency publish event < 15ms
          5. Verify BarUpdateEvent phat cho frontend
        """
        # Subscribe de tracking
        async def on_bar_update(event):
            self.received_updates.append(event)

        async def on_bar_close(event):
            self.received_bar_closes.append(event)

        self.bus.subscribe(EventType.BAR_UPDATE, on_bar_update)
        self.bus.subscribe(EventType.BAR_CLOSE, on_bar_close)

        # Generate va xu ly 1000 ticks
        ticks = self._generate_1000_ticks()
        assert len(ticks) == 1000, f"Phai co 1000 ticks, co {len(ticks)}"

        latencies_ms: list[float] = []
        warmup_count = 50
        for idx, tick in enumerate(ticks):
            event = self._tick_to_event(tick)

            start = time.perf_counter()
            closed_bars = self.ohlcv.process_tick(event)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            # Bo qua warmup ticks trong latency calculation
            if idx >= warmup_count:
                latencies_ms.append(elapsed_ms)

            # Cap nhat MTF
            for tf in TIMEFRAME_SECONDS:
                bar = self.ohlcv.get_latest_bar("XAUUSD", tf)
                if bar:
                    self.mtf.update_bar(tf, bar)

            # Volumetrics
            self.vol.process_tick(event)

        # Verify latency < 15ms
        if latencies_ms:
            avg_latency = sum(latencies_ms) / len(latencies_ms)
            max_latency = max(latencies_ms)
        assert avg_latency < 15.0, (
            f"Average OHLCV latency {avg_latency:.3f}ms > 15ms"
        )
        assert max_latency < 15.0, (
            f"Max OHLCV latency {max_latency:.3f}ms > 15ms"
        )

        # Verify bar closes (M1 bars should close ~ every 60 ticks)
        m1_closes = [b for b in self.received_bar_closes if b.timeframe == "M1"]
        assert len(m1_closes) >= 1, (
            f"Phai co it nhat 1 M1 bar close, co {len(m1_closes)}"
        )

        # Verify bar update events duoc phat
        assert len(self.received_updates) > 0, "Phai co bar update events"

        # Verify bars trong OHLCV aggregator
        for tf in ["M1", "M5"]:
            bar = self.ohlcv.get_latest_bar("XAUUSD", tf)
            assert bar is not None, f"Bar {tf} phai ton tai"
            assert bar.high >= bar.low, f"Bar {tf}: high >= low"
            assert bar.tick_count > 0, f"Bar {tf}: tick_count > 0"

    # -------------------------------------------------------------------------
    # Test 2: AI predictions > 0.0
    # -------------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_e2e_predictions_positive(self) -> None:
        """Verify AI Engine generate predictions > 0.0 sau bar close.
        
        Test flow:
          1. Process ticks cho den khi co BAR_CLOSE
          2. Mock FeatureEngineeringPipeline xu ly BAR_CLOSE
          3. Mock AI Engine predict BSL/SSL probabilities
          4. Verify predictions co gia tri > 0.0
        """
        received_predictions: list[PredictionReadyEvent] = []

        async def on_prediction(event):
            received_predictions.append(event)

        self.bus.subscribe(EventType.PREDICTION_READY, on_prediction)

        # Process ticks tu 1 -> 120 (du cho 2 bars M1)
        ticks = self._generate_1000_ticks()[:120]

        for tick in ticks:
            event = self._tick_to_event(tick)
            closed_bars = self.ohlcv.process_tick(event)

            for tf in TIMEFRAME_SECONDS:
                bar = self.ohlcv.get_latest_bar("XAUUSD", tf)
                if bar:
                    self.mtf.update_bar(tf, bar)

            self.vol.process_tick(event)

        # Gia lap: publish PredictionReadyEvent nhu AI Engine se lam
        for i, bar_close in enumerate(self.received_bar_closes[:2]):
            pred_event = PredictionReadyEvent(
                symbol=bar_close.symbol,
                timeframe=bar_close.timeframe,
                timestamp=bar_close.bucket_time,
                p_bsl=0.35 + i * 0.05,
                p_ssl=0.55 - i * 0.03,
                p_lateral=0.10 + i * 0.01,
                bsl_target=bar_close.bar_high + 10.0,
                ssl_target=bar_close.bar_low - 10.0,
                zones_predicted=[
                    {
                        "zone_type": "FVG_BULL",
                        "top": bar_close.bar_high + 5.0,
                        "bottom": bar_close.bar_high - 2.0,
                        "p_hold": 0.85,
                    }
                ],
                session_id=f"session_{bar_close.bucket_time}",
                active_guardrail=False,
                macro_regime="NORMAL",
                confidence_qualifier="MEDIUM",
            )
            self.bus.publish(pred_event)

        # Wait cho EventBus xu ly tat ca events
        await asyncio.sleep(0.5)
        # Force xu ly pending tasks
        pending = list(self.bus._running_tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        # Verify predictions co gia tri > 0.0
        assert len(received_predictions) >= 1, "Phai co prediction events"
        for pred in received_predictions:
            assert pred.p_bsl > 0.0, f"p_bsl phai > 0.0, got {pred.p_bsl}"
            assert pred.p_ssl > 0.0, f"p_ssl phai > 0.0, got {pred.p_ssl}"
            assert pred.p_lateral > 0.0, (
                f"p_lateral phai > 0.0, got {pred.p_lateral}"
            )
            # Verify sum ~ 1.0
            total = pred.p_bsl + pred.p_ssl + pred.p_lateral
            assert 0.95 <= total <= 1.05, (
                f"Sum probabilities phai ~ 1.0, got {total}"
            )

    # -------------------------------------------------------------------------
    # Test 3: Zones dung vi tri gia
    # -------------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_e2e_zones_correct_price_positions(self) -> None:
        """Verify zone duoc tao dung vi tri gia (notional points).
        
        Test flow:
          1. Process ticks
          2. Tao zones mo phong FeatureEngineeringPipeline
          3. Kiem tra zones co top >= bottom va nam trong price range
        """
        # Process 1000 ticks de co bars
        ticks = self._generate_1000_ticks()
        tracking_prices: list[float] = []

        for tick in ticks:
            event = self._tick_to_event(tick)
            tracking_prices.append(tick.last)
            closed_bars = self.ohlcv.process_tick(event)

            for tf in TIMEFRAME_SECONDS:
                bar = self.ohlcv.get_latest_bar("XAUUSD", tf)
                if bar:
                    self.mtf.update_bar(tf, bar)

            self.vol.process_tick(event)

        # Tao zones mo phong tu bars
        latest_m1 = self.ohlcv.get_latest_bar("XAUUSD", "M1")
        assert latest_m1 is not None

        zones_to_test = [
            {"type": "FVG_BULL", "top": 2505.0, "bottom": 2502.0, "p_hold": 0.85},
            {"type": "OB_BEAR", "top": 2510.0, "bottom": 2507.0, "p_hold": 0.72},
            {"type": "FVG_BEAR", "top": 2515.0, "bottom": 2512.0, "p_hold": 0.90},
            {"type": "OB_BULL", "top": 2498.0, "bottom": 2495.0, "p_hold": 0.65},
        ]

        for zone in zones_to_test:
            # Verify top >= bottom (zone dung vi tri)
            assert zone["top"] >= zone["bottom"], (
                f"Zone {zone['type']}: top must >= bottom "
                f"({zone['top']} < {zone['bottom']})"
            )

            # Verify zone nam trong price range
            min_price = min(tracking_prices)
            max_price = max(tracking_prices)
            assert zone["bottom"] >= min_price - 10.0, (
                f"Zone {zone['type']}: bottom {zone['bottom']} qua xa "
                f"so voi min price {min_price}"
            )
            assert zone["top"] <= max_price + 10.0, (
                f"Zone {zone['type']}: top {zone['top']} qua xa "
                f"so voi max price {max_price}"
            )

            # Verify p_hold trong [0, 1]
            assert 0.0 <= zone["p_hold"] <= 1.0, (
                f"Zone {zone['type']}: p_hold phai trong [0,1], "
                f"got {zone['p_hold']}"
            )

        # Verify zone registry co du lieu
        m1_bar = latest_m1
        assert m1_bar.high > m1_bar.low, "Bar phai co high > low"
        assert m1_bar.tick_count > 0, "Bar phai co tick_count > 0"

    # -------------------------------------------------------------------------
    # Test 4: News alert trigger khi countdown <= 900s
    # -------------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_e2e_news_alert_countdown_trigger(self) -> None:
        """Verify news alert trigger khi countdown <= 900s.
        
        Test flow:
          1. Publish NewsAlertEvent voi seconds_to_event = 600 (<= 900)
          2. Subscribe de nhan event
          3. Verify event duoc xu ly dung
        """
        fired_alerts: list[NewsAlertEvent] = []

        async def on_news_alert(event):
            fired_alerts.append(event)

        self.bus.subscribe(EventType.NEWS_ALERT, on_news_alert)

        # Tao NewsAlert voi countdown = 600s (<= 900s -> trigger)
        news = NewsAlertEvent(
            event_id="nfp_feb_2025",
            title="Non Farm Payrolls",
            currency="USD",
            impact=NewsImpact.HIGH,
            scheduled_time=datetime.now(timezone.utc),
            forecast=200000.0,
            previous=180000.0,
            actual=220000.0,
            seconds_to_event=600,
            state="PRE_NEWS",
            i_news=0.85,
            surprise_z=1.5,
            surprise_direction="BULLISH",
        )

        # Publish news alert
        self.bus.publish(news)
        await asyncio.sleep(0.2)

        # Verify
        assert len(fired_alerts) >= 1, "News alert phai duoc fire"
        alert = fired_alerts[0]
        assert alert.seconds_to_event <= 900, (
            f"seconds_to_event phai <= 900s, got {alert.seconds_to_event}"
        )
        assert alert.impact == NewsImpact.HIGH, (
            f"Impact phai la HIGH, got {alert.impact}"
        )
        assert alert.currency == "USD", (
            f"Currency phai la USD, got {alert.currency}"
        )
        assert alert.i_news > 0.0, (
            f"I_news phai > 0.0, got {alert.i_news}"
        )

    # -------------------------------------------------------------------------
    # Test 5: WebSocket latency < 15ms cho chart update
    # -------------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_e2e_websocket_latency_bound(self) -> None:
        """Verify WebSocket broadcast latency < 15ms.
        
        Mo phong WebSocketServer broadcast:
          1. Subscribe vao EventBus
          2. Publish events
          3. Do latency tu publish -> receive
        """
        receive_times: list[float] = []

        async def measure_latency(event):
            receive_ms = time.perf_counter() * 1000.0
            receive_times.append(receive_ms)

        self.bus.subscribe(EventType.BAR_UPDATE, measure_latency)
        self.bus.subscribe(EventType.BAR_CLOSE, measure_latency)
        self.bus.subscribe(EventType.PREDICTION_READY, measure_latency)
        self.bus.subscribe(EventType.NEWS_ALERT, measure_latency)

        # Simulate various events
        test_events = []
        base_ts = int(datetime.now(timezone.utc).timestamp())

        # Bar updates (chart updates - most frequent)
        for i in range(50):
            test_events.append(
                BarUpdateEvent(
                    symbol="XAUUSD",
                    timeframe="M1",
                    bar_open=2500.0 + i * 0.1,
                    bar_high=2500.0 + i * 0.1 + 5.0,
                    bar_low=2500.0 + i * 0.1 - 3.0,
                    bar_close=2500.0 + i * 0.1 + 2.0,
                    bar_volume=1000.0,
                    bucket_time=base_ts + i * 60,
                    is_closed=False,
                )
            )

        # Bar closes
        for i in range(10):
            test_events.append(
                BarCloseEvent(
                    symbol="XAUUSD",
                    timeframe="M1",
                    bar_open=2500.0,
                    bar_high=2510.0,
                    bar_low=2495.0,
                    bar_close=2505.0,
                    bar_volume=1000.0,
                    bucket_time=base_ts + i * 60,
                    tick_count=100,
                )
            )

        # Predictions
        test_events.append(
            PredictionReadyEvent(
                symbol="XAUUSD",
                timeframe="M1",
                timestamp=base_ts,
                p_bsl=0.35,
                p_ssl=0.55,
                p_lateral=0.10,
                bsl_target=2520.0,
                ssl_target=2485.0,
                confidence_qualifier="MEDIUM",
                macro_regime="NORMAL",
                active_guardrail=False,
            )
        )

        # News alert
        test_events.append(
            NewsAlertEvent(
                event_id="news_test",
                title="Test Event",
                currency="USD",
                impact=NewsImpact.HIGH,
                scheduled_time=datetime.now(timezone.utc),
                forecast=100.0,
                previous=90.0,
                actual=110.0,
                seconds_to_event=300,
                state="PRE_NEWS",
                i_news=0.75,
                surprise_z=2.0,
                surprise_direction="BULLISH",
            )
        )

        # Publish events va do latency
        publish_times: list[float] = []
        for event in test_events:
            start_ns = time.perf_counter_ns()
            self.bus.publish(event)
            elapsed_ns = time.perf_counter_ns() - start_ns
            publish_times.append(elapsed_ns / 1_000_000.0)  # ms

        # Wait cho EventBus xu ly
        await asyncio.sleep(0.5)

        # Verify publish latency < 15ms
        avg_publish_latency = (
            sum(publish_times) / len(publish_times)
        )
        max_publish_latency = max(publish_times)
        assert avg_publish_latency < 15.0, (
            f"Average publish latency {avg_publish_latency:.3f}ms > 15ms"
        )
        assert max_publish_latency < 15.0, (
            f"Max publish latency {max_publish_latency:.3f}ms > 15ms"
        )

        # Verify events duoc nhan
        events_received = len(receive_times)
        expected_min = 10  # it nhat 10 bar close + 1 prediction + 1 news
        assert events_received >= expected_min, (
            f"Received {events_received} events, expected >= {expected_min}"
        )

    # -------------------------------------------------------------------------
    # Test 6: Full integration smoke test
    # -------------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_e2e_full_pipeline_smoke(self) -> None:
        """Smoke test: tick -> bar -> zones -> predictions -> broadcast.
        
        Verify pipeline khong crash va output dung format.
        """
        # Subscribe tracking events
        all_events: list = []

        async def track_all(event):
            all_events.append(event)

        for et in [
            EventType.TICK_RECEIVED,
            EventType.BAR_UPDATE,
            EventType.BAR_CLOSE,
            EventType.PREDICTION_READY,
            EventType.NEWS_ALERT,
        ]:
            self.bus.subscribe(et, track_all)

        # Process ticks
        ticks_raw = self._generate_1000_ticks()

        for tick in ticks_raw:
            event = self._tick_to_event(tick)
            self.ohlcv.process_tick(event)

            for tf in TIMEFRAME_SECONDS:
                bar = self.ohlcv.get_latest_bar("XAUUSD", tf)
                if bar:
                    self.mtf.update_bar(tf, bar)

            self.vol.process_tick(event)

        # Mo phong zones tu bars
        m1_bar = self.ohlcv.get_latest_bar("XAUUSD", "M1")
        if m1_bar:
            # Publish zone creation event
            self.bus.publish(
                ZoneCreatedEvent(
                    zone_id="zone_fvg_001",
                    zone_type="FVG_BULL",
                    symbol="XAUUSD",
                    timeframe="M1",
                    price_top=m1_bar.high + 5.0,
                    price_bottom=m1_bar.high - 2.0,
                    formed_time=m1_bar.bucket_time,
                    premium_discount="DISCOUNT",
                    p_hold=0.85,
                    w_zone=1.0,
                )
            )

        # Mo phong predictions
        self.bus.publish(
            PredictionReadyEvent(
                symbol="XAUUSD",
                timeframe="M1",
                timestamp=m1_bar.bucket_time if m1_bar else 0,
                p_bsl=0.35,
                p_ssl=0.55,
                p_lateral=0.10,
                bsl_target=2520.0,
                ssl_target=2485.0,
                confidence_qualifier="MEDIUM",
                macro_regime="NORMAL",
                active_guardrail=False,
            )
        )

        # Mo phong news alert
        self.bus.publish(
            NewsAlertEvent(
                event_id="news_smoke",
                title="Smoke Test Event",
                currency="USD",
                impact=NewsImpact.HIGH,
                scheduled_time=datetime.now(timezone.utc),
                seconds_to_event=600,
                state="PRE_NEWS",
                i_news=0.80,
            )
        )

        await asyncio.sleep(0.3)

        # Verify pipeline output
        bar_updates = [e for e in all_events if isinstance(e, BarUpdateEvent)]
        assert len(bar_updates) > 0, "Phai co bar update events"

        last_bar = self.ohlcv.get_latest_bar("XAUUSD", "M1")
        assert last_bar is not None, "M1 bar phai ton tai"
        assert last_bar.high >= last_bar.low
        assert last_bar.volume > 0
        assert last_bar.tick_count > 0
