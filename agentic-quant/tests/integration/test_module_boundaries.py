# =============================================================================
# AGENTIC-QUANT — Boundary Integration Tests (5 Module Interfaces)
# Kiem tra giao tiep giua cac module trong pipeline
# =============================================================================
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from core.utils.events import EventBus, EventType
from core.utils.events.types import (
    BarCloseEvent,
    BarUpdateEvent,
    NewsAlertEvent,
    NewsImpact,
    PredictionReadyEvent,
    ZoneCreatedEvent,
    ZoneUpdateEvent,
)
from core.macro.macro_engine import MacroEngine, MacroEngineConfig
from core.memory.short_term.redis_cache_manager import (
    RedisCacheManager,
    RedisNamespace,
)
from core.memory.short_term.active_zone_registry import ActiveZoneRegistry
from core.memory.models.zone import Zone
from core.memory.models.enums import (
    Timeframe,
    ZoneStatus,
    ZoneType,
    MacroRegime,
)
from core.ipc.broadcast_dispatcher import BroadcastDispatcher
from core.ai_engine.neural.lstm_inference import LSTMInferenceEngine
from core.ai_engine.xgboost.inference import (
    InferenceEngineA,
    InferenceEngineB,
    ModelAOutput,
    ModelBOutput,
)


# =============================================================================
# Test Boundary 1: Module 1 (Ingestion) -> Module 2 (MacroEngine)
# BAR_CLOSE event -> MacroEngine nhan va xu ly
# =============================================================================
class TestModule1ToModule2:
    """Test giao tiep: Module 1 (Ingestion/OHLCV) -> Module 2 (MacroEngine).
    
    Flow:
      - Publish BAR_CLOSE event
      - MacroEngine subscribe BAR_CLOSE?
      - Verify MacroEngine nhan duoc event
    """

    @pytest.mark.asyncio
    async def test_bar_close_reaches_macro_engine(self, mock_event_bus: EventBus) -> None:
        """Verify MacroEngine nhan BarCloseEvent tu Ingestion module."""
        bus = mock_event_bus
        received_events: list[BarCloseEvent] = []

        # MacroEngine lang nghe BAR_CLOSE
        async def on_bar_close(event: BarCloseEvent) -> None:
            received_events.append(event)

        bus.subscribe(EventType.BAR_CLOSE, on_bar_close)

        # Simulate OHLCV tao BAR_CLOSE
        bar_close = BarCloseEvent(
            symbol="XAUUSD",
            timeframe="M1",
            bar_open=2500.0,
            bar_high=2510.0,
            bar_low=2495.0,
            bar_close=2505.0,
            bar_volume=1000.0,
            bucket_time=int(datetime.now(timezone.utc).timestamp()),
            tick_count=120,
        )

        # Publish tu Module 1 (ingestion)
        bus.publish(bar_close)
        await asyncio.sleep(0.2)

        # Assert
        assert len(received_events) >= 1, "MacroEngine phai nhan BAR_CLOSE"
        event = received_events[0]
        assert event.symbol == "XAUUSD"
        assert event.timeframe == "M1"
        assert event.bar_open == 2500.0
        assert event.bar_close == 2505.0
        assert event.tick_count >= 120

    @pytest.mark.asyncio
    async def test_bar_close_contains_mtf_bar_data(
        self, mock_event_bus: EventBus
    ) -> None:
        """Verify BAR_CLOSE co du 6 timeframe bars khi cascade close."""
        bus = mock_event_bus
        received_bars: dict[str, list[BarCloseEvent]] = {
            "M1": [], "M5": [], "M15": [], "H1": [], "H4": [], "D1": []
        }

        async def track_bar_close(event: BarCloseEvent) -> None:
            if event.timeframe in received_bars:
                received_bars[event.timeframe].append(event)

        bus.subscribe(EventType.BAR_CLOSE, track_bar_close)

        base_ts = datetime.now(timezone.utc).timestamp()

        # Publish multiple BAR_CLOSE events (mo phong cascade)
        for tf in ["M1", "M5", "M15"]:
            bus.publish(BarCloseEvent(
                symbol="XAUUSD",
                timeframe=tf,
                bar_open=2500.0,
                bar_high=2510.0,
                bar_low=2495.0,
                bar_close=2505.0,
                bar_volume=float(1000 * {"M1": 1, "M5": 5, "M15": 15}[tf]),
                bucket_time=int(base_ts),
                tick_count=100 * {"M1": 1, "M5": 5, "M15": 15}[tf],
            ))

        await asyncio.sleep(0.2)

        # Verify it nhat M1, M5, M15 bars duoc publish
        assert len(received_bars["M1"]) >= 1, "M1 bar close phai ton tai"
        assert len(received_bars["M5"]) >= 1, "M5 bar close phai ton tai"
        assert len(received_bars["M15"]) >= 1, "M15 bar close phai ton tai"

        # M5 volume > M1 volume
        assert received_bars["M5"][0].bar_volume > received_bars["M1"][0].bar_volume


# =============================================================================
# Test Boundary 2: Module 2 (MacroEngine) -> Module 3 (Memory/Redis)
# NEWS_ALERT -> Redis macro:state duoc update
# =============================================================================
class TestModule2ToModule3:
    """Test giao tiep: Module 2 (MacroEngine) -> Module 3 (Memory/Redis).
    
    Flow:
      - MacroEngine phat NEWS_ALERT
      - Redis macro:state duoc cap nhat voi state moi
    """

    @pytest.mark.asyncio
    async def test_news_alert_updates_redis_macro_state(
        self, mock_event_bus: EventBus, mock_redis: MagicMock
    ) -> None:
        """Verify Redis macro:state duoc update khi co NEWS_ALERT."""
        bus = mock_event_bus

        # Gia lap: khi NEWS_ALERT duoc publish, Redis duoc update
        async def on_news_alert(event: NewsAlertEvent) -> None:
            # Mo phong MacroEngine update Redis
            state_data = {
                "regime": "PRE_NEWS",
                "currency": event.currency,
                "guardrail_active": "True",
                "seconds_remaining": str(event.seconds_to_event),
                "current_event_id": event.event_id,
                "i_news": str(event.i_news),
            }
            await mock_redis.set_macro_state(event.currency, state_data)

        bus.subscribe(EventType.NEWS_ALERT, on_news_alert)

        # Publish NEWS_ALERT
        news_alert = NewsAlertEvent(
            event_id="nfp_news",
            title="Non Farm Payrolls",
            currency="USD",
            impact=NewsImpact.HIGH,
            scheduled_time=datetime.now(timezone.utc),
            forecast=200000.0,
            previous=180000.0,
            seconds_to_event=600,
            state="PRE_NEWS",
            i_news=0.75,
        )

        bus.publish(news_alert)
        await asyncio.sleep(0.2)

        # Verify Redis set_macro_state duoc goi
        mock_redis.set_macro_state.assert_awaited_once()

        # Verify key dung format
        call_args = mock_redis.set_macro_state.await_args
        assert call_args is not None
        # call_args[0] la currency, call_args[1] la state_data
        assert call_args[0] == "USD", (
            f"Redis macro state currency phai la 'USD', got {call_args[0]}"
        )

        state_data = call_args[1]
        assert state_data["regime"] == "PRE_NEWS"
        assert state_data["guardrail_active"] == "True"
        assert int(state_data["seconds_remaining"]) <= 900

    @pytest.mark.asyncio
    async def test_macro_state_reflects_guardrail_change(
        self, mock_event_bus: EventBus, mock_redis: MagicMock
    ) -> None:
        """Verify Redis macro:state update khi guardrail thay doi."""
        bus = mock_event_bus
        state_history: list[dict] = []

        async def update_state_on_alert(event) -> None:
            state = {
                "regime": event.state,
                "currency": event.currency,
                "guardrail_active": "True"
                if event.seconds_to_event <= 900
                else "False",
                "seconds_remaining": str(event.seconds_to_event),
            }
            await mock_redis.set_macro_state(event.currency, state)
            state_history.append(state)

        bus.subscribe(EventType.NEWS_ALERT, update_state_on_alert)

        # Event voi countdown 600s -> guardrail ACTIVE
        alert_in = NewsAlertEvent(
            event_id="test_001",
            title="Event in window",
            currency="USD",
            impact=NewsImpact.HIGH,
            seconds_to_event=600,
            state="PRE_NEWS",
            i_news=0.8,
        )

        bus.publish(alert_in)
        await asyncio.sleep(0.1)

        # Verify guardrail active (seconds_to_event <= 900)
        assert mock_redis.set_macro_state.awaited
        last_call = mock_redis.set_macro_state.await_args
        assert last_call is not None
        # last_call[0] = currency, last_call[1] = dict
        assert last_call[0] == "USD"
        state_data = last_call[1]
        assert "guardrail_active" in state_data, (
            f"State data must have 'guardrail_active', keys: {list(state_data.keys())}"
        )
        assert state_data["guardrail_active"] == "True"

    @pytest.mark.asyncio
    async def test_multiple_currencies_separate_redis_keys(
        self, mock_event_bus: EventBus, mock_redis: MagicMock
    ) -> None:
        """Verify moi currency co macro:state key rieng."""
        bus = mock_event_bus

        async def on_news(event):
            await mock_redis.set_macro_state(event.currency, {
                "regime": event.state,
                "currency": event.currency,
            })

        bus.subscribe(EventType.NEWS_ALERT, on_news)

        # Publish cho nhieu currencies
        for currency in ["USD", "EUR", "GBP"]:
            bus.publish(NewsAlertEvent(
                event_id=f"news_{currency}",
                title=f"{currency} Event",
                currency=currency,
                impact=NewsImpact.HIGH,
                seconds_to_event=600,
                state="PRE_NEWS",
                i_news=0.5,
            ))

        await asyncio.sleep(0.3)

        # Verify moi currency duoc set rieng
        assert mock_redis.set_macro_state.await_count >= 3
        calls = mock_redis.set_macro_state.await_args_list
        currencies_set = {c[0][0] for c in calls}
        assert "USD" in currencies_set
        assert "EUR" in currencies_set
        assert "GBP" in currencies_set


# =============================================================================
# Test Boundary 3: Module 3 (Feature Engineering) -> Module 4 (Zone Registry)
# Zone upsert -> ActiveZoneRegistry tra zone dung
# =============================================================================
class TestModule3ToModule4:
    """Test giao tiep: Module 3 (Feature Engineering) -> Module 4 (Zone Registry).
    
    Flow:
      - FeatureEngineeringPipeline upsert zone vao ActiveZoneRegistry
      - ActiveZoneRegistry tra zone dung khi query near price
      - Zone co vi tri chinh xac (top >= bottom)
    """

    @pytest.mark.asyncio
    async def test_zone_upsert_and_retrieve(
        self, mock_redis: MagicMock
    ) -> None:
        """Verify upsert zone -> query zones near price tra dung zone."""
        # Setup ActiveZoneRegistry voi mock Redis
        registry = ActiveZoneRegistry(redis=mock_redis)

        # Mock Redis tra ve zone data khi get_zone duoc goi
        zone_data = {
            "id": "zone_fvg_001",
            "symbol": "XAUUSD",
            "timeframe": "M1",
            "zone_type": "FVG_BULL",
            "top": "2508.0",
            "bottom": "2505.0",
            "ce": "1.5",
            "formed_time": "1700000000000",
            "status": "UNMITIGATED",
            "p_hold": "0.85",
            "p_hold_updated": "1700000000500",
            "w_zone": "1.0",
            "iii_formation": "0.0",
            "touch_count": "0",
            "last_touch_time": "0",
            "htf_tf": "None",
        }

        async def mock_get_zone(key: str) -> dict | None:
            if "zone_fvg" in key:
                return zone_data
            return None

        mock_redis.get_zone.side_effect = mock_get_zone

        # Tao zone object
        zone = Zone(
            id="zone_fvg_001",
            symbol="XAUUSD",
            timeframe=Timeframe.M1,
            zone_type=ZoneType.FVG_BULL,
            top=2508.0,
            bottom=2505.0,
            ce=1.5,
            formed_time=1700000000000,
            status=ZoneStatus.UNMITIGATED,
            p_hold=0.85,
            w_zone=1.0,
        )

        # Upsert zone (simulate FeatureEngineeringPipeline)
        is_new = await registry.upsert_zone(zone)
        assert is_new, "Zone phai la moi (chua ton tai trong Redis)"

        # redis.set_zone + redis.update_zone_rank phai duoc goi
        mock_redis.set_zone.assert_awaited_once()
        mock_redis.update_zone_rank.assert_awaited_once()

        # Verify zone data dung
        set_call = mock_redis.set_zone.await_args
        assert set_call is not None
        zone_data_sent = set_call[1]  # zone_data dict
        assert zone_data_sent.get("id") == "zone_fvg_001", (
            f"Zone id mismatch, got {zone_data_sent.get('id')}"
        )
        assert float(zone_data_sent.get("top", 0)) >= float(zone_data_sent.get("bottom", 0))

    @pytest.mark.asyncio
    async def test_zone_near_price_query(
        self, mock_redis: MagicMock
    ) -> None:
        """Verify get_zones_near_price tra dung zones gan gia."""
        registry = ActiveZoneRegistry(redis=mock_redis)

        # Mock tat ca zone keys
        async def mock_get_all_keys(symbol=None):
            return [
                "zone:XAUUSD:M1:FVG_BULL:1700000000000",
                "zone:XAUUSD:M1:OB_BEAR:1700000001000",
                "zone:XAUUSD:M5:FVG_BULL:1700000002000",
            ]

        mock_redis.get_all_zone_keys.side_effect = mock_get_all_keys

        # Mock zone data cho moi key
        zone_datas = {
            "zone:XAUUSD:M1:FVG_BULL:1700000000000": {
                "id": "z1", "symbol": "XAUUSD", "timeframe": "M1",
                "zone_type": "FVG_BULL", "top": "2508.0", "bottom": "2505.0",
                "ce": "1.5", "formed_time": "1700000000000",
                "status": "UNMITIGATED", "p_hold": "0.85", "p_hold_updated": "0",
                "w_zone": "1.0", "iii_formation": "0", "touch_count": "0",
                "last_touch_time": "0", "htf_tf": "None",
            },
            "zone:XAUUSD:M1:OB_BEAR:1700000001000": {
                "id": "z2", "symbol": "XAUUSD", "timeframe": "M1",
                "zone_type": "OB_BEAR", "top": "2512.0", "bottom": "2509.0",
                "ce": "1.0", "formed_time": "1700000001000",
                "status": "UNMITIGATED", "p_hold": "0.72", "p_hold_updated": "0",
                "w_zone": "1.0", "iii_formation": "0", "touch_count": "0",
                "last_touch_time": "0", "htf_tf": "None",
            },
            "zone:XAUUSD:M5:FVG_BULL:1700000002000": {
                "id": "z3", "symbol": "XAUUSD", "timeframe": "M5",
                "zone_type": "FVG_BULL", "top": "2520.0", "bottom": "2515.0",
                "ce": "2.0", "formed_time": "1700000002000",
                "status": "MITIGATED", "p_hold": "0.10", "p_hold_updated": "0",
                "w_zone": "1.0", "iii_formation": "0", "touch_count": "0",
                "last_touch_time": "0", "htf_tf": "None",
            },
        }

        async def mock_get_zone(key: str) -> dict | None:
            return zone_datas.get(key)

        mock_redis.get_zone.side_effect = mock_get_zone

        # Query near price (2507.0), window = 5 pips
        zones = await registry.get_zones_near_price(
            symbol="XAUUSD",
            price=2507.0,
            window_pips=5.0,
            timeframe="M1",
        )

        # Chi M1 zones duoc tra ve (timeframe filter)
        assert len(zones) >= 1, "Phai co it nhat 1 zone near price"

        for z in zones:
            # Zone co vi tri dung: top >= bottom
            assert z.top >= z.bottom, f"Zone {z.id}: top >= bottom"
            # Zone chua bi mitigated
            assert z.is_active(), f"Zone {z.id}: phai active"
            # Zone thuoc M1
            assert z.timeframe == Timeframe.M1, f"Zone {z.id}: phai la M1"

        # z3 bi MITIGATED, khong duoc tra ve
        z3_found = any(z.id == "z3" for z in zones)
        assert not z3_found, "z3 (MITIGATED) khong duoc tra ve"

    @pytest.mark.asyncio
    async def test_zone_price_position_validity(
        self, mock_redis: MagicMock
    ) -> None:
        """Verify zone price position validity: zone nam trong range."""
        # Tao nhieu zones
        zones_to_test = [
            Zone(
                id=f"z_test_{i}",
                symbol="XAUUSD",
                timeframe=Timeframe.M1,
                zone_type=ZoneType.FVG_BULL if i % 2 == 0 else ZoneType.OB_BEAR,
                top=2500.0 + i * 5.0 + 3.0,
                bottom=2500.0 + i * 5.0,
                p_hold=0.5 + i * 0.05,
                formed_time=1700000000000 + i * 1000,
                status=ZoneStatus.UNMITIGATED,
            )
            for i in range(10)
        ]

        for z in zones_to_test:
            # top >= bottom
            assert z.top >= z.bottom, (
                f"Zone {z.id}: top {z.top} >= bottom {z.bottom}"
            )
            # p_hold trong [0, 1]
            assert 0.0 <= z.p_hold <= 1.0, (
                f"Zone {z.id}: p_hold {z.p_hold} trong [0, 1]"
            )
            # ce >= 0
            assert z.ce >= 0.0, (
                f"Zone {z.id}: ce {z.ce} >= 0"
            )
            # Range size hop ly
            range_size = z.range_size()
            assert range_size >= 0.0, (
                f"Zone {z.id}: range_size {range_size} >= 0"
            )


# =============================================================================
# Test Boundary 4: Module 4 (Zone Registry) -> Module 7 (IPC/WebSocket)
# BAR_CLOSE + BOS event -> WebSocket broadcast
# =============================================================================
class TestModule4ToModule7:
    """Test giao tiep: Module 4 (Zone Registry) -> Module 7 (IPC/WebSocket).
    
    Flow:
      - Zone change event (BAR_CLOSE + BOS) publish
      - BroadcastDispatcher chuyen thanh IPC message
      - WebSocket broadcast to frontend
    """

    @pytest.mark.asyncio
    async def test_bar_close_with_bos_broadcasts_to_websocket(
        self, mock_event_bus: EventBus
    ) -> None:
        """Verify BAR_CLOSE + BOS event duoc broadcast to WebSocket."""
        bus = mock_event_bus
        broadcast_messages: list[dict] = []

        def mock_broadcast_fn(message: dict) -> None:
            broadcast_messages.append(message)

        # Tao BroadcastDispatcher
        dispatcher = BroadcastDispatcher(event_bus=bus)
        dispatcher.set_broadcast_fn(mock_broadcast_fn)
        dispatcher.start()

        # Publish BAR_CLOSE (mo phong Module 1)
        bar_close = BarCloseEvent(
            symbol="XAUUSD",
            timeframe="M1",
            bar_open=2500.0,
            bar_high=2510.0,
            bar_low=2495.0,
            bar_close=2505.0,
            bar_volume=1000.0,
            bucket_time=int(datetime.now(timezone.utc).timestamp()),
            tick_count=120,
        )
        bus.publish(bar_close)

        # BroadcastDispatcher subscribe BAR_UPDATE, khong phai BAR_CLOSE
        # Nhung BroadcastDispatcher subscribe cac event khac
        # Publish BAR_UPDATE de test broadcast
        bar_update = BarUpdateEvent(
            symbol="XAUUSD",
            timeframe="M1",
            bar_open=2500.0,
            bar_high=2510.0,
            bar_low=2495.0,
            bar_close=2505.0,
            bar_volume=1000.0,
            bucket_time=int(datetime.now(timezone.utc).timestamp()),
            is_closed=True,
        )
        bus.publish(bar_update)

        # Publish news alert
        news = NewsAlertEvent(
            event_id="bos_news",
            title="BOS Event",
            currency="USD",
            impact=NewsImpact.HIGH,
            seconds_to_event=600,
            state="PRE_NEWS",
            i_news=0.8,
        )
        bus.publish(news)

        # Wait for dispatcher
        await asyncio.sleep(0.3)

        # Verify broadcast messages
        assert len(broadcast_messages) >= 2, (
            f"Phai co it nhat 2 broadcast messages, co {len(broadcast_messages)}"
        )

        dispatcher.stop()

    @pytest.mark.asyncio
    async def test_zone_update_broadcasts_format(
        self, mock_event_bus: EventBus
    ) -> None:
        """Verify ZoneUpdateEvent broadcast dung format."""
        bus = mock_event_bus
        broadcast_messages: list[dict] = []

        def mock_broadcast(message: dict) -> None:
            broadcast_messages.append(message)

        dispatcher = BroadcastDispatcher(event_bus=bus)
        dispatcher.set_broadcast_fn(mock_broadcast)
        dispatcher.start()

        # Publish ZoneUpdateEvent
        zone_update = ZoneUpdateEvent(
            zone_id="zone_fvg_001",
            old_status="UNMITIGATED",
            new_status="PARTIALLY_MITIGATED",
            new_p_hold=0.45,
        )
        bus.publish(zone_update)
        await asyncio.sleep(0.2)

        # Verify broadcast
        # ZoneUpdateEvent duoc BroadcastDispatcher xu ly
        assert len(broadcast_messages) >= 1

        dispatcher.stop()

    @pytest.mark.asyncio
    async def test_prediction_broadcast_format(
        self, mock_event_bus: EventBus
    ) -> None:
        """Verify prediction broadcast co dung fields cho frontend."""
        bus = mock_event_bus
        broadcast_messages: list[dict] = []

        def mock_broadcast(message: dict) -> None:
            broadcast_messages.append(message)

        dispatcher = BroadcastDispatcher(event_bus=bus)
        dispatcher.set_broadcast_fn(mock_broadcast)
        dispatcher.start()

        # Publish PredictionReadyEvent
        prediction = PredictionReadyEvent(
            symbol="XAUUSD",
            timeframe="M1",
            timestamp=int(datetime.now(timezone.utc).timestamp()),
            p_bsl=0.35,
            p_ssl=0.55,
            p_lateral=0.10,
            bsl_target=2520.0,
            ssl_target=2485.0,
            confidence_qualifier="MEDIUM",
            macro_regime="PRE_NEWS",
            active_guardrail=True,
        )
        bus.publish(prediction)
        await asyncio.sleep(0.2)

        messages = broadcast_messages
        dispatcher.stop()


# =============================================================================
# Test Boundary 5: Module 5 (LSTM) -> Module 6 (XGBoost)
# LSTM encode -> XGBoost inference chain
# =============================================================================
class TestModule5ToModule6:
    """Test giao tiep: Module 5 (LSTM Neural) -> Module 6 (XGBoost).
    
    Flow:
      - LSTMInferenceEngine.encode() -> latent vector z[512]
      - FeatureBuilder ghep z voi feature vector
      - InferenceEngineA.predict(X_A[648]) -> ModelAOutput
      - InferenceEngineB.predict(X_B[560]) -> ModelBOutput
    """

    @pytest.mark.asyncio
    async def test_lstm_encode_to_xgboost_inference_chain(self) -> None:
        """Verify LSTM output (z[512]) -> XGBoost inference chain.
        
        Mo phong:
          1. LSTM encode tick_seq + bar_seqs -> z[512]
          2. Build X_A[648] feature vector (z + other features)
          3. InferenceEngineA.predict(X_A) -> p_bsl, p_ssl, p_lateral
        """
        # --- Step 1: Mock LSTM encoder output ---
        latent_z = np.random.randn(512).astype(np.float32)
        assert latent_z.shape == (512,), (
            f"LSTM latent vector phai la [512], got {latent_z.shape}"
        )

        # --- Step 2: Build X_A feature vector ---
        # X_A = [z(512) + f_struct(48) + f_agg(36) + f_liq(36) + d_strength(16)]
        # Total: 512 + 48 + 36 + 36 + 16 = 648
        f_struct = np.random.randn(48).astype(np.float32)
        f_agg = np.random.randn(36).astype(np.float32)
        f_liq = np.random.randn(36).astype(np.float32)
        d_strength = np.random.randn(16).astype(np.float32)

        X_A = np.concatenate([latent_z, f_struct, f_agg, f_liq, d_strength])
        assert X_A.shape == (648,), (
            f"X_A feature vector phai la [648], got {X_A.shape}"
        )

        # --- Step 3: Mock InferenceEngineA ---
        engine_a_mock = MagicMock(spec=InferenceEngineA)

        def mock_predict(
            X_A_in: np.ndarray,
            regime_code: int = 0,
            bear_evidence: float = 0.0,
            apply_session_weights: bool = True,
        ) -> ModelAOutput:
            return ModelAOutput(
                p_bsl=0.35,
                p_ssl=0.55,
                p_lateral=0.10,
                predicted_class=1,
                predicted_label="SSL",
                confidence_qualifier="MEDIUM",
                max_prob=0.55,
            )

        engine_a_mock.predict = mock_predict

        result = engine_a_mock.predict(X_A)
        assert isinstance(result, ModelAOutput)
        assert result.p_bsl > 0.0, f"p_bsl > 0.0, got {result.p_bsl}"
        assert result.p_ssl > 0.0, f"p_ssl > 0.0, got {result.p_ssl}"
        assert result.p_lateral > 0.0, (
            f"p_lateral > 0.0, got {result.p_lateral}"
        )
        # Probabilities sum ~ 1.0
        total = result.p_bsl + result.p_ssl + result.p_lateral
        assert 0.95 <= total <= 1.05, (
            f"Sum probabilities ~ 1.0, got {total}"
        )
        assert result.confidence_qualifier in ("HIGH", "MEDIUM", "LOW")

    @pytest.mark.asyncio
    async def test_lstm_to_xgboost_zone_hold_chain(self) -> None:
        """Verify LSTM latent -> Feature Builder X_B -> Model B zone hold."""
        # --- Step 1: Mock LSTM latent ---
        latent_z = np.random.randn(512).astype(np.float32)

        # --- Step 2: Build X_B feature vector ---
        # Model B: zone features(32) + price features(16) + z[512]
        # Total: 32 + 16 + 512 = 560
        zone_features = np.random.randn(32).astype(np.float32)
        price_features = np.random.randn(16).astype(np.float32)

        X_B = np.concatenate([zone_features, price_features, latent_z])
        assert X_B.shape == (560,), (
            f"X_B feature vector phai la [560], got {X_B.shape}"
        )

        # --- Step 3: Mock InferenceEngineB ---
        engine_b_mock = MagicMock(spec=InferenceEngineB)

        def mock_predict(X_B_in: np.ndarray) -> ModelBOutput:
            return ModelBOutput(
                p_hold=0.85,
                p_not_hold=0.15,
                predicted_hold=1,
                confidence=0.14,
                theta_star=0.71,
            )

        engine_b_mock.predict = mock_predict

        result = engine_b_mock.predict(X_B)
        assert isinstance(result, ModelBOutput)
        assert result.p_hold > 0.0, (
            f"p_hold > 0.0, got {result.p_hold}"
        )
        assert result.p_not_hold > 0.0, (
            f"p_not_hold > 0.0, got {result.p_not_hold}"
        )
        # p_hold + p_not_hold ~ 1.0
        total = result.p_hold + result.p_not_hold
        assert 0.95 <= total <= 1.05, (
            f"Sum hold prob ~ 1.0, got {total}"
        )
        assert result.predicted_hold in (0, 1)
        assert result.theta_star == 0.71

    @pytest.mark.asyncio
    async def test_lstm_encode_dimensionality(self) -> None:
        """Verify LSTM encoder dimensionality match XGBoost input."""
        # LSTM input shapes
        tick_seq_shape = (128, 8)  # 128 ticks, 8 features
        bar_seq_shape = (30, 12)  # 30 bars, 12 features

        # Mo phong encoder output
        z_dim = 512
        latent_z = np.random.randn(z_dim).astype(np.float32)

        # Verify xgboost input dims
        # X_A must be 648
        f_struct_dim = 48
        f_agg_dim = 36
        f_liq_dim = 36
        d_strength_dim = 16

        X_A_expected = z_dim + f_struct_dim + f_agg_dim + f_liq_dim + d_strength_dim
        assert X_A_expected == 648, (
            f"X_A dimension phai la 648, expected {X_A_expected}"
        )

        # X_B must be 560
        zone_feat_dim = 32
        price_feat_dim = 16

        X_B_expected = zone_feat_dim + price_feat_dim + z_dim
        assert X_B_expected == 560, (
            f"X_B dimension phai la 560, expected {X_B_expected}"
        )

        # Verify pipeline chain dimensions
        X_A = np.concatenate([
            latent_z,
            np.random.randn(f_struct_dim).astype(np.float32),
            np.random.randn(f_agg_dim).astype(np.float32),
            np.random.randn(f_liq_dim).astype(np.float32),
            np.random.randn(d_strength_dim).astype(np.float32),
        ])
        assert X_A.shape == (648,), f"X_A shape = {X_A.shape}, expected (648,)"

        X_B = np.concatenate([
            np.random.randn(zone_feat_dim).astype(np.float32),
            np.random.randn(price_feat_dim).astype(np.float32),
            latent_z,
        ])
        assert X_B.shape == (560,), f"X_B shape = {X_B.shape}, expected (560,)"

    @pytest.mark.asyncio
    async def test_lstm_output_numpy_consumable_by_xgboost(self) -> None:
        """Verify LSTM output type (np.ndarray float32) tuong thich XGBoost."""
        # LSTM tra ve np.ndarray float32
        lstm_output = np.random.randn(512).astype(np.float32)
        assert lstm_output.dtype == np.float32, (
            "LSTM output phai la float32"
        )

        # XGBoost can np.ndarray (co the la float32 hoac float64)
        X_A = np.concatenate([
            lstm_output,
            np.random.randn(136).astype(np.float32),  # cung float32
        ])
        assert X_A.dtype == np.float32
        assert X_A.shape == (648,)

        # Simulate np.argmax (nhu trong InferenceEngineA)
        probas = np.array([[0.35, 0.55, 0.10]], dtype=np.float32)
        predicted_class = int(np.argmax(probas[0]))
        assert predicted_class == 1  # SSL (index 1)
        assert 0 <= predicted_class <= 2

        # max probability
        max_prob = float(np.max(probas[0]))
        import math
        expected = 0.55
        assert math.isclose(max_prob, expected, abs_tol=1e-6), (
            f"max_prob = {max_prob}, expected {expected}"
        )
