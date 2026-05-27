# =============================================================================
# AGENTIC-QUANT — Integration Tests Shared Fixtures
# Mock EventBus, Redis, ZMQ, sample data cho integration tests
# =============================================================================
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import numpy as np
import pytest
import pytest_asyncio

from core.utils.events import EventBus, EventType
from core.utils.events.types import (
    BarCloseEvent,
    BarUpdateEvent,
    NewsAlertEvent,
    NewsImpact,
    TickReceivedEvent,
)
from core.ingestion.tick_frame import TickFrame
from core.ingestion.ohlcv_aggregator import BarState, TIMEFRAME_SECONDS


# =============================================================================
# Mock EventBus Singleton
# =============================================================================
@pytest.fixture
def mock_event_bus() -> EventBus:
    """Tao EventBus singleton mock cho tests.
    
    Moi test nhan mot EventBus moi, khong anh huong global singleton.
    """
    return EventBus(max_queue_size=5000)


@pytest.fixture
async def mock_event_bus_reset(mock_event_bus: EventBus) -> AsyncGenerator[EventBus, None]:
    """EventBus voi cleanup sau test."""
    yield mock_event_bus
    await mock_event_bus.shutdown()


# =============================================================================
# Mock Redis Cache Manager
# =============================================================================
@pytest.fixture
def mock_redis() -> MagicMock:
    """Tao RedisCacheManager mock hoan chinh.
    
    Mock get/set zones, macro state, latent vectors.
    """
    redis = MagicMock()

    # Zone methods
    redis.set_zone = AsyncMock(return_value=True)
    redis.get_zone = AsyncMock(return_value=None)
    redis.delete_zone = AsyncMock(return_value=True)
    redis.get_all_zone_keys = AsyncMock(return_value=[])
    redis.update_zone_rank = AsyncMock(return_value=True)
    redis.get_top_zones_by_rank = AsyncMock(return_value=[])

    # Macro state
    redis.set_macro_state = AsyncMock(return_value=True)
    redis.get_macro_state = AsyncMock(return_value=None)

    # AI / Latent
    redis.set_ai_output = AsyncMock(return_value=True)
    redis.get_ai_output = AsyncMock(return_value=None)
    redis.set_latent_vector = AsyncMock(return_value=True)
    redis.get_latent_vector = AsyncMock(return_value=None)

    # Features
    redis.set_features = AsyncMock(return_value=True)
    redis.get_features = AsyncMock(return_value=None)

    # Debate
    redis.set_debate = AsyncMock(return_value=True)
    redis.get_debate = AsyncMock(return_value=None)

    # Macro events
    redis.add_macro_event = AsyncMock(return_value=True)
    redis.get_macro_events = AsyncMock(return_value=[])
    redis.clear_macro_events = AsyncMock(return_value=True)

    # Client mock
    redis.client = AsyncMock()
    redis.client.ping = AsyncMock(return_value=True)
    redis.client.hset = AsyncMock(return_value=1)
    redis.client.hgetall = AsyncMock(return_value={})
    redis.client.expire = AsyncMock(return_value=True)
    redis.client.delete = AsyncMock(return_value=1)
    redis.client.config_set = AsyncMock(return_value=True)
    redis.client.scan_iter = AsyncMock()
    redis.client.scan_iter.__aiter__.return_value = iter([])

    # Connection state
    redis.is_connected = PropertyMock(return_value=True)

    return redis


@pytest.fixture
def mock_redis_with_state(mock_redis: MagicMock) -> MagicMock:
    """Redis mock voi macro state san.
    
    Khi goi get_macro_state(\"USD\"), tra ve state du lieu gia.
    """
    async def _get_macro_state(currency: str) -> dict | None:
        if currency == "USD":
            return {
                "regime": "PRE_NEWS",
                "currency": "USD",
                "guardrail_active": "True",
                "dampening_factor": "0.5",
                "seconds_remaining": "600",
            }
        return None

    mock_redis.get_macro_state = AsyncMock(side_effect=_get_macro_state)
    return mock_redis


# =============================================================================
# Mock ZMQ Context / Socket
# =============================================================================
@pytest.fixture
def mock_zmq() -> MagicMock:
    """Tao ZeroMQ mock cho tick receiver tests."""
    zmq_mock = MagicMock()

    # Context
    ctx = MagicMock()
    ctx.socket = MagicMock(return_value=MagicMock())
    zmq_mock.Context = MagicMock(return_value=ctx)
    zmq_mock.asyncio = MagicMock()
    zmq_mock.asyncio.Context = MagicMock(return_value=ctx)
    zmq_mock.PULL = 1

    return zmq_mock


# =============================================================================
# Sample Data Fixtures
# =============================================================================
@pytest.fixture
def sample_ohlcv() -> dict[str, BarState]:
    """Tao sample OHLCV bars cho cac tests.
    
    Tra ve dict voi 6 timeframe bars (M1, M5, M15, H1, H4, D1).
    """
    base_time = int(datetime.now(timezone.utc).timestamp())

    def _make_bar(
        tf: str,
        bucket_offset: int = 0,
        open_p: float = 2500.0,
        high_p: float = 2510.0,
        low_p: float = 2495.0,
        close_p: float = 2505.0,
        vol: float = 1000.0,
        tick_ct: int = 100,
    ) -> BarState:
        tf_sec = TIMEFRAME_SECONDS.get(tf, 60)
        bucket = ((base_time // tf_sec) + bucket_offset) * tf_sec
        return BarState(
            open=open_p,
            high=high_p,
            low=low_p,
            close=close_p,
            volume=vol,
            tick_count=tick_ct,
            bucket_time=bucket,
            is_closed=True,
        )

    return {
        "M1": _make_bar("M1", 0),
        "M5": _make_bar("M5", 0, open_p=2498.0, high_p=2512.0, low_p=2490.0, close_p=2503.0, vol=5000.0),
        "M15": _make_bar("M15", 0, open_p=2502.0, high_p=2520.0, low_p=2485.0, close_p=2508.0, vol=15000.0),
        "H1": _make_bar("H1", 0, open_p=2510.0, high_p=2530.0, low_p=2480.0, close_p=2505.0, vol=60000.0),
        "H4": _make_bar("H4", -1, open_p=2490.0, high_p=2540.0, low_p=2470.0, close_p=2510.0, vol=200000.0),
        "D1": _make_bar("D1", -2, open_p=2480.0, high_p=2550.0, low_p=2460.0, close_p=2515.0, vol=1000000.0),
    }


@pytest.fixture
def sample_ticks() -> list[TickFrame]:
    """Tao 1000 sample ticks mock MT5 EA.
    
    Simulate 1000 ticks trong ~100s, price tu 2500 -> 2510.
    """
    ticks: list[TickFrame] = []
    base_us = int(datetime.now(timezone.utc).timestamp() * 1_000_000)

    for i in range(1000):
        ts_us = base_us + i * 100_000  # 100ms / tick
        price = 2500.0 + (i / 1000) * 10.0
        bid = round(price - 0.5, 2)
        ask = round(price + 0.5, 2)
        last = round(price, 2)

        tick = TickFrame(
            symbol="XAUUSD",
            timestamp_us=ts_us,
            bid=bid,
            ask=ask,
            last=last,
            volume=10.0,
            flags=0,
        )
        ticks.append(tick)

    return ticks


@pytest.fixture
def sample_bar_close_event() -> BarCloseEvent:
    """Tao sample BAR_CLOSE event cho tests."""
    return BarCloseEvent(
        symbol="XAUUSD",
        timeframe="M1",
        bar_open=2500.0,
        bar_high=2510.0,
        bar_low=2495.0,
        bar_close=2505.0,
        bar_volume=1000.0,
        bucket_time=int(datetime.now(timezone.utc).timestamp()),
        tick_count=100,
    )


@pytest.fixture
def sample_news_alert_event() -> NewsAlertEvent:
    """Tao sample NEWS_ALERT event (countdown <= 900s)."""
    return NewsAlertEvent(
        event_id="news_001",
        title="Non Farm Payrolls",
        currency="USD",
        impact=NewsImpact.HIGH,
        scheduled_time=datetime.now(timezone.utc),
        forecast=200000.0,
        previous=180000.0,
        actual=None,
        seconds_to_event=600,  # 10 phut <= 900s -> trigger
        state="PRE_NEWS",
        i_news=0.75,
        surprise_z=None,
        surprise_direction="NEUTRAL",
    )


@pytest.fixture
def sample_prediction_ready_data() -> dict[str, Any]:
    """Tao sample prediction data cho XGBoost output."""
    return {
        "symbol": "XAUUSD",
        "timeframe": "M1",
        "p_bsl": 0.35,
        "p_ssl": 0.55,
        "p_lateral": 0.10,
        "bsl_target": 2520.0,
        "ssl_target": 2485.0,
        "confidence_qualifier": "MEDIUM",
        "macro_regime": "NORMAL",
        "active_guardrail": False,
    }


# =============================================================================
# Helper: Mock LSTM encoder output
# =============================================================================
@pytest.fixture
def mock_lstm_output() -> np.ndarray:
    """Tao latent vector gia lap tu LSTM encoder [512]."""
    return np.random.randn(512).astype(np.float32)


# =============================================================================
# Helper: Mock FeatureVector
# =============================================================================
@pytest.fixture
def mock_feature_vector() -> np.ndarray:
    """Tao feature vector gia lap cho XGBoost [648]."""
    return np.random.randn(648).astype(np.float32)
