# =============================================================================
# AGENTIC-QUANT — Prometheus Metrics
# =============================================================================

from __future__ import annotations

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.registry import REGISTRY
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiohttp import web


# =============================================================================
# Counters
# =============================================================================

TICK_RECEIVED_TOTAL = Counter(
    "aq_tick_received_total",
    "Tong so tick nhan duoc",
    ["symbol", "aggressor_side"],
)

BAR_CLOSED_TOTAL = Counter(
    "aq_bar_closed_total",
    "Tong so bar da dong cua",
    ["symbol", "timeframe"],
)

PREDICTION_MADE_TOTAL = Counter(
    "aq_prediction_made_total",
    "Tong so prediction duoc tao",
    ["symbol", "model_version", "direction"],
)

DEBATE_TRIGGERED_TOTAL = Counter(
    "aq_debate_triggered_total",
    "Tong so debate duoc kich hoat",
    ["trigger"],
)

NEWS_EVENT_TOTAL = Counter(
    "aq_news_event_total",
    "Tong so su kien tin tuc",
    ["currency", "impact", "is_surprise"],
)

ZONE_CREATED_TOTAL = Counter(
    "aq_zone_created_total",
    "Tong so zone duoc tao",
    ["zone_type", "timeframe"],
)

ZONE_MITIGATED_TOTAL = Counter(
    "aq_zone_mitigated_total",
    "Tong so zone bi mitigated",
    ["zone_type", "mitigation_type"],
)

MODEL_DEGRADED_TOTAL = Counter(
    "aq_model_degraded_total",
    "So lan model bi danh gia la degraded",
    ["model_name"],
)


# =============================================================================
# Histograms
# =============================================================================

INFERENCE_LATENCY_MS = Histogram(
    "aq_inference_latency_ms",
    "Do tre inference (ms)",
    ["model_name", "component"],
    buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000),
)

IPC_LATENCY_MS = Histogram(
    "aq_ipc_latency_ms",
    "Do tre tu Python tinh toan den khi frontend nhan duoc (ms)",
    ["message_type"],
    buckets=(5, 10, 15, 20, 30, 50, 75, 100, 150, 200),
)

USV_BUILD_LATENCY_MS = Histogram(
    "aq_usv_build_latency_ms",
    "Do tre xay dung USV (ms)",
    ["component"],
    buckets=(1, 2, 5, 10, 20, 50),
)

LSTM_INFERENCE_MS = Histogram(
    "aq_lstm_inference_ms",
    "Do tre LSTM inference (ms)",
    buckets=(5, 10, 15, 20, 30, 50, 75, 100),
)

XGBOOST_INFERENCE_MS = Histogram(
    "aq_xgboost_inference_ms",
    "Do tre XGBoost inference (ms)",
    ["model_name"],
    buckets=(1, 2, 3, 5, 10, 20),
)

DEBATE_LATENCY_MS = Histogram(
    "aq_debate_latency_ms",
    "Do tre toan bo debate (ms)",
    buckets=(500, 1000, 2000, 3000, 5000, 8000, 10000),
)

TICK_PROCESSING_MS = Histogram(
    "aq_tick_processing_ms",
    "Do tre xu ly mot tick (ms)",
    buckets=(0.1, 0.5, 1, 2, 5, 10),
)

BAR_AGGREGATION_MS = Histogram(
    "aq_bar_aggregation_ms",
    "Do tre aggregation mot bar (ms)",
    ["timeframe"],
    buckets=(0.1, 0.5, 1, 2, 5, 10),
)

CALENDAR_SCRAPE_MS = Histogram(
    "aq_calendar_scrape_ms",
    "Do tre scrap lich kinh te (ms)",
    ["source"],
    buckets=(100, 200, 500, 1000, 2000, 5000),
)


# =============================================================================
# Gauges
# =============================================================================

REDIS_MEMORY_BYTES = Gauge(
    "aq_redis_memory_bytes",
    "Redis memory usage (bytes)",
)

REDIS_KEY_COUNT = Gauge(
    "aq_redis_key_count",
    "Tong so key trong Redis",
)

ITQ_QUEUE_DEPTH = Gauge(
    "aq_itq_queue_depth",
    "So phan tu trong Incoming Tick Queue",
)

ZONE_ACTIVE_COUNT = Gauge(
    "aq_zone_active_count",
    "So luong zone dang active",
    ["zone_type"],
)

WEBSOCKET_CLIENT_COUNT = Gauge(
    "aq_websocket_client_count",
    "So luong WebSocket client ket noi",
)

DEBATE_QUEUE_SIZE = Gauge(
    "aq_debate_queue_size",
    "So luong debate trong hang doi",
)

COLD_START_ELAPSED_MS = Gauge(
    "aq_cold_start_elapsed_ms",
    "Thoi gian khoi dong (ms)",
    ["stage"],
)

MODEL_IC_ROLLING = Gauge(
    "aq_model_ic_rolling",
    "Information Coefficient (rolling 20 predictions)",
    ["model_name"],
)

MODEL_BRIER_SCORE = Gauge(
    "aq_model_brier_score",
    "Brier Score (rolling 50 predictions)",
    ["model_name"],
)

MODEL_DEGRADED_FLAG = Gauge(
    "aq_model_degraded_flag",
    "Model degraded flag (0=stable, 1=degraded)",
    ["model_name"],
)

FEATURE_DRIFT_SCORE = Gauge(
    "aq_feature_drift_score",
    "Feature Drift Score (PSI-based)",
    ["model_name"],
)


# =============================================================================
# Prometheus endpoint handler
# =============================================================================

async def prometheus_handler(request: "web.Request") -> "web.Response":
    """aiohttp handler tra ve metrics theo Prometheus format."""
    metrics_output = generate_latest(REGISTRY)
    return web.Response(
        body=metrics_output,
        content_type=CONTENT_TYPE_LATEST,
    )
