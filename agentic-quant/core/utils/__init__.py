# =============================================================================
# AGENTIC-QUANT — Utils Module
# =============================================================================

from .logging import setup_logging, get_logger
from .timing import measure_latency, LatencyTracker, MovingAverageLatency
from .metrics import (
    TICK_RECEIVED_TOTAL,
    BAR_CLOSED_TOTAL,
    PREDICTION_MADE_TOTAL,
    DEBATE_TRIGGERED_TOTAL,
    NEWS_EVENT_TOTAL,
    ZONE_CREATED_TOTAL,
    ZONE_MITIGATED_TOTAL,
    MODEL_DEGRADED_TOTAL,
    INFERENCE_LATENCY_MS,
    IPC_LATENCY_MS,
    USV_BUILD_LATENCY_MS,
    LSTM_INFERENCE_MS,
    XGBOOST_INFERENCE_MS,
    DEBATE_LATENCY_MS,
    TICK_PROCESSING_MS,
    BAR_AGGREGATION_MS,
    CALENDAR_SCRAPE_MS,
    REDIS_MEMORY_BYTES,
    REDIS_KEY_COUNT,
    ITQ_QUEUE_DEPTH,
    ZONE_ACTIVE_COUNT,
    WEBSOCKET_CLIENT_COUNT,
    DEBATE_QUEUE_SIZE,
    COLD_START_ELAPSED_MS,
    MODEL_IC_ROLLING,
    MODEL_BRIER_SCORE,
    MODEL_DEGRADED_FLAG,
    FEATURE_DRIFT_SCORE,
    prometheus_handler,
)

__all__ = [
    "setup_logging",
    "get_logger",
    "measure_latency",
    "LatencyTracker",
    "MovingAverageLatency",
]
