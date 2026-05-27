# =============================================================================
# AGENTIC-QUANT — Data Ingestion Module
# =============================================================================

from .tick_frame import TickFrame
from .tick_receiver import TickReceiver
from .ohlcv_aggregator import (
    OHLCVAggregator,
    BarState,
    TIMEFRAME_SECONDS,
    CASCADE_ORDER,
    LTF_TIMEFRAMES,
    HTF_TIMEFRAMES,
)
from .volumetrics_engine import VolumetricsEngine, ATRCalculator
from .mtf_synchronizer import MTFSynchronizer, LeakageGuard, UnifiedStateVector
from .itq_queue import IncomingTickQueue, QueuedTick
from .desync_detector import (
    TimeframeDesyncDetector,
    DesyncReport,
    DesyncSeverity,
)
from .tv_webhook import TVWebhookHandler, TVAlertPayload, TokenBucketRateLimiter
from .historical_tick_loader import (
    HistoricalTickLoader,
    BacktestConfig,
    export_mt5_ticks,
    validate_coverage,
)

__all__ = [
    # Tick data
    "TickFrame",
    # Receivers
    "TickReceiver",
    # OHLCV
    "OHLCVAggregator",
    "BarState",
    "TIMEFRAME_SECONDS",
    "CASCADE_ORDER",
    "LTF_TIMEFRAMES",
    "HTF_TIMEFRAMES",
    # Volumetrics
    "VolumetricsEngine",
    "ATRCalculator",
    # MTF Sync
    "MTFSynchronizer",
    "LeakageGuard",
    "UnifiedStateVector",
    # ITQ
    "IncomingTickQueue",
    "QueuedTick",
    # Desync
    "TimeframeDesyncDetector",
    "DesyncReport",
    "DesyncSeverity",
    # TV Webhook
    "TVWebhookHandler",
    "TVAlertPayload",
    "TokenBucketRateLimiter",
    # Backtest
    "HistoricalTickLoader",
    "BacktestConfig",
    "export_mt5_ticks",
    "validate_coverage",
]
