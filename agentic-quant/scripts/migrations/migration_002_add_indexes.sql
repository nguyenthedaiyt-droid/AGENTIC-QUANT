-- =============================================================================
-- AGENTIC-QUANT — Additional Indexes Migration
-- Version: 002
-- Description: Them composite indexes cho performance queries
-- Run sau: migration_001_initial_schema.sql
-- =============================================================================

-- =============================================================================
-- Composite indexes for Zone History
-- =============================================================================

-- Index cho truy van zone theo symbol + timeframe + zone_type (common SMC query)
CREATE INDEX IF NOT EXISTS idx_zone_history_smc_query
ON zone_history(symbol, zone_type, formed_time DESC);

-- Index cho truy van zone theo price range (used by get_zones_near_price)
-- Khi can, SQLite se scan idx_zone_history_symbol_time va filter theo top/bottom

-- Index cho truy van zone result theo symbol (performance analysis)
CREATE INDEX IF NOT EXISTS idx_zone_history_result_analysis
ON zone_history(symbol, result, result_time DESC)
WHERE result IS NOT NULL;

-- =============================================================================
-- Composite indexes for Predictions
-- =============================================================================

-- Index cho truy van prediction theo symbol + regime (RAG training data)
CREATE INDEX IF NOT EXISTS idx_predictions_regime_analysis
ON predictions(symbol, macro_regime, bar_close_time DESC);

-- Index cho truy van prediction theo symbol + outcome (model eval)
CREATE INDEX IF NOT EXISTS idx_predictions_outcome_analysis
ON predictions(symbol, outcome, bar_close_time DESC)
WHERE outcome IS NOT NULL;

-- Index cho truy van prediction batch (lookup by bar_close_time range)
CREATE INDEX IF NOT EXISTS idx_predictions_time_range
ON predictions(symbol, bar_close_time)
WHERE outcome_determined = 0;

-- =============================================================================
-- Composite indexes for Model Performance
-- =============================================================================

-- Index cho truy van model perf theo symbol + model + date range
CREATE INDEX IF NOT EXISTS idx_model_perf_trend
ON model_performance(symbol, model_name, evaluation_window_end DESC)
WHERE ic_composite IS NOT NULL;

-- =============================================================================
-- Composite indexes for System Metrics
-- =============================================================================

-- Index cho truy van system metrics theo component + time range
CREATE INDEX IF NOT EXISTS idx_system_metrics_time_range
ON system_metrics(component, timestamp DESC);

-- Index cho truy van system health theo time (dashboard queries)
CREATE INDEX IF NOT EXISTS idx_system_metrics_health
ON system_metrics(timestamp DESC, feed_failure, model_degraded, calendar_stale);

-- =============================================================================
-- Table: debate_archive_snapshot
-- Backup cua debate records da archive sang VectorDB
-- Dung lam fallback khi VectorDB query that bai
-- =============================================================================

CREATE TABLE IF NOT EXISTS debate_archive_snapshot (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,

    symbol TEXT NOT NULL,
    bar_close_time INTEGER NOT NULL,
    e_usv_id TEXT,                              -- VectorDB point ID

    -- Content
    bull_thesis_json TEXT,
    bear_thesis_json TEXT,
    consensus_json TEXT,

    -- Metadata
    macro_regime TEXT NOT NULL,
    session_id TEXT NOT NULL,
    outcome TEXT,
    outcome_time INTEGER,
    archived_at TEXT NOT NULL,

    UNIQUE(symbol, bar_close_time)
);

CREATE INDEX IF NOT EXISTS idx_debate_snapshot_symbol_time
ON debate_archive_snapshot(symbol, bar_close_time DESC);

CREATE INDEX IF NOT EXISTS idx_debate_snapshot_outcome
ON debate_archive_snapshot(outcome, archived_at DESC)
WHERE outcome IS NOT NULL;
