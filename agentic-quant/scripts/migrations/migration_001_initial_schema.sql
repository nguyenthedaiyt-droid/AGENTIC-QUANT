-- =============================================================================
-- AGENTIC-QUANT — Initial Schema Migration
-- Version: 001
-- Description: Tao cac bang chinh cho Phase 3 Memory Engine
-- Bang: predictions, zone_history, model_performance, system_metrics
-- Khong bao gom news_outcomes vi da ton tai trong calendar_scraper.py
-- =============================================================================

-- PRAGMA: cai dat WAL mode cho performance
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = 10000;
PRAGMA mmap_size = 268435456;  -- 256MB
PRAGMA temp_store = MEMORY;
PRAGMA foreign_keys = ON;

-- =============================================================================
-- Table: predictions
-- Luu prediction tu Model A/B moi bar, ket qua outcome, context, diagnostics
-- =============================================================================
CREATE TABLE IF NOT EXISTS predictions (
    -- Primary key
    prediction_id TEXT PRIMARY KEY,

    -- Symbol & timing
    symbol TEXT NOT NULL,
    bar_close_time INTEGER NOT NULL,         -- Unix ms
    bar_close_time_str TEXT NOT NULL,         -- ISO string for human readability

    -- Model A outputs
    p_bsl REAL NOT NULL,
    p_ssl REAL NOT NULL,
    p_lateral REAL NOT NULL,
    predicted_bsl_level REAL NOT NULL,
    predicted_ssl_level REAL NOT NULL,
    bsl_tf TEXT NOT NULL,
    ssl_tf TEXT NOT NULL,
    confidence_qualifier TEXT NOT NULL,      -- HIGH | MEDIUM | LOW
    model_version TEXT NOT NULL,
    inference_latency_ms REAL NOT NULL,

    -- Session & macro context
    session_id TEXT NOT NULL,                -- ASIAN | LONDON_OPEN_KZ | LONDON | ...
    macro_regime TEXT NOT NULL,              -- NORMAL | PRE_NEWS | NEWS_WINDOW | POST_NEWS
    i_news REAL NOT NULL DEFAULT 0.0,
    active_guardrail INTEGER NOT NULL DEFAULT 0,

    -- Model B outputs (zone predictions)
    zone_predictions_json TEXT,               -- JSON array of ZonePrediction objects

    -- Consensus (from Multi-Agent Debate)
    consensus_rating INTEGER,
    consensus_direction TEXT,
    consensus_agreement REAL,
    consensus_conviction_price REAL,
    debate_used_fallback INTEGER NOT NULL DEFAULT 0,
    debate_latency_ms REAL,

    -- RAG context
    rag_precedents_count INTEGER NOT NULL DEFAULT 0,
    rag_precedents_json TEXT,                 -- JSON array of precedent summaries

    -- Outcome (determined by OutcomeDeterminator)
    outcome_determined INTEGER NOT NULL DEFAULT 0,  -- 0 = pending, 1 = confirmed
    outcome TEXT,                              -- BSL_HIT | SSL_HIT | LATERAL | TIMEOUT | ZONE_HOLD
    outcome_time INTEGER,                      -- Unix ms khi xac dinh

    -- Diagnostics
    ic_at_prediction REAL,                     -- Information Coefficient luc predict
    created_at TEXT NOT NULL                  -- ISO timestamp
);

-- Indexes cho predictions
CREATE INDEX IF NOT EXISTS idx_predictions_symbol_time
ON predictions(symbol, bar_close_time DESC);

CREATE INDEX IF NOT EXISTS idx_predictions_outcome_pending
ON predictions(symbol, outcome_determined)
WHERE outcome_determined = 0;

CREATE INDEX IF NOT EXISTS idx_predictions_outcome
ON predictions(outcome)
WHERE outcome IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_predictions_session
ON predictions(session_id, bar_close_time DESC);

CREATE INDEX IF NOT EXISTS idx_predictions_regime
ON predictions(macro_regime, bar_close_time DESC);

-- =============================================================================
-- Table: zone_history
-- Lich su lifecycle cua zone (insert khi tao, update khi co ket qua)
-- =============================================================================
CREATE TABLE IF NOT EXISTS zone_history (
    -- Primary key
    zone_id TEXT PRIMARY KEY,

    -- Zone identity
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    zone_type TEXT NOT NULL,                   -- FVG_BULL | FVG_BEAR | OB_BULL | OB_BEAR | VI_BULL | VI_BEAR
    top REAL NOT NULL,
    bottom REAL NOT NULL,
    ce REAL NOT NULL DEFAULT 0.0,

    -- Formation
    formed_time INTEGER NOT NULL,               -- Unix ms
    htf_tf TEXT,                                -- Higher timeframe context

    -- Metrics
    p_hold REAL NOT NULL,
    w_zone REAL NOT NULL DEFAULT 1.0,
    iii_formation REAL NOT NULL DEFAULT 0.0,
    touch_count INTEGER NOT NULL DEFAULT 0,

    -- Status at creation
    initial_status TEXT NOT NULL DEFAULT 'UNMITIGATED',

    -- Result (updated by OutcomeDeterminator)
    result TEXT,                               -- HOLD | PARTIAL_HOLD | MITIGATED_EARLY | BROKEN
    result_time INTEGER,                        -- Unix ms
    result_bar_close_time INTEGER,              -- Bar close time when result was determined

    -- Context
    session_id TEXT NOT NULL,
    macro_regime TEXT NOT NULL DEFAULT 'NORMAL',
    prediction_id TEXT,                         -- FK to predictions.prediction_id

    created_at TEXT NOT NULL
);

-- Indexes cho zone_history
CREATE INDEX IF NOT EXISTS idx_zone_history_symbol_time
ON zone_history(symbol, formed_time DESC);

CREATE INDEX IF NOT EXISTS idx_zone_history_type
ON zone_history(symbol, zone_type, formed_time DESC);

CREATE INDEX IF NOT EXISTS idx_zone_history_result
ON zone_history(result)
WHERE result IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_zone_history_prediction
ON zone_history(prediction_id)
WHERE prediction_id IS NOT NULL;

-- =============================================================================
-- Table: model_performance
-- Theo doi IC, Brier, ECE, F1 theo thoi gian
-- =============================================================================
CREATE TABLE IF NOT EXISTS model_performance (
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,

    symbol TEXT NOT NULL,
    model_name TEXT NOT NULL,                  -- model_a | model_b | lstm | ensemble

    -- Timing
    evaluation_window_start INTEGER NOT NULL,   -- Unix ms
    evaluation_window_end INTEGER NOT NULL,      -- Unix ms
    window_label TEXT NOT NULL,                 -- e.g. "2024-W24" hoac "2024-06-15"

    -- Model A metrics
    ic_bsl REAL,
    ic_ssl REAL,
    ic_composite REAL,
    brier_score REAL,
    ece REAL,

    -- Model B metrics
    f1_hold REAL,
    precision_hold REAL,
    recall_hold REAL,

    -- Model B thresholds used
    optimal_threshold REAL,

    -- Feature drift
    feature_drift_score REAL,
    drifted_features_json TEXT,                 -- JSON array of drifted feature names

    -- Model version
    model_version TEXT NOT NULL,

    -- Outcome counts in window
    n_total INTEGER NOT NULL DEFAULT 0,
    n_bsl_hit INTEGER NOT NULL DEFAULT 0,
    n_ssl_hit INTEGER NOT NULL DEFAULT 0,
    n_lateral INTEGER NOT NULL DEFAULT 0,

    created_at TEXT NOT NULL
);

-- Indexes cho model_performance
CREATE INDEX IF NOT EXISTS idx_model_perf_symbol_model
ON model_performance(symbol, model_name, evaluation_window_end DESC);

CREATE INDEX IF NOT EXISTS idx_model_perf_ic
ON model_performance(symbol, model_name, ic_composite DESC)
WHERE ic_composite IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_model_perf_window
ON model_performance(window_label, model_name);

-- =============================================================================
-- Table: system_metrics
-- Telemetry: latency, memory, throughput, health
-- =============================================================================
CREATE TABLE IF NOT EXISTS system_metrics (
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Timing
    timestamp INTEGER NOT NULL,                 -- Unix ms
    timestamp_str TEXT NOT NULL,                -- ISO string

    -- Component
    component TEXT NOT NULL,                   -- ipc | redis | sqlite | xgboost_a | xgboost_b | lstm | debate

    -- Latency metrics
    latency_avg_ms REAL,
    latency_p95_ms REAL,
    latency_p99_ms REAL,

    -- Throughput
    messages_per_sec REAL,
    throughput_per_sec REAL,

    -- Memory
    memory_mb REAL,
    redis_memory_mb REAL,
    redis_memory_pct REAL,

    -- Redis
    redis_hit_rate REAL,
    redis_key_count INTEGER,

    -- Queue depth
    itq_queue_depth INTEGER,

    -- System health
    feed_failure INTEGER NOT NULL DEFAULT 0,
    calendar_stale INTEGER NOT NULL DEFAULT 0,
    model_degraded INTEGER NOT NULL DEFAULT 0,

    -- Model quality
    ic_rolling_20 REAL,
    brier_score_50 REAL,
    ece_rolling_20 REAL,

    -- Extra tags
    tags_json TEXT                              -- JSON object of additional tags
);

-- Indexes cho system_metrics
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp
ON system_metrics(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_system_metrics_component
ON system_metrics(component, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_system_metrics_degraded
ON system_metrics(model_degraded, timestamp DESC)
WHERE model_degraded = 1;

-- =============================================================================
-- Table: pending_archive
-- Queue fallback khi VectorDB (Qdrant/ChromaDB) khong kha dung
-- =============================================================================
CREATE TABLE IF NOT EXISTS pending_archive (
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Content type
    content_type TEXT NOT NULL,                 -- debate | zone_embedding
    content_id TEXT NOT NULL,                   -- debate key or zone_id

    -- Payload
    payload_json TEXT NOT NULL,                 -- Full record as JSON

    -- Attempt tracking
    attempts INTEGER NOT NULL DEFAULT 0,
    last_attempt INTEGER,                       -- Unix ms
    last_error TEXT,
    max_attempts INTEGER NOT NULL DEFAULT 5,

    -- Status
    status TEXT NOT NULL DEFAULT 'pending',     -- pending | failed | completed

    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Indexes cho pending_archive
CREATE INDEX IF NOT EXISTS idx_pending_archive_status
ON pending_archive(status, created_at ASC)
WHERE status = 'pending';
