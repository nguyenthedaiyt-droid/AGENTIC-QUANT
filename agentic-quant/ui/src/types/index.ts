// =============================================================================
// AGENTIC-QUANT — Core TypeScript Types
// Bieu dien cac cau truc du lieu theo cac message schemas tu Python backend
// =============================================================================

// --- Common ---

export type Timeframe = 'M1' | 'M5' | 'M15' | 'H1' | 'H4' | 'D1'

export type Session =
  | 'ASIAN'
  | 'LONDON_OPEN_KZ'
  | 'LONDON'
  | 'NY_OPEN_KZ'
  | 'NY_AM'
  | 'NY_PM'

export type MacroRegime = 'NORMAL' | 'PRE_NEWS' | 'NEWS_WINDOW' | 'POST_NEWS'

export type ZoneStatus =
  | 'UNMITIGATED'
  | 'WICK_TOUCHED'
  | 'WICK_FILLED_HALF'
  | 'WODY_FILLED'
  | 'MITIGATED'

export type ZoneType = 'FVG_BULL' | 'FVG_BEAR' | 'OB_BULL' | 'OB_BEAR' | 'VI_BULL' | 'VI_BEAR'

export type PredictionOutcome = 'BSL_HIT' | 'SSL_HIT' | 'LATERAL'

export type ConfidenceQualifier = 'HIGH' | 'MEDIUM' | 'LOW'

export type ModelDegradedFlag = 'STABLE' | 'MINOR' | 'SIGNIFICANT'

export type SystemStatus = 'CONNECTED' | 'WARNING' | 'CRITICAL' | 'DISCONNECTED'

export type RegimeType = 'TRENDING_LV' | 'TRENDING_HV' | 'CHOPPY_HV' | 'NORMAL'

// --- OHLCV ---

export interface OHLCV {
  timestamp: number       // Unix ms
  open: number
  high: number
  low: number
  close: number
  volume: number
  tf: Timeframe
}

export interface TickData {
  symbol: string
  timestamp_us: number    // Unix microseconds
  bid: number
  ask: number
  last: number
  volume: number
  flags: number
}

// --- Zones ---

export interface Zone {
  id: string
  symbol: string
  timeframe: Timeframe
  zone_type: ZoneType
  top: number
  bottom: number
  ce: number              // Consequent Encroachment
  formed_time: number     // Unix ms
  status: ZoneStatus
  p_hold: number         // [0, 1]
  p_hold_updated: number  // Unix ms
  w_zone: number         // HTF alignment weight {0.5, 1.0, 2.0}
  iii_formation: number
  touch_count: number
  last_touch_time: number // Unix ms
  htf_tf: Timeframe | null
}

export interface LiquidityTarget {
  target_type: 'BSL' | 'SSL'
  price: number
  timeframe: Timeframe
  p_probability: number   // P_BSL or P_SSL
  session: Session
}

// --- AI Predictions ---

export interface ModelAPrediction {
  symbol: string
  bar_close_time: number  // Unix ms
  p_bsl: number
  p_ssl: number
  p_lateral: number
  predicted_bsl_level: number
  predicted_ssl_level: number
  bsl_tf: Timeframe
  ssl_tf: Timeframe
  confidence_qualifier: ConfidenceQualifier
  model_version: string
  inference_latency_ms: number
}

export interface ZonePrediction {
  zone_id: string
  zone_type: ZoneType
  p_hold: number
  p_hold_pre_adj: number
  zone_top: number
  zone_bottom: number
  zone_ce: number
  threshold_used: number
}

// --- Debate ---

export interface DebateEvidence {
  text: string
  source: string
  weight: number
}

export interface BullThesis {
  direction: 'BULLISH'
  confidence: number
  target_price: number
  invalidation_price: number
  evidence: DebateEvidence[]
}

export interface BearThesis {
  direction: 'BEARISH'
  confidence: number
  target_price: number
  invalidation_price: number
  evidence: DebateEvidence[]
}

export interface ConsensusResult {
  rating: number           // [-4, +4]
  preferred_direction: 'BULLISH' | 'BEARISH' | 'NEUTRAL'
  conviction_zone_price: number | null
  reasoning: string
  agreement_score: number  // [0, 1]
  confidence_qualifier: ConfidenceQualifier
  is_fallback: boolean
}

export interface DebateRecord {
  symbol: string
  bar_close_time: number   // Unix ms
  bull: BullThesis
  bear: BearThesis
  consensus: ConsensusResult
  precedents_count: number
  latency_ms: number
}

// --- Macro / News ---

export interface NewsEvent {
  id: string
  name: string
  currency: string
  scheduled_time: number   // Unix ms
  impact: 'Low' | 'Medium' | 'High'
  forecast: number | null
  actual: number | null
  previous: number | null
  surprise_factor: number // Z-score, 0 if no actual yet
  surprise_direction: -1 | 0 | 1
  i_news: number          // [0, ~3.0]
  regime_phase: MacroRegime
  active_guardrail: boolean
}

export interface CountdownState {
  currency: string
  next_event: NewsEvent | null
  seconds_to_next: number
  regime_phase: MacroRegime
  active_guardrail: boolean
}

// --- System ---

export interface SystemMetrics {
  ws_latency_avg_ms: number
  ws_latency_p95_ms: number
  ws_latency_p99_ms: number
  messages_per_sec: number
  client_count: number
  ic_rolling_20: number
  brier_score_50: number
  model_degraded: boolean
  redis_memory_pct: number
  itq_queue_depth: number
}

// --- WebSocket Messages (Backend -> Frontend) ---

export type WSMessageType =
  | 'bar_update'
  | 'bar_closed'
  | 'zone_update'
  | 'prediction_update'
  | 'consensus_ready'
  | 'countdown_update'
  | 'system_status'
  | 'full_state_snapshot'

export interface BaseWSMessage {
  type: WSMessageType
  emit_time_ms: number    // Timestamp when backend emitted
  symbol: string
  timestamp: number        // Unix ms
}

export interface BarUpdateMessage extends BaseWSMessage {
  type: 'bar_update'
  ohlcv: OHLCV
}

export interface BarClosedMessage extends BaseWSMessage {
  type: 'bar_closed'
  ohlcv: OHLCV
  usv_ready: boolean
}

export interface ZoneUpdateMessage extends BaseWSMessage {
  type: 'zone_update'
  zones: Zone[]
  new_zones: Zone[]
  mitigated_zone_ids: string[]
}

export interface PredictionUpdateMessage extends BaseWSMessage {
  type: 'prediction_update'
  prediction: ModelAPrediction
  zone_predictions: ZonePrediction[]
  liquidity_targets: LiquidityTarget[]
}

export interface ConsensusReadyMessage extends BaseWSMessage {
  type: 'consensus_ready'
  debate: DebateRecord
}

export interface CountdownUpdateMessage extends BaseWSMessage {
  type: 'countdown_update'
  countdowns: Record<string, CountdownState>
  regime_phase: MacroRegime
  active_guardrail: boolean
}

export interface SystemStatusMessage extends BaseWSMessage {
  type: 'system_status'
  status: SystemStatus
  message: string
  metrics: SystemMetrics
  model_degraded: boolean
  calendar_stale: boolean
  feed_failure: boolean
}

export interface FullStateSnapshotMessage extends BaseWSMessage {
  type: 'full_state_snapshot'
  symbol: string
  current_bar: OHLCV
  bars: OHLCV[]
  zones: Zone[]
  active_predictions: ModelAPrediction[]
  recent_debate: DebateRecord | null
  countdowns: Record<string, CountdownState>
  system_metrics: SystemMetrics
  session: Session
  regime: RegimeType
}

export type WSMessage =
  | BarUpdateMessage
  | BarClosedMessage
  | ZoneUpdateMessage
  | PredictionUpdateMessage
  | ConsensusReadyMessage
  | CountdownUpdateMessage
  | SystemStatusMessage
  | FullStateSnapshotMessage
