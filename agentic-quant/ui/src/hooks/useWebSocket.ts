// =============================================================================
// useWebSocket — WebSocket hook với reconnect exponential backoff
// =============================================================================

import { useEffect, useRef, useCallback } from 'react'
import { useAppDispatch } from './useAppStore'
import {
  updateLastBar,
  addBar,
  setZones,
  setLiquidityTargets,
  loadFullState,
} from '../store/chartSlice'
import {
  setPrediction,
  setZonePredictions,
  setDebate,
  setModelDegraded,
} from '../store/aiStateSlice'
import {
  setCountdowns,
  setActiveGuardrail,
  setRegimePhase,
  setCalendarStale,
} from '../store/macroSlice'
import type {
  WSMessage,
  BarUpdateMessage,
  ZoneUpdateMessage,
  PredictionUpdateMessage,
  CountdownUpdateMessage,
  ConsensusReadyMessage,
  SystemStatusMessage,
  FullStateSnapshotMessage,
} from '../types/index'

// =============================================================================
// Cấu hình
// =============================================================================
const WS_URL = 'ws://localhost:47290'
const MAX_BACKOFF_MS = 30_000 // 30 giây
const INITIAL_BACKOFF_MS = 1_000 // 1 giây

// =============================================================================
// Helper: dispatch message dựa trên type
// =============================================================================
function dispatchMessage(dispatch: ReturnType<typeof useAppDispatch>, msg: WSMessage) {
  switch (msg.type) {
    case 'bar_update':
    case 'bar_closed': {
      const barMsg = msg as BarUpdateMessage
      dispatch(updateLastBar(barMsg.ohlcv))
      // Nếu là bar_closed, thêm bar mới
      if (msg.type === 'bar_closed') {
        dispatch(addBar(barMsg.ohlcv))
      }
      break
    }

    case 'zone_update': {
      const zoneMsg = msg as ZoneUpdateMessage
      dispatch(setZones(zoneMsg.zones))
      break
    }

    case 'prediction_update': {
      const predMsg = msg as PredictionUpdateMessage
      dispatch(setPrediction(predMsg.prediction))
      dispatch(setZonePredictions(predMsg.zone_predictions))
      dispatch(setLiquidityTargets(predMsg.liquidity_targets))
      break
    }

    case 'consensus_ready': {
      const consMsg = msg as ConsensusReadyMessage
      dispatch(setDebate(consMsg.debate))
      break
    }

    case 'countdown_update': {
      const countMsg = msg as CountdownUpdateMessage
      dispatch(setCountdowns(countMsg.countdowns))
      dispatch(setActiveGuardrail(countMsg.active_guardrail))
      dispatch(setRegimePhase(countMsg.regime_phase))
      break
    }

    case 'system_status': {
      const sysMsg = msg as SystemStatusMessage
      dispatch(setModelDegraded(sysMsg.model_degraded))
      dispatch(setCalendarStale(sysMsg.calendar_stale))
      break
    }

    case 'full_state_snapshot': {
      const snapMsg = msg as FullStateSnapshotMessage
      // Load bars, zones, liquidity targets từ snapshot
      dispatch(
        loadFullState({
          bars: snapMsg.bars,
          zones: snapMsg.zones,
          liquidityTargets: [],
          session: snapMsg.session,
          regime: snapMsg.regime,
        }),
      )
      // Load predictions
      if (snapMsg.active_predictions.length > 0) {
        dispatch(setPrediction(snapMsg.active_predictions[0]))
      }
      // Load debate
      dispatch(setDebate(snapMsg.recent_debate))
      // Load countdowns
      dispatch(setCountdowns(snapMsg.countdowns))
      break
    }

    default:
      console.warn('[WS] Unknown message type:', (msg as WSMessage).type)
  }
}

// =============================================================================
// Hook chính
// =============================================================================
export function useWebSocket() {
  const dispatch = useAppDispatch()
  const wsRef = useRef<WebSocket | null>(null)
  const backoffRef = useRef<number>(INITIAL_BACKOFF_MS)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const latencyRef = useRef<number>(0)

  // --- Hàm connect ---
  const connect = useCallback(() => {
    // Cleanup connection cũ nếu có
    if (wsRef.current) {
      wsRef.current.onopen = null
      wsRef.current.onmessage = null
      wsRef.current.onerror = null
      wsRef.current.onclose = null
      if (
        wsRef.current.readyState === WebSocket.OPEN ||
        wsRef.current.readyState === WebSocket.CONNECTING
      ) {
        wsRef.current.close()
      }
    }

    console.log(`[WS] Connecting to ${WS_URL} (backoff=${backoffRef.current}ms)...`)

    try {
      const ws = new WebSocket(WS_URL)
      wsRef.current = ws

      ws.onopen = () => {
        console.log('[WS] Connected')
        // Reset backoff về giá trị ban đầu
        backoffRef.current = INITIAL_BACKOFF_MS
      }

      ws.onmessage = (event: MessageEvent) => {
        try {
          const msg: WSMessage = JSON.parse(event.data)

          // Tính latency = thời gian hiện tại - thời gian backend emit
          if (msg.emit_time_ms) {
            latencyRef.current = Date.now() - msg.emit_time_ms
          }

          dispatchMessage(dispatch, msg)
        } catch (err) {
          console.error('[WS] Failed to parse message:', err)
        }
      }

      ws.onerror = (err: Event) => {
        console.error('[WS] Error:', err)
      }

      ws.onclose = (event: CloseEvent) => {
        console.log(`[WS] Disconnected (code=${event.code}). Reconnecting in ${backoffRef.current}ms...`)

        // Schedule reconnect với exponential backoff
        reconnectTimerRef.current = setTimeout(() => {
          connect()
          // Exponential backoff: nhân đôi, tối đa 30s
          backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF_MS)
        }, backoffRef.current)
      }
    } catch (err) {
      console.error('[WS] Failed to create WebSocket:', err)
      // Retry sau backoff nếu tạo connection thất bại
      reconnectTimerRef.current = setTimeout(() => {
        connect()
        backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF_MS)
      }, backoffRef.current)
    }
  }, [dispatch])

  // --- Cleanup ---
  useEffect(() => {
    connect()

    return () => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current)
        reconnectTimerRef.current = null
      }
      if (wsRef.current) {
        wsRef.current.onopen = null
        wsRef.current.onmessage = null
        wsRef.current.onerror = null
        wsRef.current.onclose = null
        if (
          wsRef.current.readyState === WebSocket.OPEN ||
          wsRef.current.readyState === WebSocket.CONNECTING
        ) {
          wsRef.current.close()
        }
        wsRef.current = null
      }
    }
  }, [connect])

  // Expose send function để components có thể gửi message
  const send = useCallback((data: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    } else {
      console.warn('[WS] Cannot send — not connected')
    }
  }, [])

  const getLatency = useCallback(() => latencyRef.current, [])

  return { send, getLatency }
}
