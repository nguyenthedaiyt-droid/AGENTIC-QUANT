import { createSlice, type PayloadAction } from '@reduxjs/toolkit'
import type { OHLCV, Zone, Session, RegimeType } from '../types/index'
import type { LiquidityTarget } from '../types/index'

// --- chartSlice ---

interface ChartState {
  currentBar: OHLCV | null
  bars: OHLCV[]
  zones: Zone[]
  liquidityTargets: LiquidityTarget[]
  session: Session
  regime: RegimeType
}

const initialChartState: ChartState = {
  currentBar: null,
  bars: [],
  zones: [],
  liquidityTargets: [],
  session: 'ASIAN',
  regime: 'NORMAL',
}

const chartSlice = createSlice({
  name: 'chart',
  initialState: initialChartState,
  reducers: {
    setCurrentBar(state, action: PayloadAction<OHLCV>) {
      state.currentBar = action.payload
    },
    addBar(state, action: PayloadAction<OHLCV>) {
      state.bars.push(action.payload)
      if (state.bars.length > 2000) {
        state.bars = state.bars.slice(-2000)
      }
    },
    updateLastBar(state, action: PayloadAction<OHLCV>) {
      const last = state.bars[state.bars.length - 1]
      if (last && last.timestamp === action.payload.timestamp) {
        state.bars[state.bars.length - 1] = action.payload
      } else {
        state.bars.push(action.payload)
      }
      state.currentBar = action.payload
    },
    setZones(state, action: PayloadAction<Zone[]>) {
      state.zones = action.payload
    },
    setLiquidityTargets(state, action: PayloadAction<LiquidityTarget[]>) {
      state.liquidityTargets = action.payload
    },
    setSession(state, action: PayloadAction<Session>) {
      state.session = action.payload
    },
    setRegime(state, action: PayloadAction<RegimeType>) {
      state.regime = action.payload
    },
    loadFullState(
      state,
      action: PayloadAction<{
        bars: OHLCV[]
        zones: Zone[]
        liquidityTargets: LiquidityTarget[]
        session: Session
        regime: RegimeType
      }>,
    ) {
      state.bars = action.payload.bars
      state.zones = action.payload.zones
      state.liquidityTargets = action.payload.liquidityTargets
      state.session = action.payload.session
      state.regime = action.payload.regime
    },
  },
})

export const {
  setCurrentBar,
  addBar,
  updateLastBar,
  setZones,
  setLiquidityTargets,
  setSession,
  setRegime,
  loadFullState,
} = chartSlice.actions
export default chartSlice.reducer
