import { createSlice, type PayloadAction } from '@reduxjs/toolkit'
import type { NewsEvent, CountdownState, MacroRegime } from '../types/index'

// --- macroSlice ---

interface MacroState {
  events: NewsEvent[]
  countdowns: Record<string, CountdownState>
  activeGuardrail: boolean
  regimePhase: MacroRegime
  calendarStale: boolean
}

const initialMacroState: MacroState = {
  events: [],
  countdowns: {},
  activeGuardrail: false,
  regimePhase: 'NORMAL',
  calendarStale: false,
}

const macroSlice = createSlice({
  name: 'macro',
  initialState: initialMacroState,
  reducers: {
    setEvents(state, action: PayloadAction<NewsEvent[]>) {
      state.events = action.payload
    },
    setCountdowns(state, action: PayloadAction<Record<string, CountdownState>>) {
      state.countdowns = action.payload
    },
    setActiveGuardrail(state, action: PayloadAction<boolean>) {
      state.activeGuardrail = action.payload
    },
    setRegimePhase(state, action: PayloadAction<MacroRegime>) {
      state.regimePhase = action.payload
    },
    setCalendarStale(state, action: PayloadAction<boolean>) {
      state.calendarStale = action.payload
    },
  },
})

export const { setEvents, setCountdowns, setActiveGuardrail, setRegimePhase, setCalendarStale } =
  macroSlice.actions
export default macroSlice.reducer
