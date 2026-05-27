import { configureStore } from '@reduxjs/toolkit'
import chartReducer from './chartSlice'
import aiStateReducer from './aiStateSlice'
import macroReducer from './macroSlice'

export const store = configureStore({
  reducer: {
    chart: chartReducer,
    aiState: aiStateReducer,
    macro: macroReducer,
  },
})

export type RootState = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch
