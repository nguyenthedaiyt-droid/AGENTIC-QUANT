import { createSlice, type PayloadAction } from '@reduxjs/toolkit'
import type { ModelAPrediction, ZonePrediction, DebateRecord, ConfidenceQualifier } from '@types/index'

// --- aiStateSlice ---

interface AIDebateState {
  debate: DebateRecord | null
  prediction: ModelAPrediction | null
  zonePredictions: ZonePrediction[]
  confidence: ConfidenceQualifier
  modelDegraded: boolean
  modelVersion: string
}

const initialAIState: AIDebateState = {
  debate: null,
  prediction: null,
  zonePredictions: [],
  confidence: 'MEDIUM',
  modelDegraded: false,
  modelVersion: '',
}

const aiStateSlice = createSlice({
  name: 'aiState',
  initialState: initialAIState,
  reducers: {
    setPrediction(state, action: PayloadAction<ModelAPrediction>) {
      state.prediction = action.payload
      state.confidence = action.payload.confidence_qualifier
      state.modelVersion = action.payload.model_version
    },
    setZonePredictions(state, action: PayloadAction<ZonePrediction[]>) {
      state.zonePredictions = action.payload
    },
    setDebate(state, action: PayloadAction<DebateRecord | null>) {
      state.debate = action.payload
    },
    setModelDegraded(state, action: PayloadAction<boolean>) {
      state.modelDegraded = action.payload
    },
  },
})

export const { setPrediction, setZonePredictions, setDebate, setModelDegraded } =
  aiStateSlice.actions
export default aiStateSlice.reducer
