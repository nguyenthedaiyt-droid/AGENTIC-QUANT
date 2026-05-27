// =============================================================================
// ws_messages.ts — TypeScript interfaces matching Python message_schema
// Tat ca cac message nhan tu WebSocket backend
// =============================================================================

// Cac type duoc dinh nghia o ./index.ts
// File nay chi re-export de dam bao module structure

import type {
  BarUpdateMessage as BUM,
  ZoneUpdateMessage as ZUM,
  PredictionUpdateMessage as PUM,
  CountdownUpdateMessage as CUM,
  ConsensusReadyMessage as CRM,
  SystemStatusMessage as SSM,
  FullStateSnapshotMessage as FSM,
  WSMessage as WSM,
  WSMessageType as WMT,
} from './index'

export type BarUpdateMessage = BUM
export type ZoneUpdateMessage = ZUM
export type PredictionUpdateMessage = PUM
export type CountdownUpdateMessage = CUM
export type ConsensusReadyMessage = CRM
export type SystemStatusMessage = SSM
export type FullStateSnapshotMessage = FSM
export type WSMessage = WSM
export type WSMessageType = WMT
