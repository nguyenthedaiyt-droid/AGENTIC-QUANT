// =============================================================================
// SystemStatusBanner.tsx — Warning banners
// Hien thi canh bao: disconnection, model_degraded, calendar_stale
// =============================================================================

import { memo } from 'react'
import { useAppSelector } from '../../hooks/useAppStore'

// =============================================================================
// SystemStatusBanner Component
// =============================================================================
const SystemStatusBanner = memo(function SystemStatusBanner() {
  const modelDegraded = useAppSelector((state) => state.aiState.modelDegraded)
  const calendarStale = useAppSelector((state) => state.macro.calendarStale)

  const warnings: { type: string; message: string; color: string }[] = []

  if (modelDegraded) {
    warnings.push({
      type: 'model_degraded',
      message: '⚠ Model degraded — predictions may be unreliable',
      color: '#ffa726',
    })
  }

  if (calendarStale) {
    warnings.push({
      type: 'calendar_stale',
      message: '⚠ Economic calendar data is stale — news events may be outdated',
      color: '#ffa726',
    })
  }

  // Neu khong co warning, khong render
  if (warnings.length === 0) return null

  return (
    <div className="flex flex-col gap-1 px-4 py-2">
      {warnings.map((w) => (
        <div
          key={w.type}
          className="flex items-center gap-2 px-3 py-1.5 rounded text-xs font-medium animate-pulse"
          style={{
            backgroundColor: `${w.color}22`,
            color: w.color,
            borderLeft: `3px solid ${w.color}`,
          }}
        >
          {w.message}
        </div>
      ))}
    </div>
  )
})

export default SystemStatusBanner
