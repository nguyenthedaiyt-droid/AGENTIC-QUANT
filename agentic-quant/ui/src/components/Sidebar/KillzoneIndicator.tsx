// =============================================================================
// KillzoneIndicator.tsx — Session indicator
// Asian=#009688, London=#2196f3, NY=#ff5722
// =============================================================================

import { memo } from 'react'
import { useAppSelector } from '../../hooks/useAppStore'

// =============================================================================
// Mapping session -> color + label
// =============================================================================
const SESSION_CONFIG: Record<string, { color: string; label: string; time: string }> = {
  ASIAN: { color: '#009688', label: 'Asian', time: '00:00-09:00' },
  LONDON_OPEN_KZ: { color: '#2196f3', label: 'London KZ', time: '08:00-10:00' },
  LONDON: { color: '#2196f3', label: 'London', time: '08:00-17:00' },
  NY_OPEN_KZ: { color: '#ff5722', label: 'NY KZ', time: '13:00-15:00' },
  NY_AM: { color: '#ff5722', label: 'NY AM', time: '13:00-17:00' },
  NY_PM: { color: '#ff5722', label: 'NY PM', time: '17:00-20:00' },
}

// =============================================================================
// KillzoneIndicator Component
// =============================================================================
const KillzoneIndicator = memo(function KillzoneIndicator() {
  const session = useAppSelector((state) => state.chart.session)

  const config = SESSION_CONFIG[session] ?? { color: '#555', label: session, time: '' }

  return (
    <div className="bg-gray-800 rounded-lg p-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Session
      </h3>

      <div className="flex items-center gap-2">
        {/* Color indicator */}
        <div
          className="w-3 h-3 rounded-full flex-shrink-0"
          style={{ backgroundColor: config.color }}
        />

        <div className="flex flex-col">
          <span className="text-sm font-bold" style={{ color: config.color }}>
            {config.label}
          </span>
          {config.time && (
            <span className="text-xs text-gray-500">{config.time} UTC</span>
          )}
        </div>
      </div>
    </div>
  )
})

export default KillzoneIndicator
