// =============================================================================
// LatencyIndicator.tsx — Hien thi latency
// Green < 30ms, Yellow < 50ms, Red > 50ms
// =============================================================================

import { memo, useState, useEffect } from 'react'

// =============================================================================
// Props
// =============================================================================
interface LatencyIndicatorProps {
  getLatency: () => number
}

// =============================================================================
// LatencyIndicator Component
// =============================================================================
const LatencyIndicator = memo(function LatencyIndicator({ getLatency }: LatencyIndicatorProps) {
  const [latency, setLatency] = useState<number>(0)

  // Poll latency moi giay
  useEffect(() => {
    const interval = setInterval(() => {
      setLatency(getLatency())
    }, 1000)
    return () => clearInterval(interval)
  }, [getLatency])

  // Xac dinh color
  const color = latency < 30 ? '#4caf50' : latency < 50 ? '#ffeb3b' : '#f44336'
  const bgColor = latency < 30 ? 'rgba(76, 175, 80, 0.15)' : latency < 50 ? 'rgba(255, 235, 59, 0.15)' : 'rgba(244, 67, 54, 0.15)'
  const level = latency < 30 ? 'Good' : latency < 50 ? 'Fair' : 'Poor'

  return (
    <div
      className="flex items-center gap-2 px-2 py-1 rounded text-xs"
      style={{ backgroundColor: bgColor }}
    >
      {/* Dot indicator */}
      <div
        className="w-2 h-2 rounded-full"
        style={{ backgroundColor: color }}
      />

      {/* Latency value */}
      <span style={{ color }} className="font-mono font-bold">
        {latency}ms
      </span>

      {/* Level */}
      <span className="text-gray-400">{level}</span>
    </div>
  )
})

export default LatencyIndicator
