// =============================================================================
// AIHeatmapOverlay.tsx — AI heatmap color overlay
// To mau zone theo p_hold:
//   p_hold 1.0 -> hsla(120,80%,50%,0.4) bright green
//   p_hold 0.5 -> hsla(60,50%,50%,0.25) yellow
//   p_hold 0.0 -> hsla(0,80%,50%,0.12) red
// Pulse animation khi delta_p_hold > 0.15
// Pre-news mode: LTF opacity * 0.4 + '?' label
// =============================================================================

import { memo, useMemo, useRef, useEffect } from 'react'
import { useAppSelector } from '../../hooks/useAppStore'
import type { Zone } from '../../types/index'

// =============================================================================
// Helper: tinh mau hsla tu p_hold
// =============================================================================
function heatColor(p_hold: number): string {
  const hue = p_hold * 120 // 0 -> 0 (red), 0.5 -> 60 (yellow), 1.0 -> 120 (green)
  const saturation = 80 - p_hold * 30
  const lightness = 50
  const alpha = 0.12 + p_hold * 0.28
  return `hsla(${hue}, ${saturation}%, ${lightness}%, ${alpha})`
}

// =============================================================================
// AIHeatmapOverlay Component
// =============================================================================
const AIHeatmapOverlay = memo(function AIHeatmapOverlay() {
  const zones = useAppSelector((state) => state.chart.zones)
  const prevZonesRef = useRef<Map<string, number>>(new Map())
  const pulsingZonesRef = useRef<Set<string>>(new Set())
  const regimePhase = useAppSelector((state) => state.macro.regimePhase)
  const currentTf = useAppSelector((state) => state.chart.currentBar?.tf ?? null)

  // Detect pulse zones (delta_p_hold > 0.15)
  useEffect(() => {
    const newPulsing = new Set<string>()
    const currentMap = new Map<string, number>()

    for (const zone of zones) {
      currentMap.set(zone.id, zone.p_hold)
      const prev = prevZonesRef.current.get(zone.id)
      if (prev !== undefined) {
        const delta = Math.abs(zone.p_hold - prev)
        if (delta > 0.15) {
          newPulsing.add(zone.id)
        }
      }
    }

    pulsingZonesRef.current = newPulsing
    prevZonesRef.current = currentMap

    const timer = setTimeout(() => {
      pulsingZonesRef.current = new Set()
    }, 500)

    return () => clearTimeout(timer)
  }, [zones])

  // Filter zones cho heatmap
  const heatZones = useMemo(() => {
    let result = zones

    const isH1OrHigher = currentTf === 'H1' || currentTf === 'H4' || currentTf === 'D1'
    if (isH1OrHigher) {
      result = result.filter((z: Zone) => z.timeframe !== 'M1' && z.timeframe !== 'M5' && z.timeframe !== 'M15')
    }

    return result
  }, [zones, currentTf])

  if (heatZones.length === 0) return null

  const isPreNews = regimePhase === 'PRE_NEWS' || regimePhase === 'NEWS_WINDOW'
  const preNewsOpacity = isPreNews ? 0.4 : 1.0

  return (
    <div className="absolute inset-0 pointer-events-none" style={{ zIndex: 11 }}>
      {heatZones.map((zone: Zone) => {
        const color = heatColor(zone.p_hold)
        const isPulsing = pulsingZonesRef.current.has(zone.id)

        return (
          <div
            key={`heat-${zone.id}`}
            className="absolute"
            style={{
              left: '0',
              right: '0',
              backgroundColor: color,
              opacity: preNewsOpacity,
              animation: isPulsing ? 'pulse-heat 0.5s ease-in-out' : undefined,
              top: '0',
              bottom: '0',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
            title={`${zone.zone_type} [${zone.timeframe}] p_hold=${(zone.p_hold * 100).toFixed(1)}%`}
          >
            {isPreNews && (
              <span
                className="text-white text-xs font-bold"
                style={{ textShadow: '0 0 4px rgba(0,0,0,0.8)', opacity: 0.7 }}
              >
                ?
              </span>
            )}
          </div>
        )
      })}
    </div>
  )
})

export default AIHeatmapOverlay
