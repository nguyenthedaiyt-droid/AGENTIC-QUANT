// =============================================================================
// MTFGhostZones.tsx — Ghost zones overlay tren chart
// Hien thi cac zone voi border dashed theo timeframe
// - H4/D1: 2px dashed
// - H1: 1.5px dashed
// - LTF (M1/M5/M15): 1px dotted
// PARTIALLY_MITIGATED: half opacity
// Max 50 zones, filter p_hold < 0.50 neu exceeded
// Auto-hide LTF khi viewing H1+
// =============================================================================

import { memo, useMemo } from 'react'
import { useAppSelector } from '../../hooks/useAppStore'
import type { Zone, Timeframe } from '../../types/index'

// =============================================================================
// Helper: style cho zone theo timeframe
// =============================================================================
function getZoneStyle(tf: Timeframe) {
  switch (tf) {
    case 'D1':
    case 'H4':
      return { borderWidth: 2, borderStyle: 'dashed' as const, opacity: 0.8 }
    case 'H1':
      return { borderWidth: 1.5, borderStyle: 'dashed' as const, opacity: 0.7 }
    default: // M1, M5, M15
      return { borderWidth: 1, borderStyle: 'dotted' as const, opacity: 0.6 }
  }
}

// =============================================================================
// Helper: lay mau zone theo zone_type
// =============================================================================
function getZoneColor(zoneType: string): string {
  if (zoneType.includes('BULL') || zoneType.includes('VI_BULL')) return '#26a69a'
  if (zoneType.includes('BEAR') || zoneType.includes('VI_BEAR')) return '#ef5350'
  if (zoneType.includes('OB')) return '#ffa726'
  return '#42a5f5'
}

// =============================================================================
// MTFGhostZones Component
// =============================================================================
const MTFGhostZones = memo(function MTFGhostZones() {
  const zones = useAppSelector((state) => state.chart.zones)
  const currentBar = useAppSelector((state) => state.chart.currentBar)

  // Xac dinh timeframe hien tai tu so bars (estimate)
  // Neu co currentBar, lay tf tu do
  const currentTf: Timeframe | null = currentBar?.tf ?? null

  // Filter zones: auto-hide LTF khi viewing H1+
  const filteredZones = useMemo(() => {
    // LTF = M1, M5, M15
    const isH1OrHigher = currentTf === 'H1' || currentTf === 'H4' || currentTf === 'D1'

    let result = zones
    // Neu dang o H1+, an LTF zones
    if (isH1OrHigher) {
      result = result.filter((z) => z.timeframe !== 'M1' && z.timeframe !== 'M5' && z.timeframe !== 'M15')
    }

    // Max 50 zones
    if (result.length > 50) {
      // Sort theo p_hold giam dan, giu 50 zone co p_hold cao nhat
      result = result
        .slice()
        .sort((a, b) => b.p_hold - a.p_hold)
        .filter((z) => z.p_hold >= 0.5)
        .slice(0, 50)
    }

    return result
  }, [zones, currentTf])

  // Neu khong co du lieu, khong render
  if (filteredZones.length === 0) return null

  return (
    <div className="absolute inset-0 pointer-events-none" style={{ zIndex: 10 }}>
      {filteredZones.map((zone: Zone) => {
        const style = getZoneStyle(zone.timeframe)
        const color = getZoneColor(zone.zone_type)
        const isMitigated = zone.status === 'MITIGATED' || (zone.status as string) === 'PARTIALLY_MITIGATED'
        const opacity = isMitigated ? style.opacity * 0.5 : style.opacity

        return (
          <div
            key={zone.id}
            className="absolute"
            style={{
              left: '0',
              right: '0',
              // NOTE: Vi tri thuc te (top/bottom) can duoc tinh toan dua tren
              // price-to-y mapping tu chart. O day ta su dung overlay absolute
              // voi position duoc tinh tu chart scale.
              // Trong thuc te, overlay nay can duoc Tauri chart handle de tinh toa do.
              borderTop: `${style.borderWidth}px ${style.borderStyle} ${color}`,
              borderBottom: `${style.borderWidth}px ${style.borderStyle} ${color}`,
              opacity,
              transition: 'opacity 0.3s ease',
              backgroundColor: `${color}11`,
              top: '0',
              bottom: '0',
            }}
            title={`${zone.zone_type} [${zone.timeframe}] p_hold=${(zone.p_hold * 100).toFixed(1)}% status=${zone.status}`}
          />
        )
      })}
    </div>
  )
})

export default MTFGhostZones
