// =============================================================================
// MacroTimeline.tsx — Vertical lines by scheduled_time
// Low=gray, Medium=yellow, High=red
// Future=dotted, Past=solid
// Tooltip: event_name, currency, impact, forecast, actual, S
// Pre-News shading (-15min +5min)
// Session backgrounds: Asian, London, NY
// =============================================================================

import { memo, useMemo, useState } from 'react'
import { useAppSelector } from '../../hooks/useAppStore'
import type { NewsEvent } from '../../types/index'

// =============================================================================
// Session time ranges (UTC)
// =============================================================================
const SESSION_RANGES = [
  { id: 'asian', label: 'Asian', color: '#009688', start: 0, end: 9 }, // 00:00-09:00 UTC
  { id: 'london', label: 'London', color: '#2196f3', start: 8, end: 17 }, // 08:00-17:00 UTC
  { id: 'ny', label: 'NY', color: '#ff5722', start: 13, end: 22 }, // 13:00-22:00 UTC
] as const

// =============================================================================
// Helper: lay session hien tai theo hour (UTC)
// =============================================================================
function getCurrentSession(): string | null {
  const hour = new Date().getUTCHours()
  for (const s of SESSION_RANGES) {
    if (hour >= s.start && hour < s.end) return s.id
  }
  return null
}

// =============================================================================
// Helper: lay impact color
// =============================================================================
function impactColor(impact: string): string {
  switch (impact) {
    case 'High':
      return '#ef5350'
    case 'Medium':
      return '#ffa726'
    default:
      return '#9e9e9e'
  }
}

// =============================================================================
// Helper: format time (HH:MM UTC)
// =============================================================================
function formatEventTime(ts: number): string {
  const d = new Date(ts)
  return `${d.getUTCHours().toString().padStart(2, '0')}:${d.getUTCMinutes().toString().padStart(2, '0')}`
}

// =============================================================================
// Tooltip component
// =============================================================================
function EventTooltip({ event }: { event: NewsEvent }) {
  return (
    <div
      className="absolute z-20 bg-gray-900 border border-gray-700 rounded-lg p-3 shadow-xl text-xs whitespace-nowrap"
      style={{
        left: '50%',
        bottom: '100%',
        transform: 'translateX(-50%)',
        marginBottom: '8px',
        minWidth: '200px',
      }}
    >
      <div className="font-bold text-sm mb-1" style={{ color: impactColor(event.impact) }}>
        {event.impact}
      </div>
      <div className="text-gray-200 font-semibold mb-1">{event.name}</div>
      <div className="text-gray-400 mb-1">{event.currency} · {formatEventTime(event.scheduled_time)}</div>
      <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-gray-400">
        <span>Forecast:</span>
        <span className="text-gray-200 text-right">{event.forecast ?? '—'}</span>
        <span>Actual:</span>
        <span className="text-gray-200 text-right">{event.actual ?? '—'}</span>
        <span>S (Z-score):</span>
        <span
          className="text-right font-mono"
          style={{
            color:
              event.surprise_factor > 1
                ? '#26a69a'
                : event.surprise_factor < -1
                  ? '#ef5350'
                  : '#9e9e9e',
          }}
        >
          {event.surprise_factor.toFixed(2)}
        </span>
      </div>
      {event.surprise_direction !== 0 && (
        <div className="mt-1 text-xs text-gray-500">
          Surprise: {event.surprise_direction > 0 ? '↑ Positive' : '↓ Negative'}
        </div>
      )}
    </div>
  )
}

// =============================================================================
// MacroTimeline Component
// =============================================================================
const MacroTimeline = memo(function MacroTimeline() {
  const events = useAppSelector((state) => state.macro.events)
  const [hoveredEventId, setHoveredEventId] = useState<string | null>(null)

  // Lay current hour de xac dinh session background
  const currentSession = useMemo(() => getCurrentSession(), [])

  // Sap xep events theo scheduled_time
  const sortedEvents = useMemo(() => {
    return [...events].sort((a, b) => a.scheduled_time - b.scheduled_time)
  }, [events])

  // Xac dinh pre-news windows
  const now = Date.now()

  const isInPreNewsWindow = useMemo(() => {
    return sortedEvents.some((e) => {
      const diff = e.scheduled_time - now
      return diff > -5 * 60 * 1000 && diff < 15 * 60 * 1000 // -5min to +15min
    })
  }, [sortedEvents, now])

  if (sortedEvents.length === 0) return null

  return (
    <div className="bg-gray-800 rounded-lg p-3 relative overflow-hidden">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Macro Timeline
      </h3>

      {/* Timeline container */}
      <div className="relative h-20 overflow-x-auto overflow-y-hidden">
        {/* Session backgrounds */}
        {SESSION_RANGES.map((session) => (
          <div
            key={session.id}
            className="absolute inset-y-0"
            style={{
              left: `${(session.start / 24) * 100}%`,
              width: `${((session.end - session.start) / 24) * 100}%`,
              backgroundColor: currentSession === session.id ? `${session.color}15` : 'transparent',
              borderLeft: currentSession === session.id ? `2px solid ${session.color}30` : undefined,
              borderRight: currentSession === session.id ? `2px solid ${session.color}30` : undefined,
            }}
          />
        ))}

        {/* Pre-News shading */}
        {isInPreNewsWindow && (
          <div
            className="absolute inset-y-0"
            style={{
              left: '0',
              right: '0',
              backgroundColor: 'rgba(255, 167, 38, 0.08)',
            }}
          />
        )}

        {/* Event markers */}
        <div className="relative w-full h-full flex items-end pb-1">
          {sortedEvents.map((event) => {
            const d = new Date(event.scheduled_time)
            const hour = d.getUTCHours() + d.getUTCMinutes() / 60
            const leftPct = (hour / 24) * 100

            const isPast = event.scheduled_time < now
            const borderStyle = isPast ? 'solid' : 'dashed'
            const color = impactColor(event.impact)

            // Tinh toan pre-news window (-15min +5min)
            const inPreNewsWindow =
              event.scheduled_time - now > 0 &&
              event.scheduled_time - now < 15 * 60 * 1000

            return (
              <div
                key={event.id}
                className="absolute bottom-0 flex flex-col items-center cursor-pointer transition-all duration-200"
                style={{ left: `${leftPct}%`, transform: 'translateX(-50%)' }}
                onMouseEnter={() => setHoveredEventId(event.id)}
                onMouseLeave={() => setHoveredEventId(null)}
              >
                {/* Vertical line */}
                <div
                  className="w-0.5 transition-all duration-200"
                  style={{
                    height: hoveredEventId === event.id ? '100%' : '70%',
                    borderLeft: `2px ${borderStyle} ${color}`,
                    opacity: isPast ? 0.6 : 1,
                  }}
                />

                {/* Dot */}
                <div
                  className="w-2 h-2 rounded-full mt-0.5"
                  style={{
                    backgroundColor: color,
                    opacity: isPast ? 0.6 : 1,
                  }}
                />

                {/* Tooltip */}
                {hoveredEventId === event.id && <EventTooltip event={event} />}

                {/* Pre-news indicator */}
                {inPreNewsWindow && (
                  <div
                    className="absolute bottom-full mb-5 text-xs font-bold"
                    style={{ color: '#ffa726' }}
                  >
                    ⚡
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* Legend */}
      <div className="flex gap-3 mt-2 text-xs text-gray-500">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-gray-400" /> Low
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-yellow-500" /> Medium
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-red-500" /> High
        </span>
        <span className="flex items-center gap-1 ml-2">
          <span className="w-3 border-t border-dashed border-gray-400" /> Future
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 border-t border-solid border-gray-400" /> Past
        </span>
      </div>
    </div>
  )
})

export default MacroTimeline
