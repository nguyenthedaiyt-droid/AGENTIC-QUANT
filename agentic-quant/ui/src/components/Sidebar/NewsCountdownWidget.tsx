// =============================================================================
// NewsCountdownWidget.tsx — Countdown timer den su kien tin tiep theo
// Red blink khi < 5 phut
// =============================================================================

import { memo, useState, useEffect } from 'react'
import { useAppSelector } from '../../hooks/useAppStore'

// =============================================================================
// Helper: format seconds -> MM:SS
// =============================================================================
function formatCountdown(seconds: number): string {
  if (seconds <= 0) return '00:00'
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
}

// =============================================================================
// NewsCountdownWidget Component
// =============================================================================
const NewsCountdownWidget = memo(function NewsCountdownWidget() {
  const countdowns = useAppSelector((state) => state.macro.countdowns)
  const regimePhase = useAppSelector((state) => state.macro.regimePhase)

  // Lay countdown dau tien (gan nhat)
  const countdownEntries = Object.entries(countdowns)
  const [localSeconds, setLocalSeconds] = useState<number>(0)
  const [nextEventName, setNextEventName] = useState<string>('')

  // Cap nhat local countdown moi giay
  useEffect(() => {
    if (countdownEntries.length === 0) return

    const [currency, state] = countdownEntries[0]
    setLocalSeconds(state.seconds_to_next)
    setNextEventName(state.next_event?.name ?? `${currency} event`)

    const interval = setInterval(() => {
      setLocalSeconds((prev) => {
        if (prev <= 0) return 0
        return prev - 1
      })
    }, 1000)

    return () => clearInterval(interval)
  }, [countdowns])

  // Neu khong co countdown
  if (countdownEntries.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-3">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
          News Countdown
        </h3>
        <p className="text-gray-500 text-xs italic">No upcoming events</p>
      </div>
    )
  }

  const [currency] = countdownEntries[0]
  const isBlinking = localSeconds > 0 && localSeconds < 300 // < 5 minutes
  const isUrgent = localSeconds <= 60 // < 1 minute
  const isPast = localSeconds <= 0

  // Regime phase label
  const phaseLabel =
    regimePhase === 'PRE_NEWS'
      ? 'PRE-NEWS'
      : regimePhase === 'NEWS_WINDOW'
        ? '⚠ NEWS'
        : regimePhase === 'POST_NEWS'
          ? 'POST-NEWS'
          : ''

  return (
    <div className="bg-gray-800 rounded-lg p-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        News Countdown
      </h3>

      <div className="text-center">
        {/* Currency */}
        <span className="text-sm font-bold text-gray-300">{currency}</span>

        {/* Countdown number */}
        <div
          className={`text-2xl font-mono font-bold mt-1 transition-colors duration-300 ${
            isBlinking ? 'text-red-400 animate-pulse' : isUrgent ? 'text-red-500' : isPast ? 'text-gray-500' : 'text-gray-200'
          }`}
          style={{ animation: isBlinking ? 'blink-red 1s ease-in-out infinite' : undefined }}
        >
          {isPast ? 'LIVE' : formatCountdown(localSeconds)}
        </div>

        {/* Event name */}
        <div className="text-xs text-gray-400 mt-1 truncate">{nextEventName}</div>

        {/* Regime phase */}
        {phaseLabel && (
          <span
            className="inline-block mt-2 px-2 py-0.5 rounded text-xs font-bold"
            style={{
              backgroundColor:
                regimePhase === 'NEWS_WINDOW'
                  ? 'rgba(239, 83, 80, 0.3)'
                  : 'rgba(255, 167, 38, 0.2)',
              color:
                regimePhase === 'NEWS_WINDOW' ? '#ef5350' : '#ffa726',
            }}
          >
            {phaseLabel}
          </span>
        )}
      </div>
    </div>
  )
})

export default NewsCountdownWidget
