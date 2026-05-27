// =============================================================================
// AgentDebatePanel.tsx — Debate panel gauge -4 to +4
// Hien thi consensus rating voi gauge bar va evidence
// =============================================================================

import { memo } from 'react'
import { useAppSelector } from '../../hooks/useAppStore'

// =============================================================================
// AgentDebatePanel Component
// =============================================================================
const AgentDebatePanel = memo(function AgentDebatePanel() {
  const debate = useAppSelector((state) => state.aiState.debate)

  // Khong co debate — hien thi placeholder
  if (!debate) {
    return (
      <div className="bg-gray-800 rounded-lg p-3">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
          Agent Debate
        </h3>
        <p className="text-gray-500 text-xs italic">Waiting for consensus...</p>
      </div>
    )
  }

  const { consensus, bull, bear } = debate
  const rating = consensus.rating // -4 to +4

  // Tinh phan tram gauge (-4 -> 0 -> +4)
  const gaugePercent = ((rating + 4) / 8) * 100
  const isBullish = rating > 0
  const isBearish = rating < 0

  // Mau sac
  const gaugeColor = isBullish ? '#26a69a' : isBearish ? '#ef5350' : '#ffa726'

  return (
    <div className="bg-gray-800 rounded-lg p-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Agent Debate
      </h3>

      {/* Rating gauge -4 to +4 */}
      <div className="mb-3">
        <div className="flex justify-between text-xs text-gray-500 mb-1">
          <span>Bearish -4</span>
          <span className="font-bold text-sm" style={{ color: gaugeColor }}>
            {rating > 0 ? `+${rating}` : rating}
          </span>
          <span>Bullish +4</span>
        </div>

        {/* Gauge background */}
        <div className="relative h-3 bg-gray-700 rounded-full overflow-hidden">
          {/* Center marker */}
          <div className="absolute left-1/2 top-0 bottom-0 w-0.5 bg-gray-500 z-10" />

          {/* Fill bar */}
          <div
            className="absolute top-0 bottom-0 transition-all duration-500 ease-out rounded-full"
            style={{
              width: `${Math.abs(gaugePercent - 50) * 2}%`,
              left: isBearish ? `${gaugePercent}%` : '50%',
              backgroundColor: gaugeColor,
              opacity: 0.8,
            }}
          />
        </div>

        <div className="flex justify-between text-xs mt-1">
          <span className="text-red-400">{bear.confidence.toFixed(0)}%</span>
          <span className="text-gray-500">{consensus.preferred_direction}</span>
          <span className="text-green-400">{bull.confidence.toFixed(0)}%</span>
        </div>
      </div>

      {/* Evidence bullets */}
      <div className="space-y-1 max-h-24 overflow-y-auto text-xs">
        {bull.evidence.slice(0, 2).map((ev, i) => (
          <div key={`bull-${i}`} className="flex items-start gap-1 text-green-400">
            <span className="mt-0.5">▲</span>
            <span className="text-gray-300">{ev.text.slice(0, 60)}</span>
          </div>
        ))}
        {bear.evidence.slice(0, 2).map((ev, i) => (
          <div key={`bear-${i}`} className="flex items-start gap-1 text-red-400">
            <span className="mt-0.5">▼</span>
            <span className="text-gray-300">{ev.text.slice(0, 60)}</span>
          </div>
        ))}
      </div>

      {/* Consensus qualifier */}
      <div className="mt-2 flex items-center justify-between text-xs">
        <span className="text-gray-500">
          Agreement: {(consensus.agreement_score * 100).toFixed(0)}%
        </span>
        <span
          className="px-1.5 py-0.5 rounded text-xs font-medium"
          style={{
            backgroundColor:
              consensus.confidence_qualifier === 'HIGH'
                ? 'rgba(38, 166, 154, 0.2)'
                : consensus.confidence_qualifier === 'MEDIUM'
                  ? 'rgba(255, 167, 38, 0.2)'
                  : 'rgba(239, 83, 80, 0.2)',
            color:
              consensus.confidence_qualifier === 'HIGH'
                ? '#26a69a'
                : consensus.confidence_qualifier === 'MEDIUM'
                  ? '#ffa726'
                  : '#ef5350',
          }}
        >
          {consensus.confidence_qualifier}
        </span>
      </div>
    </div>
  )
})

export default AgentDebatePanel
