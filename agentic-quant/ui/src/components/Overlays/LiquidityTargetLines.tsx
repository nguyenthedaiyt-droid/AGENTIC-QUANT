// =============================================================================
// LiquidityTargetLines.tsx — BSL/SSL dashed lines
// BSL: blue dashed 'BSL ↑ {P_BSL}%', pulse khi P > 0.70
// SSL: red dashed 'SSL ↓ {P_SSL}%'
// =============================================================================

import { memo } from 'react'
import { useAppSelector } from '../../hooks/useAppStore'

// =============================================================================
// LiquidityTargetLines Component
// =============================================================================
const LiquidityTargetLines = memo(function LiquidityTargetLines() {
  const liquidityTargets = useAppSelector((state) => state.chart.liquidityTargets)
  const prediction = useAppSelector((state) => state.aiState.prediction)

  if (liquidityTargets.length === 0 && !prediction) return null

  return (
    <div className="absolute inset-0 pointer-events-none" style={{ zIndex: 12 }}>
      {/* BSL lines */}
      {liquidityTargets
        .filter((t) => t.target_type === 'BSL')
        .map((target, i) => {
          const isPulsing = target.p_probability > 0.7
          return (
            <div
              key={`bsl-${i}`}
              className="absolute left-0 right-0"
              style={{
                // Trong thuc te, can price-to-y mapping
                top: '30%',
                borderTop: '2px dashed #42a5f5',
                opacity: isPulsing ? 1 : 0.7,
                animation: isPulsing ? 'pulse-bsl 1s ease-in-out infinite' : undefined,
              }}
            >
              <span
                className="absolute right-0 -top-4 text-xs font-bold whitespace-nowrap"
                style={{
                  color: '#42a5f5',
                  textShadow: '0 0 4px rgba(0,0,0,0.8)',
                }}
              >
                BSL ↑ {(target.p_probability * 100).toFixed(0)}%
              </span>
            </div>
          )
        })}

      {/* SSL lines */}
      {liquidityTargets
        .filter((t) => t.target_type === 'SSL')
        .map((target, i) => (
          <div
            key={`ssl-${i}`}
            className="absolute left-0 right-0"
            style={{
              // Trong thuc te, can price-to-y mapping
              top: '70%',
              borderTop: '2px dashed #ef5350',
              opacity: 0.7,
            }}
          >
            <span
              className="absolute right-0 -top-4 text-xs font-bold whitespace-nowrap"
              style={{
                color: '#ef5350',
                textShadow: '0 0 4px rgba(0,0,0,0.8)',
              }}
            >
              SSL ↓ {(target.p_probability * 100).toFixed(0)}%
            </span>
          </div>
        ))}

      {/* Prediction BSL/SSL tu ModelAPrediction */}
      {prediction && (
        <>
          <div
            className="absolute left-0 right-0"
            style={{
              top: '20%',
              borderTop: '2px dashed #42a5f5',
              opacity: prediction.p_bsl > 0.7 ? 1 : 0.6,
              animation: prediction.p_bsl > 0.7 ? 'pulse-bsl 1s ease-in-out infinite' : undefined,
            }}
          >
            <span className="absolute right-0 -top-4 text-xs font-bold whitespace-nowrap" style={{ color: '#42a5f5', textShadow: '0 0 4px rgba(0,0,0,0.8)' }}>
              P_BSL ↑ {(prediction.p_bsl * 100).toFixed(0)}%
            </span>
          </div>
          <div
            className="absolute left-0 right-0"
            style={{
              top: '80%',
              borderTop: '2px dashed #ef5350',
              opacity: 0.6,
            }}
          >
            <span className="absolute right-0 -top-4 text-xs font-bold whitespace-nowrap" style={{ color: '#ef5350', textShadow: '0 0 4px rgba(0,0,0,0.8)' }}>
              P_SSL ↓ {(prediction.p_ssl * 100).toFixed(0)}%
            </span>
          </div>
        </>
      )}
    </div>
  )
})

export default LiquidityTargetLines
