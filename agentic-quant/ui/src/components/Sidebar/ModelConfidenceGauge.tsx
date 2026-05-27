// =============================================================================
// ModelConfidenceGauge.tsx — Stacked bar P_BSL + P_lateral + P_SSL
// Hien thi phan tram xac suat duoi dang stacked horizontal bar
// =============================================================================

import { memo } from 'react'
import { useAppSelector } from '../../hooks/useAppStore'

// =============================================================================
// ModelConfidenceGauge Component
// =============================================================================
const ModelConfidenceGauge = memo(function ModelConfidenceGauge() {
  const prediction = useAppSelector((state) => state.aiState.prediction)
  const modelDegraded = useAppSelector((state) => state.aiState.modelDegraded)

  // Neu khong co prediction, hien thi placeholder
  if (!prediction) {
    return (
      <div className="bg-gray-800 rounded-lg p-3">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
          AI Confidence
        </h3>
        <p className="text-gray-500 text-xs italic">Waiting for prediction...</p>
      </div>
    )
  }

  const pBsl = prediction.p_bsl
  const pLateral = prediction.p_lateral
  const pSsl = prediction.p_ssl

  // Tinh tong de normalization (dam bao 100%)
  const total = pBsl + pLateral + pSsl
  const bslPct = total > 0 ? (pBsl / total) * 100 : 0
  const lateralPct = total > 0 ? (pLateral / total) * 100 : 0
  const sslPct = total > 0 ? (pSsl / total) * 100 : 0

  // Xac dinh direction chinh
  const isBSLFavored = pBsl > pSsl && pBsl > pLateral
  const isSSLFavored = pSsl > pBsl && pSsl > pLateral
  const directionLabel = isBSLFavored ? 'BULLISH' : isSSLFavored ? 'BEARISH' : 'LATERAL'
  const directionColor = isBSLFavored ? '#26a69a' : isSSLFavored ? '#ef5350' : '#ffa726'

  return (
    <div className="bg-gray-800 rounded-lg p-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        AI Confidence
      </h3>

      {/* Direction label */}
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-bold" style={{ color: directionColor }}>
          {directionLabel}
        </span>
        <span
          className="text-xs px-1.5 py-0.5 rounded"
          style={{
            backgroundColor:
              prediction.confidence_qualifier === 'HIGH'
                ? 'rgba(38, 166, 154, 0.2)'
                : prediction.confidence_qualifier === 'MEDIUM'
                  ? 'rgba(255, 167, 38, 0.2)'
                  : 'rgba(239, 83, 80, 0.2)',
            color:
              prediction.confidence_qualifier === 'HIGH'
                ? '#26a69a'
                : prediction.confidence_qualifier === 'MEDIUM'
                  ? '#ffa726'
                  : '#ef5350',
          }}
        >
          {prediction.confidence_qualifier}
        </span>
      </div>

      {/* Stacked bar */}
      <div className="h-4 w-full rounded-full overflow-hidden flex bg-gray-700">
        <div
          className="h-full transition-all duration-500 ease-out"
          style={{
            width: `${bslPct}%`,
            backgroundColor: '#42a5f5', // BSL blue
          }}
          title={`P_BSL: ${(pBsl * 100).toFixed(1)}%`}
        />
        <div
          className="h-full transition-all duration-500 ease-out"
          style={{
            width: `${lateralPct}%`,
            backgroundColor: '#78909c', // Lateral gray
          }}
          title={`P_Lateral: ${(pLateral * 100).toFixed(1)}%`}
        />
        <div
          className="h-full transition-all duration-500 ease-out"
          style={{
            width: `${sslPct}%`,
            backgroundColor: '#ef5350', // SSL red
          }}
          title={`P_SSL: ${(pSsl * 100).toFixed(1)}%`}
        />
      </div>

      {/* Labels */}
      <div className="flex justify-between text-xs mt-1">
        <span className="text-blue-400">BSL {(pBsl * 100).toFixed(0)}%</span>
        <span className="text-gray-400">Lat {(pLateral * 100).toFixed(0)}%</span>
        <span className="text-red-400">SSL {(pSsl * 100).toFixed(0)}%</span>
      </div>

      {/* Model degraded warning */}
      {modelDegraded && (
        <div className="mt-2 text-xs text-yellow-400 flex items-center gap-1">
          <span>⚠</span>
          <span>Model degraded — predictions may be unreliable</span>
        </div>
      )}
    </div>
  )
})

export default ModelConfidenceGauge
