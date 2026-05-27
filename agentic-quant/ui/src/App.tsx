// =============================================================================
// App.tsx — AGENTIC-QUANT UI (Compose tat ca components)
// Layout: Chart (center) + Overlays + Sidebar (right) + Dashboard (top)
// =============================================================================

import { Provider } from 'react-redux'
import { store } from '@store/index'
import { useWebSocket } from '@hooks/useWebSocket'
import ChartCanvas from '@components/ChartCanvas/ChartCanvas'
import { MTFGhostZones, AIHeatmapOverlay, LiquidityTargetLines } from '@components/Overlays/index'
import { AgentDebatePanel, NewsCountdownWidget, KillzoneIndicator, ModelConfidenceGauge } from '@components/Sidebar/index'
import { LatencyIndicator, SystemStatusBanner } from '@components/Dashboard/index'
import MacroTimeline from '@components/MacroTimeline/MacroTimeline'

// =============================================================================
// CSS Animations (injected via style tag)
// =============================================================================
const animationStyles = `
@keyframes pulse-heat {
  0%, 100% { opacity: 0.6; }
  50% { opacity: 1; }
}
@keyframes pulse-bsl {
  0%, 100% { opacity: 0.7; }
  50% { opacity: 1; }
}
@keyframes blink-red {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}
`

// =============================================================================
// MainContent — Component ben trong Provider (co the useAppSelector)
// =============================================================================
function MainContent() {
  // Khoi tao WebSocket hook
  const { getLatency } = useWebSocket()

  return (
    <div className="min-h-screen bg-[#1a1a2e] text-[#ddd] flex flex-col overflow-hidden">
      {/* Inject animations */}
      <style>{animationStyles}</style>

      {/* ==================== TOP DASHBOARD ==================== */}
      <header className="flex items-center justify-between px-4 py-2 bg-gray-900 border-b border-gray-700 flex-shrink-0">
        {/* Left: Logo / Title */}
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-bold text-white tracking-tight">
            AGENTIC-QUANT
          </h1>
          <span className="text-xs text-gray-500 hidden sm:inline">
            Real-time AI-Powered Trading Dashboard
          </span>
        </div>

        {/* Right: Status indicators */}
        <div className="flex items-center gap-3">
          <LatencyIndicator getLatency={getLatency} />
        </div>
      </header>

      {/* System status banners */}
      <SystemStatusBanner />

      {/* ==================== MAIN LAYOUT ==================== */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: Chart area */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Chart + Overlays container */}
          <div className="relative flex-1 m-2 rounded-lg overflow-hidden bg-[#1a1a2e] border border-gray-700">
            {/* Chart core */}
            <ChartCanvas />

            {/* Overlays (pointer-events-none) */}
            <MTFGhostZones />
            <AIHeatmapOverlay />
            <LiquidityTargetLines />
          </div>

          {/* Macro Timeline */}
          <div className="mx-2 mb-2">
            <MacroTimeline />
          </div>
        </div>

        {/* Right: Sidebar */}
        <aside className="w-72 flex-shrink-0 overflow-y-auto p-2 space-y-2 border-l border-gray-700 bg-gray-900/50">
          {/* Killzone session indicator */}
          <KillzoneIndicator />

          {/* News countdown */}
          <NewsCountdownWidget />

          {/* AI Model confidence gauge */}
          <ModelConfidenceGauge />

          {/* Agent debate panel */}
          <AgentDebatePanel />
        </aside>
      </div>
    </div>
  )
}

// =============================================================================
// App — Root component voi Redux Provider
// =============================================================================
export default function App() {
  return (
    <Provider store={store}>
      <MainContent />
    </Provider>
  )
}
