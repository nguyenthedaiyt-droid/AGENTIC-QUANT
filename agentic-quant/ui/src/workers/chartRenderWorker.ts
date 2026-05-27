// =============================================================================
// chartRenderWorker.ts — Web Worker for bar formatting & indicator calculation
// Nhan raw bar data tu main thread, xu ly:
//   - Format timestamp (unix seconds -> ms)
//   - Tinh toan indicators: SMA(20), EMA(12), Bollinger Bands
//   - Tra ve formatted bars + indicators
// Khong block main thread
// =============================================================================

// =============================================================================
// Types
// =============================================================================

/** Raw OHLCV bar tu backend */
interface RawBar {
  timestamp: number       // Unix seconds hoac ms
  open: number
  high: number
  low: number
  close: number
  volume: number
  tf?: string             // Timeframe label (M1, M5, ...)
  [key: string]: unknown  // Cho phep fields khac
}

/** Formatted OHLCV bar (timestamp * 1000 -> ms) */
interface FormattedBar {
  timestamp: number       // Unix ms
  open: number
  high: number
  low: number
  close: number
  volume: number
  tf?: string
}

/** Cac indicator duoc tinh toan */
interface Indicators {
  sma20: (number | null)[]   // SMA(20) — null cho bars dau
  ema12: (number | null)[]   // EMA(12) — null cho bars dau
  bbUpper: (number | null)[] // Bollinger Upper Band
  bbLower: (number | null)[] // Bollinger Lower Band
  bbMiddle: (number | null)[] // Bollinger Middle Band (= SMA20)
}

/** Ket qua tra ve cho main thread */
interface WorkerResult {
  type: 'bars_processed'
  bars: FormattedBar[]
  indicators: Indicators
  barCount: number
  processedAt: number       // Unix ms
}

/** Input nhan tu main thread */
interface WorkerInput {
  type: 'process_bars'
  bars: RawBar[]
  indicators?: {
    sma?: number      // Window cho SMA (default: 20)
    ema?: number      // Window cho EMA (default: 12)
    bb?: number       // Window cho Bollinger (default: 20)
    bbStdDev?: number // So standard deviation (default: 2)
  }
}

// =============================================================================
// Indicator Calculation Helpers
// =============================================================================

/**
 * Tinh Simple Moving Average (SMA).
 * @param data — Mang gia tri (close prices)
 * @param window — Kich thuoc window (default: 20)
 * @returns Mang SMA, null cho (window-1) phan tu dau
 */
function calcSMA(data: number[], window: number = 20): (number | null)[] {
  const result: (number | null)[] = []
  for (let i = 0; i < data.length; i++) {
    if (i < window - 1) {
      result.push(null)
    } else {
      let sum = 0
      for (let j = i - window + 1; j <= i; j++) {
        sum += data[j]
      }
      result.push(sum / window)
    }
  }
  return result
}

/**
 * Tinh Exponential Moving Average (EMA).
 * @param data — Mang gia tri (close prices)
 * @param window — Kich thuoc window (default: 12)
 * @returns Mang EMA, null cho (window-1) phan tu dau
 */
function calcEMA(data: number[], window: number = 12): (number | null)[] {
  const result: (number | null)[] = []
  if (data.length === 0) return result

  const multiplier = 2 / (window + 1)

  // Khoi tao: phan tu dau tien = SMA(window)
  let ema = data.slice(0, window).reduce((a, b) => a + b, 0) / window

  for (let i = 0; i < data.length; i++) {
    if (i < window - 1) {
      result.push(null)
    } else if (i === window - 1) {
      result.push(ema)
    } else {
      ema = (data[i] - ema) * multiplier + ema
      result.push(ema)
    }
  }
  return result
}

/**
 * Tinh Bollinger Bands.
 * @param data — Mang gia tri (close prices)
 * @param window — Kich thuoc window (default: 20)
 * @param stdDev — So standard deviation (default: 2)
 * @returns { upper, lower, middle } — null cho (window-1) phan tu dau
 */
function calcBollingerBands(
  data: number[],
  window: number = 20,
  stdDev: number = 2,
): {
  upper: (number | null)[]
  lower: (number | null)[]
  middle: (number | null)[]
} {
  const middle = calcSMA(data, window)
  const upper: (number | null)[] = []
  const lower: (number | null)[] = []

  for (let i = 0; i < data.length; i++) {
    if (middle[i] === null) {
      upper.push(null)
      lower.push(null)
    } else {
      // Tinh standard deviation trong window
      const start = i - window + 1
      const slice = data.slice(start, i + 1)
      const mean = middle[i] as number
      const squaredDiffs = slice.map((v) => (v - mean) ** 2)
      const variance =
        squaredDiffs.reduce((a, b) => a + b, 0) / slice.length
      const sd = Math.sqrt(variance)

      upper.push(mean + stdDev * sd)
      lower.push(mean - stdDev * sd)
    }
  }

  return { upper, lower, middle }
}

// =============================================================================
// Main Processing Function
// =============================================================================

/**
 * Format raw bars va tinh toan indicators.
 *
 * @param bars — Raw bar data tu backend
 * @param config — Indicator configuration
 * @returns WorkerResult
 */
function processBars(
  bars: RawBar[],
  config: WorkerInput['indicators'] = {},
): WorkerResult {
  const smaWindow = config?.sma ?? 20
  const emaWindow = config?.ema ?? 12
  const bbWindow = config?.bb ?? 20
  const bbStdDev = config?.bbStdDev ?? 2

  // --- Step 1: Format bars ---
  const formatted: FormattedBar[] = bars.map((bar) => ({
    timestamp:
      bar.timestamp < 1e12
        ? bar.timestamp * 1000   // Unix seconds -> ms
        : bar.timestamp,          // Da la ms
    open: bar.open,
    high: bar.high,
    low: bar.low,
    close: bar.close,
    volume: bar.volume,
    tf: bar.tf,
  }))

  // Sort theo timestamp (ascending)
  formatted.sort((a, b) => a.timestamp - b.timestamp)

  // --- Step 2: Tinh indicators ---
  const closePrices = formatted.map((b) => b.close)

  const sma20 = calcSMA(closePrices, smaWindow)
  const ema12 = calcEMA(closePrices, emaWindow)
  const bb = calcBollingerBands(closePrices, bbWindow, bbStdDev)

  const result: WorkerResult = {
    type: 'bars_processed',
    bars: formatted,
    indicators: {
      sma20,
      ema12,
      bbUpper: bb.upper,
      bbLower: bb.lower,
      bbMiddle: bb.middle,
    },
    barCount: formatted.length,
    processedAt: Date.now(),
  }

  return result
}

// =============================================================================
// Web Worker Message Handler
// =============================================================================

/**
 * Lang nghe message tu main thread.
 *
 * Input message format:
 * ```json
 * {
 *   "type": "process_bars",
 *   "bars": [...],
 *   "indicators": { "sma": 20, "ema": 12, "bb": 20, "bbStdDev": 2 }
 * }
 * ```
 *
 * Output message format:
 * ```json
 * {
 *   "type": "bars_processed",
 *   "bars": [...],
 *   "indicators": { "sma20": [...], "ema12": [...], ... },
 *   "barCount": 123,
 *   "processedAt": 1712345678901
 * }
 * ```
 */
self.onmessage = (event: MessageEvent<WorkerInput>) => {
  const { type, bars, indicators } = event.data

  if (type !== 'process_bars') {
    // Gui loi ve main thread
    self.postMessage({
      type: 'error',
      message: `Unknown message type: ${type}`,
      receivedAt: Date.now(),
    })
    return
  }

  if (!Array.isArray(bars) || bars.length === 0) {
    self.postMessage({
      type: 'error',
      message: 'bars phai la mot array khong empty',
      receivedAt: Date.now(),
    })
    return
  }

  try {
    const result = processBars(bars, indicators)
    self.postMessage(result)
  } catch (err) {
    self.postMessage({
      type: 'error',
      message: `Loi xu ly bars: ${err instanceof Error ? err.message : String(err)}`,
      receivedAt: Date.now(),
    })
  }
}

// =============================================================================
// Export type de main thread import
// =============================================================================

export type {
  RawBar,
  FormattedBar,
  Indicators,
  WorkerResult,
  WorkerInput,
}
