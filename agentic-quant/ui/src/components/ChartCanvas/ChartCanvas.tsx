// =============================================================================
// ChartCanvas.tsx — TradingView-style chart su dung lightweight-charts
// Candlestick series + Volume histogram subchart
// Dark theme (#1a1a2e background, #ddd text)
// Responsive resize
// =============================================================================

import { useEffect, useRef, memo } from 'react'
import { createChart, type IChartApi, type ISeriesApi, type CandlestickData, type HistogramData, type Time } from 'lightweight-charts'
import { useAppSelector } from '../../hooks/useAppStore'
import type { OHLCV } from '../../types/index'

// =============================================================================
// Theme colors
// =============================================================================
const THEME = {
  background: '#1a1a2e',
  text: '#ddd',
  gridLines: '#2a2a3e',
  crosshair: '#555',
  wickUp: '#26a69a',
  wickDown: '#ef5350',
  candleUp: '#26a69a',
  candleDown: '#ef5350',
  volumeUp: 'rgba(38, 166, 154, 0.5)',
  volumeDown: 'rgba(239, 83, 80, 0.5)',
} as const

// =============================================================================
// ChartCanvas Component
// =============================================================================
const ChartCanvas = memo(function ChartCanvas() {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null)

  const bars = useAppSelector((state) => state.chart.bars)

  // --- Khoi tao chart ---
  useEffect(() => {
    if (!containerRef.current) return

    const container = containerRef.current
    const chart = createChart(container, {
      layout: {
        background: { color: THEME.background },
        textColor: THEME.text,
      },
      grid: {
        vertLines: { color: THEME.gridLines },
        horzLines: { color: THEME.gridLines },
      },
      crosshair: {
        vertLine: { color: THEME.crosshair, labelBackgroundColor: THEME.background },
        horzLine: { color: THEME.crosshair, labelBackgroundColor: THEME.background },
      },
      timeScale: {
        borderColor: THEME.gridLines,
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: THEME.gridLines,
      },
      width: container.clientWidth,
      height: container.clientHeight,
      handleScroll: { vertTouchDrag: false },
    })

    chartRef.current = chart

    // Candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: THEME.candleUp,
      downColor: THEME.candleDown,
      borderUpColor: THEME.candleUp,
      borderDownColor: THEME.candleDown,
      wickUpColor: THEME.wickUp,
      wickDownColor: THEME.wickDown,
      priceFormat: {
        type: 'price',
        precision: 5,
        minMove: 0.00001,
      },
    })
    candleSeriesRef.current = candleSeries

    // Volume histogram subchart
    const volumeSeries = chart.addHistogramSeries({
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: 'volume',
    })
    chart.priceScale('volume').applyOptions({
      scaleMargins: {
        top: 0.85,
        bottom: 0,
      },
    })
    volumeSeriesRef.current = volumeSeries

    // Responsive resize
    const handleResize = () => {
      if (containerRef.current && chartRef.current) {
        const { clientWidth, clientHeight } = containerRef.current
        chartRef.current.applyOptions({
          width: clientWidth,
          height: clientHeight,
        })
      }
    }

    const resizeObserver = new ResizeObserver(handleResize)
    resizeObserver.observe(container)

    return () => {
      resizeObserver.disconnect()
      chart.remove()
      chartRef.current = null
      candleSeriesRef.current = null
      volumeSeriesRef.current = null
    }
  }, [])

  // --- Cap nhat data khi bars thay doi ---
  useEffect(() => {
    if (!candleSeriesRef.current || !volumeSeriesRef.current) return

    const candleData: CandlestickData[] = bars.map((bar: OHLCV) => ({
      time: (bar.timestamp / 1000) as Time,
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close,
    }))

    const volumeData: HistogramData[] = bars.map((bar: OHLCV) => ({
      time: (bar.timestamp / 1000) as Time,
      value: bar.volume,
      color: bar.close >= bar.open ? THEME.volumeUp : THEME.volumeDown,
    }))

    candleSeriesRef.current.setData(candleData)
    volumeSeriesRef.current.setData(volumeData)
  }, [bars])

  return (
    <div
      ref={containerRef}
      className="w-full h-full"
      style={{ minHeight: '400px' }}
    />
  )
})

export default ChartCanvas
