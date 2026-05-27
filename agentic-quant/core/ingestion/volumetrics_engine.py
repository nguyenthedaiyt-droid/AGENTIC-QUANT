# =============================================================================
# AGENTIC-QUANT — Volumetrics Engine
# Tinh CVD, OBI, III, Divergence Score
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from core.utils.events import (
    EventBus,
    EventType,
    TickReceivedEvent,
    BarCloseEvent,
    VolumetricsUpdateEvent,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# ATR Calculator (Wilder's Smoothing)
# =============================================================================
class ATRCalculator:
    """
    ATR tinh theo Wilder's Smoothed Moving Average.

    ATR_t = (ATR_{t-1} × (n-1) + TR_t) / n
    TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)
    """

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self._values: list[float] = []
        self._atr: float | None = None
        self._prev_close: float | None = None

    def update(self, high: float, low: float, close: float) -> float:
        """Cap nhat ATR voi mot bar moi, tra ve ATR hien tai."""
        if self._prev_close is not None:
            tr = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close),
            )
        else:
            tr = high - low

        self._values.append(tr)
        self._prev_close = close

        if len(self._values) < self.period:
            # True Range don gian trong giai doan khoi tao
            return sum(self._values) / len(self._values)

        if self._atr is None:
            # Khoi tao: SMA
            self._atr = sum(self._values[-self.period :]) / self.period
        else:
            # Wilder's smoothing
            self._atr = (self._atr * (self.period - 1) + tr) / self.period

        return self._atr

    @property
    def atr(self) -> float:
        """Gia tri ATR hien tai (0 neu chua du du lieu)."""
        return self._atr or 0.0


# =============================================================================
# Volumetrics Engine
# =============================================================================
class VolumetricsEngine:
    """
    Tinh toan tat ca chỉ so volumetrics.

    Bao gom:
    - CVD (Cumulative Volume Delta)
    - CVD normalized
    - CVD rolling windows (MA5/10/20/50)
    - OBI (Order Book Imbalance) — neu co DOM data
    - III (Institutional Intensity Index)
    - Divergence Score

    Args:
        event_bus: EventBus instance
    """

    def __init__(
        self,
        event_bus: EventBus | None = None,
        atr_periods: dict[str, int] | None = None,
    ) -> None:
        self.event_bus = event_bus

        # ATR calculators per timeframe
        self._atr_periods = atr_periods or {
            "M1": 14, "M5": 14, "M15": 14,
            "H1": 14, "H4": 14, "D1": 14,
        }
        self._atr_calculators: dict[str, ATRCalculator] = {
            tf: ATRCalculator(period=period)
            for tf, period in self._atr_periods.items()
        }

        # State per symbol/timeframe
        self._state: dict[str, dict[str, "_VolumetricsState"]] = {}

        # Rolling windows cho CVD MA
        self._cvd_rolling_windows = [5, 10, 20, 50]

        # ATR average cho III
        self._atr_avg30: dict[str, list[float]] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def process_tick(self, event: TickReceivedEvent) -> VolumetricsUpdateEvent | None:
        """
        Xu ly tick, cap nhat volumetrics cho M1.

        Tra ve VolumetricsUpdateEvent hoac None neu tick bi loai bo.
        """
        if event.is_abnormal_spread:
            # Khong update CVD khi co spread bat thuong
            return None

        symbol = event.symbol or "XAUUSD"
        tf = "M1"

        state = self._get_state(symbol, tf)

        # Accumulate CVD
        delta = self._compute_delta(event)
        state.cvd += delta
        state.cvd_history.append(state.cvd)
        state.total_volume += event.volume

        # Rolling CVD MA
        cvd_ma = self._compute_cvd_ma(state.cvd_history)

        # CVD normalized
        cvd_norm = state.cvd / state.total_volume if state.total_volume > 0 else 0.0

        # III
        atr = self._atr_calculators.get(tf)
        atr_val = atr.atr if atr else 0.0

        # Tinh price change
        price_change = abs(event.last - state.prev_price) if state.prev_price > 0 else 0.0
        state.prev_price = event.last

        # III = (CVD / ATR_avg30) × (|ΔP| / ATR_current)
        self._update_atr_avg30(symbol, atr_val)
        atr_avg30 = sum(self._atr_avg30.get(symbol, [0.0])) / max(len(self._atr_avg30.get(symbol, [1])), 1)
        iii = 0.0
        if atr_avg30 > 0 and atr_val > 0:
            iii = (state.cvd / atr_avg30) * (price_change / atr_val)

        # Divergence Score
        # +1: price up + CVD up (confirmation)
        # -1: price up + CVD down (bearish divergence)
        # +1: price down + CVD down (confirmation)
        # -1: price down + CVD up (bullish divergence)
        prev_price = state.last_price
        state.last_price = event.last
        if prev_price > 0:
            price_dir = 1 if event.last > prev_price else -1
            cvd_dir = 1 if delta > 0 else -1
            div = price_dir * cvd_dir  # same = +1, diff = -1
        else:
            div = 0.0

        event_out = VolumetricsUpdateEvent(
            symbol=symbol,
            timeframe=tf,
            cvd=state.cvd,
            cvd_norm=cvd_norm,
            cvd_ma5=cvd_ma.get(5, 0.0),
            cvd_ma10=cvd_ma.get(10, 0.0),
            cvd_ma20=cvd_ma.get(20, 0.0),
            cvd_ma50=cvd_ma.get(50, 0.0),
            obi=0.0,  # OBI can DOM data, xu ly rieng
            iii=iii,
            divergence_score=div,
            bucket_time=event.timestamp_us // 1_000_000,
        )

        if self.event_bus:
            self.event_bus.publish(event_out)

        return event_out

    def process_bar_close(self, event: BarCloseEvent) -> None:
        """Cap nhat ATR khi bar dong, reset CVD cho bar moi."""
        symbol = event.symbol or "XAUUSD"
        tf = event.timeframe

        # Update ATR
        atr = self._atr_calculators.get(tf)
        if atr:
            atr.update(event.bar_high, event.bar_low, event.bar_close)

        # Reset CVD cho bar tiep theo
        state = self._get_state(symbol, tf)
        state.cvd = 0.0
        state.cvd_history.clear()
        state.total_volume = 0.0

    def process_bar_close_for_tf(
        self,
        symbol: str,
        timeframe: str,
        bar: dict,
    ) -> None:
        """Cap nhat ATR cho timeframe bat ky (dung cho backtest)."""
        atr = self._atr_calculators.get(timeframe)
        if atr:
            atr.update(bar["high"], bar["low"], bar["close"])

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------
    def _get_state(self, symbol: str, tf: str) -> "_VolumetricsState":
        """Lay hoac tao state cho symbol/timeframe."""
        if symbol not in self._state:
            self._state[symbol] = {}
        if tf not in self._state[symbol]:
            self._state[symbol][tf] = _VolumetricsState()
        return self._state[symbol][tf]

    def _compute_delta(self, event: TickReceivedEvent) -> float:
        """
        Tinh Volume Delta.

        +volume neu BUY aggressor
        -volume neu SELL aggressor
        0 neu UNKNOWN
        """
        if event.aggressor == "BUY":
            return event.volume
        elif event.aggressor == "SELL":
            return -event.volume
        return 0.0

    def _compute_cvd_ma(self, history: list[float]) -> dict[int, float]:
        """Tinh CVD rolling moving averages."""
        result = {}
        for w in self._cvd_rolling_windows:
            if len(history) >= w:
                result[w] = sum(history[-w:]) / w
        return result

    def _update_atr_avg30(self, symbol: str, atr_val: float) -> None:
        """Duy tri rolling ATR 30-gia tri cho III."""
        if symbol not in self._atr_avg30:
            self._atr_avg30[symbol] = []
        self._atr_avg30[symbol].append(atr_val)
        if len(self._atr_avg30[symbol]) > 30:
            self._atr_avg30[symbol].pop(0)


# =============================================================================
# Internal State
# =============================================================================
@dataclass
class _VolumetricsState:
    """Trang thai volumetrics cho mot symbol/timeframe."""

    cvd: float = 0.0
    total_volume: float = 0.0
    prev_price: float = 0.0
    last_price: float = 0.0

    cvd_history: list[float] = field(default_factory=list)
    obi_history: list[float] = field(default_factory=list)
