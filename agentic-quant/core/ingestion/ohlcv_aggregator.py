# =============================================================================
# AGENTIC-QUANT — OHLCV Aggregator da khung thoi gian
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from loguru import logger

from core.utils.events import (
    EventBus,
    EventType,
    BarCloseEvent,
    BarUpdateEvent,
    TickReceivedEvent,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Timeframe Definitions
# =============================================================================
TIMEFRAME_SECONDS: dict[str, int] = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
}

# Thu tu uu tien de kiem tra cascade closure
CASCADE_ORDER = ["M1", "M5", "M15", "H1", "H4", "D1"]

# TF nho hon (LTF) de tinh ATR
LTF_TIMEFRAMES = ["M1", "M5", "M15"]
HTF_TIMEFRAMES = ["H1", "H4", "D1"]


@dataclass
class BarState:
    """
    Trang thai cua mot bar dang hinh thanh hoac da dong.

    Args:
        open: Gia mo cua
        high: Gia cao nhat
        low: Gia thap nhat
        close: Gia dong cua (None neu bar dang hinh thanh)
        volume: Khoi luong tong
        tick_count: So tick trong bar
        bucket_time: Unix timestamp cua bar (lam tron xuong)
        is_closed: Bar da dong chua
        opened_at_us: Microsecond timestamp khi bar mo
        ticks: List tick prices trong bar (dung cho volumetrics)
        buy_volume: Khoi luong mua
        sell_volume: Khoi luong ban
    """

    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    tick_count: int = 0
    bucket_time: int = 0
    is_closed: bool = False
    opened_at_us: int = 0
    is_forward_locked: bool = False  # Danh dau bar bi khoa boi LeakageGuard

    ticks: list[float] = field(default_factory=list)
    buy_volume: float = 0.0
    sell_volume: float = 0.0

    def update(self, price: float, volume: float, aggressor: str) -> None:
        """Cap nhat bar voi tick moi."""
        self.tick_count += 1
        self.close = price
        self.volume += volume

        # Update high/low
        if self.high == 0.0 or price > self.high:
            self.high = price
        if self.low == 0.0 or price < self.low:
            self.low = price

        # Update dau tien
        if self.tick_count == 1:
            self.open = price

        # Volumetrics
        self.ticks.append(price)
        if aggressor == "BUY":
            self.buy_volume += volume
        elif aggressor == "SELL":
            self.sell_volume += volume

    def to_bar_close(self, timeframe: str, symbol: str) -> BarCloseEvent:
        """Chuyen thanh BarCloseEvent."""
        return BarCloseEvent(
            symbol=symbol,
            timeframe=timeframe,
            bar_open=self.open,
            bar_high=self.high,
            bar_low=self.low,
            bar_close=self.close,
            bar_volume=self.volume,
            bucket_time=self.bucket_time,
            tick_count=self.tick_count,
        )

    def to_bar_update(self, timeframe: str, symbol: str) -> BarUpdateEvent:
        """Chuyen thanh BarUpdateEvent."""
        return BarUpdateEvent(
            symbol=symbol,
            timeframe=timeframe,
            bar_open=self.open,
            bar_high=self.high,
            bar_low=self.low,
            bar_close=self.close,
            bar_volume=self.volume,
            bucket_time=self.bucket_time,
            is_closed=self.is_closed,
        )


class OHLCVAggregator:
    """
    Tong hop OHLCV tu tick len 6 khung thoi gian.

    Xu ly cascade closure: M1 dong -> kiem tra M5, M15, H1, H4, D1.

    Args:
        event_bus: EventBus instance de publish BAR_UPDATE va BAR_CLOSE
    """

    def __init__(
        self,
        event_bus: EventBus | None = None,
        timeframes: list[str] | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.timeframes = timeframes or list(TIMEFRAME_SECONDS.keys())

        # Active bars: {symbol: {timeframe: {bucket_time: BarState}}}
        self._active_bars: dict[str, dict[str, dict[int, BarState]]] = {}

        # Counters
        self._bar_close_counts: dict[str, int] = {}
        self._total_ticks: int = 0

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def process_tick(self, event: TickReceivedEvent) -> list[BarCloseEvent]:
        """
        Xu ly mot tick, cap nhat tat ca khung thoi gian.

        Tra ve danh sach cac BarCloseEvent (neu co bar nao dong).

        Args:
            event: TickReceivedEvent tu EventBus

        Returns:
            Danh sach cac BarCloseEvent cho bar vua dong
        """
        self._total_ticks += 1
        symbol = event.symbol or "XAUUSD"

        # Dam bao cau truc data
        if symbol not in self._active_bars:
            self._active_bars[symbol] = {}

        closed_bars: list[BarCloseEvent] = []

        for tf in self.timeframes:
            close = self._update_tf(
                symbol=symbol,
                timeframe=tf,
                price=event.last,
                volume=event.volume,
                aggressor=event.aggressor,
                tick_timestamp_us=event.timestamp_us,
            )
            if close:
                closed_bars.append(close)

        # Cascade: khi M1 dong, kiem tra tat ca TF lon hon
        # Cac TF lon hon se tu dong kiem tra boundary trong _check_and_close_tf
        # Neu boundary chua den, ham do tra ve None
        if any(c.timeframe == "M1" for c in closed_bars):
            for tf in CASCADE_ORDER[1:]:  # Bo qua M1
                cascade = self._check_and_close_tf(
                    symbol=symbol,
                    timeframe=tf,
                    tick_timestamp_us=event.timestamp_us,
                )
                if cascade:
                    closed_bars.append(cascade)

        return closed_bars

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------
    def _update_tf(
        self,
        symbol: str,
        timeframe: str,
        price: float,
        volume: float,
        aggressor: str,
        tick_timestamp_us: int,
    ) -> BarCloseEvent | None:
        """Cap nhat hoac tao bar cho mot timeframe."""
        tf_seconds = TIMEFRAME_SECONDS[timeframe]
        bucket_time = self._compute_bucket(tick_timestamp_us, tf_seconds)

        if symbol not in self._active_bars:
            self._active_bars[symbol] = {}
        if timeframe not in self._active_bars[symbol]:
            self._active_bars[symbol][timeframe] = {}

        tf_bars = self._active_bars[symbol][timeframe]
        close_event: BarCloseEvent | None = None

        if bucket_time in tf_bars:
            bar = tf_bars[bucket_time]
            if bar.is_closed:
                # Tick chuyen sang bucket moi nhung bar cu chua dong
                # Dong bar cu
                close_event = self._close_bar(symbol, timeframe, bar.bucket_time, bar)
                # Tao bar moi
                new_bar = self._create_bar(
                    bucket_time=bucket_time,
                    price=price,
                    volume=volume,
                    aggressor=aggressor,
                    tick_timestamp_us=tick_timestamp_us,
                )
                tf_bars[bucket_time] = new_bar
                bar = new_bar
            else:
                bar.update(price, volume, aggressor)
        else:
            # Tao bar moi
            bar = self._create_bar(
                bucket_time=bucket_time,
                price=price,
                volume=volume,
                aggressor=aggressor,
                tick_timestamp_us=tick_timestamp_us,
            )
            tf_bars[bucket_time] = bar

        # Publish BAR_UPDATE event
        if self.event_bus:
            self.event_bus.publish(bar.to_bar_update(timeframe, symbol))

        return close_event

    def _check_and_close_tf(
        self,
        symbol: str,
        timeframe: str,
        tick_timestamp_us: int,
    ) -> BarCloseEvent | None:
        """Kiem tra va dong bar neu can (cascade closure)."""
        tf_seconds = TIMEFRAME_SECONDS[timeframe]
        bucket_time = self._compute_bucket(tick_timestamp_us, tf_seconds)

        if (
            symbol not in self._active_bars
            or timeframe not in self._active_bars[symbol]
            or bucket_time not in self._active_bars[symbol][timeframe]
        ):
            return None

        bar = self._active_bars[symbol][timeframe][bucket_time]
        if bar.is_closed:
            return None

        # Kiem tra thoi gian
        current_sec = tick_timestamp_us // 1_000_000
        bar_end_time = bucket_time + tf_seconds

        if current_sec >= bar_end_time:
            return self._close_bar(symbol, timeframe, bucket_time, bar)

        return None

    def _create_bar(
        self,
        bucket_time: int,
        price: float,
        volume: float,
        aggressor: str,
        tick_timestamp_us: int,
    ) -> BarState:
        """Tao mot BarState moi."""
        return BarState(
            open=price,
            high=price,
            low=price,
            close=price,
            volume=volume,
            tick_count=1,
            bucket_time=bucket_time,
            is_closed=False,
            opened_at_us=tick_timestamp_us,
            ticks=[price],
            buy_volume=volume if aggressor == "BUY" else 0.0,
            sell_volume=volume if aggressor == "SELL" else 0.0,
        )

    def _close_bar(
        self,
        symbol: str,
        timeframe: str,
        bucket_time: int,
        bar: BarState,
    ) -> BarCloseEvent:
        """Dong mot bar."""
        bar.is_closed = True
        bar.close = bar.close  # Already set by last update

        close_event = bar.to_bar_close(timeframe, symbol)

        # Publish BAR_CLOSE event
        if self.event_bus:
            self.event_bus.publish(close_event)

        # Update counter
        key = f"{symbol}:{timeframe}"
        self._bar_close_counts[key] = self._bar_close_counts.get(key, 0) + 1

        logger.debug(
            "Bar dong: {tf} @{bt} O={o:.2f} H={h:.2f} "
            "L={l:.2f} C={c:.2f} V={v:.0f} ticks={n}",
            tf=timeframe,
            bt=datetime.fromtimestamp(bucket_time, tz=timezone.utc).isoformat(),
            o=bar.open,
            h=bar.high,
            l=bar.low,
            c=bar.close,
            v=bar.volume,
            n=bar.tick_count,
        )

        return close_event

    def _compute_bucket(self, timestamp_us: int, tf_seconds: int) -> int:
        """Tinh bucket time cho tick."""
        sec = timestamp_us // 1_000_000
        return (sec // tf_seconds) * tf_seconds

    # -------------------------------------------------------------------------
    # Query API
    # -------------------------------------------------------------------------
    def get_latest_bar(
        self,
        symbol: str,
        timeframe: str,
    ) -> BarState | None:
        """Lay bar gan nhat cho mot symbol/timeframe."""
        if symbol not in self._active_bars:
            return None
        if timeframe not in self._active_bars[symbol]:
            return None

        bars = self._active_bars[symbol][timeframe]
        if not bars:
            return None

        # Lay bar co bucket_time lon nhat
        latest_bucket = max(bars.keys())
        return bars[latest_bucket]

    def get_bars_since(
        self,
        symbol: str,
        timeframe: str,
        since_timestamp: int,
        limit: int = 100,
    ) -> list[BarState]:
        """Lay tat ca bar da dong kể từ timestamp (cho backtest)."""
        if symbol not in self._active_bars:
            return []
        if timeframe not in self._active_bars[symbol]:
            return []

        bars = self._active_bars[symbol][timeframe]
        closed = [b for b in bars.values() if b.is_closed and b.bucket_time >= since_timestamp]
        closed.sort(key=lambda b: b.bucket_time)
        return closed[-limit:]

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    @property
    def total_ticks(self) -> int:
        """Tong so tick da xu ly."""
        return self._total_ticks

    def get_stats(self) -> dict:
        """Tra ve statistics."""
        return {
            "total_ticks": self._total_ticks,
            "bar_closes": dict(self._bar_close_counts),
            "symbols": list(self._active_bars.keys()),
        }
