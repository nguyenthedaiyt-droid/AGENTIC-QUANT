# =============================================================================
# AGENTIC-QUANT — MTF Synchronizer & Leakage Guard
# Dong goi Unified State Vector (USV) tu tick
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from loguru import logger

from core.ingestion import BarState, TIMEFRAME_SECONDS
from core.utils.events import TickReceivedEvent

if TYPE_CHECKING:
    pass


# =============================================================================
# Leakage Guard
# =============================================================================
class LeakageGuard:
    """
    Dam bao khong co look-ahead bias trong backtest mode.

    Co che hoat dong:
    - forward_locked: tap hop cac TF bi khoa trong backtest mode
    - Khi TF bi lock: close = None (bar chua dong)

    Trong real-time mode: forward_lock_enabled = False, khong khoa gi ca
    """

    def __init__(self, forward_lock_enabled: bool = True) -> None:
        self._forward_lock_enabled = forward_lock_enabled
        self._forward_locked: set[str] = set()
        self._lock_boundaries: dict[str, int] = {}

    def apply_guard(
        self,
        bars: dict[str, BarState],
        current_timestamp_sec: int,
    ) -> dict[str, BarState]:
        """
        Ap dung guard, tra ve ban sao bars voi cac TF bi khoa.

        Args:
            bars: Dict tat ca BarState
            current_timestamp_sec: Timestamp hien tai (giay)

        Returns:
            Bars voi close=None cho cac TF bi lock
        """
        if not self._forward_lock_enabled:
            return bars

        guarded = {}
        for tf, bar in bars.items():
            guarded_bar = _copy_bar(bar)

            # Kiem tra xem TF co bi khoa khong
            if tf in self._forward_locked:
                boundary = self._lock_boundaries.get(tf, 0)
                tf_sec = TIMEFRAME_SECONDS.get(tf, 60)
                bar_end = boundary + tf_sec

                # Neu tick hien tai vuot qua boundary -> unlock
                if current_timestamp_sec >= bar_end:
                    self._forward_locked.discard(tf)
                    guarded_bar.is_forward_locked = False
                else:
                    guarded_bar.is_forward_locked = True
                    guarded_bar.close = 0.0  # Che close trong backtest

            guarded[tf] = guarded_bar

        return guarded

    def lock_tf(self, tf: str, boundary_timestamp_sec: int) -> None:
        """Khoa mot timeframe bat ky (goi khi bar dong trong backtest)."""
        if not self._forward_lock_enabled:
            return

        self._forward_locked.add(tf)
        self._lock_boundaries[tf] = boundary_timestamp_sec
        logger.debug(
            "LeakageGuard: Locked {tf} at boundary {b}",
            tf=tf,
            b=boundary_timestamp_sec,
        )

    def unlock_all(self) -> None:
        """Mo khoa tat ca TF."""
        self._forward_locked.clear()
        self._lock_boundaries.clear()

    @property
    def is_forward_lock_enabled(self) -> bool:
        return self._forward_lock_enabled

    @property
    def locked_timeframes(self) -> set[str]:
        """Tap hop TF dang bi khoa."""
        return set(self._forward_locked)


# =============================================================================
# UnifiedStateVector
# =============================================================================
@dataclass
class UnifiedStateVector:
    """
    Dong goi toan bo trang thai thi truong tai mot thoi diem.

    Day la "single source of truth" cho AI Engine.

    Attributes:
        snapshot_time: Unix timestamp (microseconds) cua tick
        snapshot_time_sec: Unix timestamp (seconds)
        symbol: Symbol giao dich
        bars: Dict BarState cho 6 TF
        tick_context: Thong tin tick hien tai
        volatilities: ATR values cho moi TF
        leakage_guard_active: Co TF nao bi lock khong
        locked_timeframes: Tap hop TF bi lock
    """

    snapshot_time: int  # microseconds
    snapshot_time_sec: int  # seconds
    symbol: str

    # BarState cho 6 TF: M1, M5, M15, H1, H4, D1
    bars: dict[str, BarState] = field(default_factory=dict)

    # Tick context
    tick_price: float = 0.0
    tick_bid: float = 0.0
    tick_ask: float = 0.0
    tick_spread_pips: float = 0.0
    tick_aggressor: str = ""

    # Volatilities (ATR per TF)
    atr: dict[str, float] = field(default_factory=dict)

    # Leakage guard
    leakage_guard_active: bool = False
    locked_timeframes: set[str] = field(default_factory=set)

    # Optional: Volumetrics (CVD, III, etc.)
    cvd: float = 0.0
    cvd_norm: float = 0.0
    iii: float = 0.0
    divergence_score: float = 0.0

    # Optional: Macro context
    session_id: str = ""
    active_guardrail: bool = False
    macro_regime: str = "NORMAL"

    @property
    def current_price(self) -> float:
        """Gia hien tai (mid)."""
        return (self.tick_bid + self.tick_ask) / 2.0

    @property
    def is_backtest_mode(self) -> bool:
        return self.leakage_guard_active

    def get_closed_bar(self, tf: str) -> BarState | None:
        """Lay bar da dong cho TF, tra ve None neu bar dang hinh thanh."""
        bar = self.bars.get(tf)
        if bar is None:
            return None
        if self.leakage_guard_active and tf in self.locked_timeframes:
            return None
        if not bar.is_closed:
            return None
        return bar

    def get_closed_bars(self) -> dict[str, BarState]:
        """Lay tat ca bars da dong."""
        result = {}
        for tf, bar in self.bars.items():
            if tf in self.locked_timeframes:
                continue
            if bar.is_closed:
                result[tf] = bar
        return result

    def to_dict(self) -> dict:
        """Chuyen thanh dictionary (cho serialization)."""
        return {
            "snapshot_time": self.snapshot_time,
            "snapshot_time_sec": self.snapshot_time_sec,
            "symbol": self.symbol,
            "bars": {
                tf: {
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close if not (self.leakage_guard_active and tf in self.locked_timeframes) else None,
                    "volume": bar.volume,
                    "tick_count": bar.tick_count,
                    "bucket_time": bar.bucket_time,
                    "is_closed": bar.is_closed,
                }
                for tf, bar in self.bars.items()
            },
            "tick_price": self.tick_price,
            "atr": self.atr,
            "leakage_guard_active": self.leakage_guard_active,
            "locked_timeframes": list(self.locked_timeframes),
            "cvd": self.cvd,
            "iii": self.iii,
            "divergence_score": self.divergence_score,
            "session_id": self.session_id,
            "active_guardrail": self.active_guardrail,
            "macro_regime": self.macro_regime,
        }


# =============================================================================
# MTF Synchronizer
# =============================================================================
class MTFSynchronizer:
    """
    Dong goi Unified State Vector (USV) tu tick.

    Su dung chung voi OHLCVAggregator va VolumetricsEngine de tao
    USV day du cho AI Engine.

    Args:
        symbol: Symbol giao dich (default: XAUUSD)
        leakage_guard: LeakageGuard instance
    """

    def __init__(
        self,
        symbol: str = "XAUUSD",
        leakage_guard: LeakageGuard | None = None,
    ) -> None:
        self.symbol = symbol
        self._leakage_guard = leakage_guard or LeakageGuard()

        # Cached latest bars (from OHLCVAggregator)
        self._latest_bars: dict[str, BarState] = {}

        # ATR cache
        self._atr: dict[str, float] = {}

        # Volumetrics cache
        self._cvd: float = 0.0
        self._cvd_norm: float = 0.0
        self._iii: float = 0.0
        self._divergence: float = 0.0

        # Macro context
        self._session_id: str = ""
        self._active_guardrail: bool = False
        self._macro_regime: str = "NORMAL"

    def update_bar(self, timeframe: str, bar: BarState) -> None:
        """Cap nhat bar moi tu OHLCVAggregator."""
        self._latest_bars[timeframe] = bar

    def update_atr(self, timeframe: str, atr_value: float) -> None:
        """Cap nhat ATR."""
        self._atr[timeframe] = atr_value

    def update_volumetrics(
        self,
        cvd: float,
        cvd_norm: float,
        iii: float,
        divergence: float,
    ) -> None:
        """Cap nhat volumetrics."""
        self._cvd = cvd
        self._cvd_norm = cvd_norm
        self._iii = iii
        self._divergence = divergence

    def set_macro_context(
        self,
        session_id: str = "",
        active_guardrail: bool = False,
        macro_regime: str = "NORMAL",
    ) -> None:
        """Dat macro context."""
        self._session_id = session_id
        self._active_guardrail = active_guardrail
        self._macro_regime = macro_regime

    def build_usv(self, tick: TickReceivedEvent) -> UnifiedStateVector:
        """
        Xay dung USV tu tick hien tai.

        Ap dung Leakage Guard de dam bao khong co look-ahead.

        Args:
            tick: Tick hien tai

        Returns:
            UnifiedStateVector day du
        """
        symbol = tick.symbol or self.symbol
        ts_sec = tick.timestamp_us // 1_000_000

        # Apply leakage guard
        guarded_bars = self._leakage_guard.apply_guard(
            self._latest_bars,
            ts_sec,
        )

        usv = UnifiedStateVector(
            snapshot_time=tick.timestamp_us,
            snapshot_time_sec=ts_sec,
            symbol=symbol,
            bars=guarded_bars,
            tick_price=tick.last,
            tick_bid=tick.bid,
            tick_ask=tick.ask,
            tick_spread_pips=tick.spread_pips,
            tick_aggressor=tick.aggressor,
            atr=dict(self._atr),
            leakage_guard_active=len(self._leakage_guard.locked_timeframes) > 0,
            locked_timeframes=set(self._leakage_guard.locked_timeframes),
            cvd=self._cvd,
            cvd_norm=self._cvd_norm,
            iii=self._iii,
            divergence_score=self._divergence,
            session_id=self._session_id,
            active_guardrail=self._active_guardrail,
            macro_regime=self._macro_regime,
        )

        return usv

    def build_usv_from_tick_event(self, tick_event) -> UnifiedStateVector:
        """Build USV tu TickReceivedEvent instance."""
        return self.build_usv(tick_event)

    @property
    def leakage_guard(self) -> LeakageGuard:
        return self._leakage_guard


# =============================================================================
# Helpers
# =============================================================================
def _copy_bar(bar: BarState) -> BarState:
    """Tao mot shallow copy cua BarState."""
    return BarState(
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        volume=bar.volume,
        tick_count=bar.tick_count,
        bucket_time=bar.bucket_time,
        is_closed=bar.is_closed,
        opened_at_us=bar.opened_at_us,
        ticks=list(bar.ticks),
        buy_volume=bar.buy_volume,
        sell_volume=bar.sell_volume,
        is_forward_locked=getattr(bar, "is_forward_locked", False),
    )
