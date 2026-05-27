# =============================================================================
# AGENTIC-QUANT — Volatility Countdown Timer
# Asyncio loop 1 giay, phat event PRE_NEWS/NEWS_WINDOW/POST_NEWS
# =============================================================================

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

from loguru import logger

from .calendar_scraper import RawNewsEvent, ImpactLevel

if TYPE_CHECKING:
    pass


# =============================================================================
# Macro State
# =============================================================================
class MacroState(str, Enum):
    """Trang thai macro hien tai."""

    NORMAL = "NORMAL"              # Khong co event sap toi
    PRE_NEWS = "PRE_NEWS"          # < 15 phut truoc event High Impact
    NEWS_WINDOW = "NEWS_WINDOW"    # 5 phut truoc -> 5 phut sau event
    POST_NEWS = "POST_NEWS"        # 5-30 phut sau event
    CLUSTER = "CLUSTER"            # 2+ High events trong 30 phut


@dataclass
class CountdownEvent:
    """Event phat boi countdown timer."""

    state: MacroState
    current_event: RawNewsEvent | None = None
    seconds_to_next: int = 0
    active_guardrail: bool = False
    dampening_factor: float = 1.0
    next_high_impact: RawNewsEvent | None = None
    all_events: list[RawNewsEvent] = field(default_factory=list)


class VolatilityCountdown:
    """
    Asyncio timer chay moi giay, theo doi su kien kinh te.

    Phat cac trang thai:
    - NORMAL: Khong co event
    - PRE_NEWS: < 15 phut truoc High Impact event
    - NEWS_WINDOW: Trong khoang event
    - POST_NEWS: 5-30 phut sau event
    - CLUSTER: Nhieu High Impact events gan nhau

    Args:
        pre_news_seconds: Giay truoc event de bat dau PRE_NEWS (default: 900 = 15 phut)
        news_window_seconds: Do rong NEWS_WINDOW (default: 300 = 5 phut)
        post_news_seconds: Do dai POST_NEWS (default: 1800 = 30 phut)
        high_only_guardrail: Chi High Impact moi kich hoat guardrail (default: True)
    """

    def __init__(
        self,
        pre_news_seconds: int = 900,
        news_window_seconds: int = 300,
        post_news_seconds: int = 1800,
        high_only_guardrail: bool = True,
    ) -> None:
        self._pre_news_seconds = pre_news_seconds
        self._news_window_seconds = news_window_seconds
        self._post_news_seconds = post_news_seconds
        self._high_only_guardrail = high_only_guardrail

        self._events: list[RawNewsEvent] = []
        self._current_event: RawNewsEvent | None = None
        self._in_news_window = False
        self._post_news_start: datetime | None = None
        self._current_state = MacroState.NORMAL
        self._dampening_factor = 1.0
        self._active_guardrail = False

        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._callbacks: list = []
        self._on_state_change_callback = None

        # Performance tracking
        self._loop_count = 0
        self._last_loop_time_ms = 0.0

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    async def start(self) -> None:
        """Bat dau countdown loop."""
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("VolatilityCountdown started")

    async def stop(self) -> None:
        """Dung countdown loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("VolatilityCountdown stopped")

    # -------------------------------------------------------------------------
    # Events Management
    # -------------------------------------------------------------------------
    def set_events(self, events: list[RawNewsEvent]) -> None:
        """Dat danh sach su kien (thuong goi sau scrape)."""
        # Giu tat ca event de POST_NEWS detection hoat dong
        self._events = sorted(events, key=lambda e: e.scheduled_time)
        logger.debug("Set {n} total events", n=len(self._events))

    def add_event(self, event: RawNewsEvent) -> None:
        """Them mot event."""
        self._events.append(event)
        self._events.sort(key=lambda e: e.scheduled_time)

    def get_next_high_impact(self) -> RawNewsEvent | None:
        """Lay su kien High Impact tiep theo (future only)."""
        now = datetime.now(timezone.utc)
        now_ts = now.timestamp()
        for e in self._events:
            if e.scheduled_time.timestamp() >= now_ts and e.impact == ImpactLevel.HIGH:
                return e
        return None

    def get_events_in_window(self, seconds: int) -> list[RawNewsEvent]:
        """Lay tat ca su kien trong khoang giay toi."""
        now = datetime.now(timezone.utc)
        future = now.timestamp() + seconds
        return [
            e for e in self._events
            if now.timestamp() <= e.scheduled_time.timestamp() <= future
        ]

    # -------------------------------------------------------------------------
    # State Change Callback
    # -------------------------------------------------------------------------
    def on_state_change(self, callback) -> None:
        """Dang ky callback khi state thay doi."""
        self._on_state_change_callback = callback

    # -------------------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------------------
    async def _loop(self) -> None:
        """Vong lap 1 giay."""
        while self._running:
            try:
                loop_start = asyncio.get_event_loop().time()

                # Tinh state moi
                countdown = self._tick()

                # Fire callback
                if self._on_state_change_callback:
                    await self._on_state_change_callback(countdown)

                # Fire registered callbacks
                for cb in self._callbacks:
                    try:
                        await cb(countdown)
                    except Exception:
                        logger.exception("Countdown callback error")

                # Performance tracking
                loop_end = asyncio.get_event_loop().time()
                self._last_loop_time_ms = (loop_end - loop_start) * 1000
                self._loop_count += 1

                # Log performance every 60 seconds
                if self._loop_count % 60 == 0 and self._last_loop_time_ms > 10:
                    logger.warning(
                        "Countdown loop slow: {ms:.1f}ms",
                        ms=self._last_loop_time_ms,
                    )

                await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Countdown loop error")
                await asyncio.sleep(1.0)

    def _tick(self) -> CountdownEvent:
        """Xu ly mot tick (1 giay)."""
        now = datetime.now(timezone.utc)
        now_ts = now.timestamp()

        # Tim event tiep theo va event vua qua
        next_event, last_past, seconds_since_past = self._find_next_event(now)
        seconds_to_next = 0
        if next_event:
            seconds_to_next = int(next_event.scheduled_time.timestamp() - now_ts)
        elif last_past:
            seconds_to_next = -seconds_since_past

        new_state = self._compute_state(
            now, next_event, last_past, seconds_since_past
        )

        # Tinh dampening factor
        dampening = self._compute_dampening(new_state, next_event, seconds_to_next)

        # Kiem tra state change
        if new_state != self._current_state:
            self._current_state = new_state
            logger.info(
                "Macro state change: {old} -> {new}, "
                "event={event}, seconds={s}, dampening={d}",
                old=self._current_state.value,
                new=new_state.value,
                event=next_event.title if next_event else "None",
                s=seconds_to_next,
                d=dampening,
            )

        # Update guardrail
        prev_guardrail = self._active_guardrail
        self._active_guardrail = (
            new_state == MacroState.PRE_NEWS
            or new_state == MacroState.NEWS_WINDOW
        ) and (not self._high_only_guardrail or (next_event and next_event.impact == ImpactLevel.HIGH))

        if prev_guardrail != self._active_guardrail:
            logger.info(
                "Guardrail {g}",
                g="ACTIVATED" if self._active_guardrail else "DEACTIVATED",
            )

        return CountdownEvent(
            state=new_state,
            current_event=next_event,
            seconds_to_next=seconds_to_next,
            active_guardrail=self._active_guardrail,
            dampening_factor=dampening,
            next_high_impact=self.get_next_high_impact(),
            all_events=self._events.copy(),
        )

    def _find_next_event(
        self,
        now: datetime,
    ) -> tuple[RawNewsEvent | None, RawNewsEvent | None, int]:
        """Tim su kien tiep theo va su kien vua qua.

        Returns:
            (next_event, last_past_event, seconds_since_last_past)
        """
        now_ts = now.timestamp()
        next_event: RawNewsEvent | None = None
        last_past: RawNewsEvent | None = None
        seconds_since_last = 0

        for e in self._events:
            e_ts = e.scheduled_time.timestamp()
            if e_ts >= now_ts:
                # Su kien tuong lai
                if next_event is None:
                    next_event = e
            else:
                # Su kien da qua - luu event gan nhat
                last_past = e
                seconds_since_last = int(now_ts - e_ts)

        return next_event, last_past, seconds_since_last

    def _compute_state(
        self,
        now: datetime,
        next_event: RawNewsEvent | None,
        last_past_event: RawNewsEvent | None,
        seconds_since_past: int,
    ) -> MacroState:
        """Tinh trang thai macro hien tai."""
        # Tim so high events trong 30 phut toi
        events_in_30min = self.get_events_in_window(1800)
        high_count = sum(1 for e in events_in_30min if e.impact == ImpactLevel.HIGH)

        # Cluster
        if high_count >= 2:
            return MacroState.CLUSTER

        # POST_NEWS: event vua qua trong 30 phut, khong co event sap toi
        if next_event is None and last_past_event is not None:
            if seconds_since_past <= self._post_news_seconds:
                return MacroState.POST_NEWS
            return MacroState.NORMAL

        # Khong co event nao
        if not next_event:
            return MacroState.NORMAL

        seconds_to_next = int(next_event.scheduled_time.timestamp() - now.timestamp())

        # NEWS_WINDOW: actual vua xuat hien hoac < 5 phut
        if self._in_news_window:
            if seconds_to_next > 0:
                return MacroState.NEWS_WINDOW
            self._in_news_window = False
            self._current_event = None

        # PRE_NEWS: < 15 phut truoc High Impact
        if seconds_to_next > 0 and seconds_to_next <= self._pre_news_seconds:
            if next_event.impact == ImpactLevel.HIGH:
                if seconds_to_next <= self._news_window_seconds:
                    self._in_news_window = True
                    self._current_event = next_event
                    return MacroState.NEWS_WINDOW
                return MacroState.PRE_NEWS

        # POST_NEWS: 5-30 phut sau event
        if seconds_to_next < 0 and abs(seconds_to_next) <= self._post_news_seconds:
            return MacroState.POST_NEWS

        return MacroState.NORMAL

    def _compute_dampening(
        self,
        state: MacroState,
        event: RawNewsEvent | None,
        seconds_to_next: int,
    ) -> float:
        """
        Tinh dampening factor cho pre-news/news-window.

        Gamma(t) = gamma_0 × (1 - max(0, t_to_news) / 900)
        """
        if state == MacroState.NEWS_WINDOW:
            return 0.1
        elif state == MacroState.PRE_NEWS:
            if seconds_to_next > 0:
                ratio = seconds_to_next / self._pre_news_seconds
                return 0.3 * ratio + 0.1
            return 0.3
        elif state == MacroState.CLUSTER:
            return 0.1
        return 1.0

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    @property
    def state(self) -> MacroState:
        return self._current_state

    @property
    def active_guardrail(self) -> bool:
        return self._active_guardrail

    @property
    def dampening_factor(self) -> float:
        return self._dampening_factor

    @property
    def current_event(self) -> RawNewsEvent | None:
        return self._current_event

    @property
    def loop_performance_ms(self) -> float:
        return self._last_loop_time_ms

    @property
    def is_running(self) -> bool:
        return self._running
