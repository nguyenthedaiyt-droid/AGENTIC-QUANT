# =============================================================================
# AGENTIC-QUANT — Macro Calendar Engine Orchestrator
# Ket noi CalendarScraper + VolatilityCountdown + NewsVectorizer + RegimeClassifier
# =============================================================================

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from core.macro import (
    CalendarScraper,
    EconomicCalendarDB,
    RawNewsEvent,
    ImpactLevel,
    NewsVectorizer,
    NewsImpactScore,
    VolatilityCountdown,
    MacroState,
    CountdownEvent,
    PostNewsRegimeClassifier,
    NewsOutcome,
    PostNewsRegime,
)
from core.utils.events.bus import EventBus
from core.utils.events.types import (
    EventType,
    NewsAlertEvent,
    NewsImpact,
    RegimeChangeEvent,
    GuardrailActivatedEvent,
    GuardrailDeactivatedEvent,
    StalenessAlertEvent,
)

_IMPACT_TO_NEWS: dict[str, NewsImpact] = {
    "LOW": NewsImpact.LOW,
    "MEDIUM": NewsImpact.MEDIUM,
    "HIGH": NewsImpact.HIGH,
}

if TYPE_CHECKING:
    pass


# =============================================================================
# Macro Engine Config
# =============================================================================
@dataclass
class MacroEngineConfig:
    """Cau hinh Macro Calendar Engine."""

    # Scheduler intervals
    refresh_interval_seconds: int = 21600   # 6 tieng
    pre_event_refresh_seconds: int = 300    # 5 phut khi con 30 phut toi event
    pre_event_warning_seconds: int = 1800   # 30 phut toi event -> tang tan suat

    # Currencies to track
    currencies: list[str] = field(default_factory=lambda: ["USD", "XAU"])

    # Minimum impact
    min_impact: ImpactLevel = ImpactLevel.MEDIUM

    # Paths
    db_path: str | Path = "data/economic_calendar.db"

    # Feature flags
    enable_event_reschedule_detection: bool = True
    enable_event_cancellation_detection: bool = True
    enable_cluster_amplification: bool = True
    cluster_amplification_factor: float = 1.25
    reschedule_threshold_minutes: int = 30

    # Performance
    max_loop_time_ms: float = 10.0


# =============================================================================
# MacroEngine — Main Orchestrator
# =============================================================================
class MacroEngine:
    """
    Orchestrator cho toan bo Macro Calendar Engine.

    Tong hop cac component:
    - CalendarScraper: Lay du lieu lich kinh te (6h / 5min)
    - VolatilityCountdown: Dem nguoc thoi gian den event, phat state change
    - NewsVectorizer: Tinh I_news, Surprise Factor
    - PostNewsRegimeClassifier: Phan loai thi truong sau tin

    Su kien phat ra tren Event Bus:
    - NEWS_ALERT: Khi co event High Impact sap toi
    - GUARDRAIL_ACTIVATED / GUARDRAIL_DEACTIVATED: Khi guardrail thay doi
    - REGIME_CHANGE: Khi macro regime thay doi
    - CALENDAR_STALE: Khi scraping fail

    Args:
        config: Cau hinh engine
        event_bus: EventBus instance (optional)
    """

    def __init__(
        self,
        config: MacroEngineConfig | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._config = config or MacroEngineConfig()
        self._bus = event_bus
        self._running = False
        self._tasks: set[asyncio.Task[None]] = set()

        # Components
        self._db: EconomicCalendarDB = EconomicCalendarDB(self._config.db_path)
        self._scraper: CalendarScraper = CalendarScraper(
            currencies=self._config.currencies,
            min_impact=self._config.min_impact,
            db=self._db,
        )
        self._countdown: VolatilityCountdown = VolatilityCountdown(
            pre_news_seconds=900,
            news_window_seconds=300,
            post_news_seconds=1800,
            high_only_guardrail=True,
        )
        self._vectorizer: NewsVectorizer = NewsVectorizer(
            alpha=0.4,
            surprise_threshold=2.0,
            db=self._db,
        )
        self._classifier: PostNewsRegimeClassifier = PostNewsRegimeClassifier(
            db_path=self._config.db_path,
        )

        # State tracking for edge cases
        self._known_event_times: dict[str, datetime] = {}
        self._last_scrape: datetime | None = None
        self._is_stale: bool = False
        self._reschedule_detected: bool = False
        self._cancellation_pending: set[str] = set()

        # Current state
        self._current_macro_state: MacroState = MacroState.NORMAL
        self._guardrail_active: bool = False
        self._current_i_news: float = 0.0
        self._next_refresh_time: datetime | None = None
        self._scrape_count: int = 0

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    async def start(self) -> None:
        """Khoi dong engine: connect DB, start components."""
        logger.info("MacroEngine starting...")
        self._db.connect()
        await self._scraper.start()

        # Lay cache tu DB truoc
        await self._load_cache_from_db()
        await self._countdown.start()

        self._running = True

        # Chay scheduler + countdown loop
        self._tasks.add(asyncio.create_task(self._scheduler_loop()))
        self._tasks.add(asyncio.create_task(self._countdown_watcher()))

        # Scrape ngay lap tuc
        await self._scrape_and_update()

        logger.info("MacroEngine started, {n} tasks running", n=len(self._tasks))

    async def stop(self) -> None:
        """Dung engine."""
        logger.info("MacroEngine stopping...")
        self._running = False

        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        await self._scraper.stop()
        self._db.close()
        logger.info("MacroEngine stopped")

    async def _load_cache_from_db(self) -> None:
        """Load su kien tu DB cache khi khoi dong."""
        now = datetime.now(timezone.utc)
        future = now + timedelta(hours=72)
        events = self._db.get_upcoming_events(
            from_time=now,
            to_time=future,
            currencies=self._config.currencies,
            min_impact=self._config.min_impact,
        )
        if events:
            self._countdown.set_events(events)
            for e in events:
                self._known_event_times[e.event_id] = e.scheduled_time
            logger.info("Loaded {n} events from DB cache", n=len(events))

    # -------------------------------------------------------------------------
    # Scheduler Loop
    # -------------------------------------------------------------------------
    async def _scheduler_loop(self) -> None:
        """
        Vong lap scheduler: scrape dinh ky, tang tan suat khi gan event.

        Normal: 6 gio
        Pre-event (30 phut): 5 phut
        """
        logger.info(
            "Scheduler loop started: normal={n}s, pre_event={p}s",
            n=self._config.refresh_interval_seconds,
            p=self._config.pre_event_refresh_seconds,
        )

        while self._running:
            try:
                # Tinh interval hien tai
                interval = self._get_current_refresh_interval()
                self._next_refresh_time = datetime.now(timezone.utc) + timedelta(seconds=interval)

                # Scrape
                await self._scrape_and_update()

                # Sleep
                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Scheduler loop error")
                self._is_stale = True
                if self._bus:
                    age = int((datetime.now(timezone.utc) - self._last_scrape).total_seconds()) if self._last_scrape else 0
                    self._bus.publish(
                        StalenessAlertEvent(
                            feed_name="CALENDAR",
                            last_update_age_seconds=age,
                            threshold_seconds=self._config.refresh_interval_seconds,
                        )
                    )
                await asyncio.sleep(60)  # Retry sau 1 phut

    def _get_current_refresh_interval(self) -> int:
        """
        Tra ve interval scrape hien tai.

        Tang tan suat khi con 30 phut toi High Impact event.
        """
        next_high = self._countdown.get_next_high_impact()
        if next_high:
            now = datetime.now(timezone.utc)
            seconds_to = int(next_high.scheduled_time.timestamp() - now.timestamp())
            if 0 < seconds_to <= self._config.pre_event_warning_seconds:
                return self._config.pre_event_refresh_seconds

        return self._config.refresh_interval_seconds

    # -------------------------------------------------------------------------
    # Scrape & Update
    # -------------------------------------------------------------------------
    async def _scrape_and_update(self) -> None:
        """Scrape lich kinh te, update countdown va DB."""
        try:
            events = await self._scraper.scrape()
            self._last_scrape = datetime.now(timezone.utc)
            self._is_stale = False
            self._scrape_count += 1

            if events:
                # Update countdown
                self._countdown.set_events(events)

                # Update known event times for reschedule detection
                for e in events:
                    self._known_event_times[e.event_id] = e.scheduled_time

                # Update ATR from config
                self._vectorizer.set_atr_d1(10.0)

                # Kiem tra edge cases
                self._detect_reschedule(events)
                self._detect_cancellations(events)

                # Phat NEWS_ALERT cho High Impact events trong 24h
                self._emit_news_alerts(events)

                logger.info(
                    "MacroEngine: scraped {n} events, total={total}",
                    n=len(events),
                    total=self._scrape_count,
                )
            else:
                logger.warning("MacroEngine: scrape returned 0 events")

        except Exception:
            logger.exception("MacroEngine: scrape failed")
            self._is_stale = True
            if self._bus:
                age = int((datetime.now(timezone.utc) - self._last_scrape).total_seconds()) if self._last_scrape else 0
                self._bus.publish(
                    StalenessAlertEvent(
                        feed_name="CALENDAR",
                        last_update_age_seconds=age,
                        threshold_seconds=self._config.refresh_interval_seconds,
                    )
                )

    # -------------------------------------------------------------------------
    # Countdown Watcher — Bridge countdown state changes to Event Bus
    # -------------------------------------------------------------------------
    async def _countdown_watcher(self) -> None:
        """
        Lang nghe countdown state changes, phat len Event Bus.

        Khi actual xuat hien trong scrape -> trigger NEWS_WINDOW detection.
        """
        async def on_countdown_update(countdown: CountdownEvent) -> None:
            await self._handle_countdown_update(countdown)

        self._countdown.on_state_change(on_countdown_update)

        while self._running:
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Countdown watcher error")

    async def _handle_countdown_update(self, countdown: CountdownEvent) -> None:
        """Xu ly countdown tick — phat event, update state."""
        # Guardrail change
        if countdown.active_guardrail != self._guardrail_active:
            self._guardrail_active = countdown.active_guardrail
            if self._bus:
                if self._guardrail_active:
                    self._bus.publish(
                        GuardrailActivatedEvent(
                            guardrail_type="pre_news",
                            event_id=countdown.current_event.event_id if countdown.current_event else None,
                            seconds_remaining=countdown.seconds_to_next,
                            dampening_factor=countdown.dampening_factor,
                        )
                    )
                else:
                    self._bus.publish(GuardrailDeactivatedEvent(guardrail_type="pre_news_passed"))

        # State change
        if countdown.state != self._current_macro_state:
            old_state = self._current_macro_state
            self._current_macro_state = countdown.state

            if self._bus:
                self._bus.publish(
                    RegimeChangeEvent(
                        previous_regime=old_state.value,
                        new_regime=countdown.state.value,
                        trigger_reason=countdown.current_event.event_id if countdown.current_event else None,
                    )
                )

        # NEWS_WINDOW: actual vua xuat hien (2.3.3)
        if countdown.state == MacroState.NEWS_WINDOW:
            await self._on_news_window(countdown)

        # Tinh I_news hien tai
        self._current_i_news = self._compute_current_i_news(countdown)

    async def _on_news_window(self, countdown: CountdownEvent) -> None:
        """
        NEWS_WINDOW detected — event bat dau.

        Kiem tra actual co trong DB:
        - Co: tinh I_news voi surprise factor, phat NEWS_ALERT
        - Khong: chua co actual, chi notify PRE_NEWS
        """
        event = countdown.current_event
        if not event:
            return

        # Kiem tra actual
        db_event = self._db.get_upcoming_events(
            from_time=datetime.now(timezone.utc) - timedelta(hours=1),
            to_time=datetime.now(timezone.utc) + timedelta(hours=1),
            currencies=[event.currency],
        )
        actual_event = next(
            (e for e in db_event if e.event_id == event.event_id and e.actual is not None),
            None,
        )

        if actual_event and actual_event.actual is not None:
            # Da co actual -> tinh I_news voi surprise
            score = self._vectorizer.vectorize(actual_event)

            # Update DB
            self._db.update_event_actual(
                event_id=actual_event.event_id,
                actual=actual_event.actual,
                is_surprise=score.is_surprise,
                surprise_z=score.surprise_z,
            )

            if self._bus:
                self._bus.publish(
                    NewsAlertEvent(
                        event_id=event.event_id,
                        title=event.title,
                        currency=event.currency,
                        impact=_IMPACT_TO_NEWS.get(event.impact.value, NewsImpact.MEDIUM),
                        actual=actual_event.actual,
                        forecast=event.forecast,
                        i_news=score.i_news,
                        surprise_z=score.surprise_z,
                        surprise_direction=score.surprise_direction,
                    )
                )

            logger.info(
                "NEWS_WINDOW: {title} — actual={a}, I_news={i:.3f}, S={s:.2f}",
                title=event.title,
                a=actual_event.actual,
                i=score.i_news,
                s=score.surprise_z or 0,
            )
        else:
            # Chua co actual
            logger.debug(
                "NEWS_WINDOW: {title} (waiting for actual)",
                title=event.title,
            )

    def _compute_current_i_news(self, countdown: CountdownEvent) -> float:
        """Tinh I_news hien tai cho strategy engine."""
        if countdown.current_event:
            score = self._vectorizer.vectorize(countdown.current_event)
            dampened = self._vectorizer.adjust_i_news(
                score.i_news,
                active_guardrail=countdown.active_guardrail,
            )
            return dampened
        return 0.0

    # -------------------------------------------------------------------------
    # Edge Case Handlers
    # -------------------------------------------------------------------------
    def _detect_reschedule(self, events: list[RawNewsEvent]) -> None:
        """
        D.2: Event reschedule > 30 phut.

        Neu event da biet nhung thoi gian thay doi > 30 phut
        -> reset guardrail, recalculate.
        """
        if not self._config.enable_event_reschedule_detection:
            return

        for event in events:
            if event.event_id in self._known_event_times:
                old_time = self._known_event_times[event.event_id]
                diff = abs((event.scheduled_time - old_time).total_seconds())

                if diff > self._config.reschedule_threshold_minutes * 60:
                    logger.warning(
                        "Event rescheduled: {title} — diff={m:.1f}min",
                        title=event.title,
                        m=diff / 60,
                    )

                    # Reset guardrail vi event da dich
                    if self._guardrail_active:
                        self._guardrail_active = False
                        if self._bus:
                            self._bus.publish(
                                GuardrailDeactivatedEvent(guardrail_type="event_rescheduled")
                            )
                        logger.info("Guardrail reset due to event reschedule")

                    # Update known time
                    self._known_event_times[event.event_id] = event.scheduled_time
                    self._reschedule_detected = True

    def _detect_cancellations(self, events: list[RawNewsEvent]) -> None:
        """
        D.3: Event cancellation.

        Neu event da biet nhung khong con trong scrape moi
        -> xoa khoi queue, reset neu dang PRE_NEWS.
        """
        if not self._config.enable_event_cancellation_detection:
            return

        current_ids = {e.event_id for e in events}
        for event_id, old_time in list(self._known_event_times.items()):
            if event_id not in current_ids:
                # Kiem tra xem co phai event trong tuong lai gan khong
                now = datetime.now(timezone.utc)
                if old_time > now - timedelta(hours=1):
                    logger.warning(
                        "Event cancelled: {id}, was scheduled at {t}",
                        id=event_id,
                        t=old_time.isoformat(),
                    )
                    # Xoa khoi countdown
                    self._countdown._events = [
                        e for e in self._countdown._events
                        if e.event_id != event_id
                    ]
                    del self._known_event_times[event_id]

                    # Reset state if this was the PRE_NEWS event
                    if self._current_macro_state in (MacroState.PRE_NEWS, MacroState.NEWS_WINDOW):
                        self._guardrail_active = False
                        self._current_macro_state = MacroState.NORMAL
                        if self._bus:
                            self._bus.publish(
                                GuardrailDeactivatedEvent(guardrail_type="event_cancelled")
                            )
                        logger.info("Guardrail reset due to event cancellation")

    # -------------------------------------------------------------------------
    # News Alerts
    # -------------------------------------------------------------------------
    def _emit_news_alerts(self, events: list[RawNewsEvent]) -> None:
        """Phat NEWS_ALERT cho High Impact events sap toi."""
        if not self._bus:
            return

        now = datetime.now(timezone.utc)
        for event in events:
            if event.impact == ImpactLevel.HIGH and event.scheduled_time > now:
                seconds_to = int(event.scheduled_time.timestamp() - now.timestamp())
                score = self._vectorizer.vectorize(event)

                self._bus.publish(
                    NewsAlertEvent(
                        event_id=event.event_id,
                        title=event.title,
                        currency=event.currency,
                        impact=_IMPACT_TO_NEWS.get(event.impact.value, NewsImpact.MEDIUM),
                        scheduled_time=event.scheduled_time,
                        seconds_to_event=seconds_to,
                        forecast=event.forecast,
                        i_news=score.i_news,
                    )
                )

    # -------------------------------------------------------------------------
    # Public API — for Strategy Engine
    # -------------------------------------------------------------------------
    def get_current_macro_state(self) -> MacroState:
        """Lay trang thai macro hien tai."""
        return self._current_macro_state

    def is_guardrail_active(self) -> bool:
        """Kiem tra guardrail co dang active khong."""
        return self._guardrail_active

    def get_i_news(self) -> float:
        """Lay I_news hien tai."""
        return self._current_i_news

    def get_upcoming_events(
        self,
        hours: int = 24,
        min_impact: ImpactLevel = ImpactLevel.MEDIUM,
    ) -> list[RawNewsEvent]:
        """Lay cac su kien sap toi."""
        now = datetime.now(timezone.utc)
        return self._db.get_upcoming_events(
            from_time=now,
            to_time=now + timedelta(hours=hours),
            currencies=self._config.currencies,
            min_impact=min_impact,
        )

    def vectorize_event(self, event: RawNewsEvent) -> NewsImpactScore:
        """Vectorize mot event (cho strategy engine goi)."""
        return self._vectorizer.vectorize(event)

    def classify_outcome(
        self,
        event_id: str,
        price_at_event: float,
        price_5min_after: float,
        atr_h1: float,
        surprise_direction: str,
        surprise_z: float | None = None,
    ) -> NewsOutcome:
        """
        Phan loai thi truong sau khi tin ra (2.4.1).

        Sau khi classify, tu dong:
        - Luu outcome vao DB
        - Update news_historical_volatility (2.4.3)
        """
        outcome = self._classifier.classify(
            event_id=event_id,
            price_at_event=price_at_event,
            price_5min_after=price_5min_after,
            atr_h1=atr_h1,
            surprise_direction=surprise_direction,
            surprise_z=surprise_z,
        )

        # Luu vao DB
        saved_id = self._classifier.save_outcome(outcome)
        if saved_id:
            outcome.outcome_id = saved_id

        # Update volatility (2.4.3)
        if surprise_z is not None:
            self._db.update_surprise_sigma(
                event_id=event_id,
                currency="USD",
                event_type="GENERAL",
                new_surprise=surprise_z,
            )

        return outcome

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    @property
    def countdown(self) -> VolatilityCountdown:
        return self._countdown

    @property
    def vectorizer(self) -> NewsVectorizer:
        return self._vectorizer

    @property
    def db(self) -> EconomicCalendarDB:
        return self._db

    @property
    def is_stale(self) -> bool:
        return self._is_stale

    @property
    def scrape_count(self) -> int:
        return self._scrape_count

    @property
    def last_scrape(self) -> datetime | None:
        return self._last_scrape

    @property
    def next_refresh(self) -> datetime | None:
        return self._next_refresh_time
