# =============================================================================
# AGENTIC-QUANT — Event-Driven Simulator (Phase 8)
# Replay tick data tu Parquet files, inject calendar events, publish events
# =============================================================================

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from core.ingestion import TickFrame
from core.utils.events import (
    EventBus,
    EventType,
    NewsAlertEvent,
    TickReceivedEvent,
    NewsImpact,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Constants
# =============================================================================
_DEFAULT_TICK_DIR = "data/ticks/parquet"
_MAX_REPLAY_SPEED = 0.0  # 0 = max speed (khong sleep)


# =============================================================================
# Event-Driven Simulator
# =============================================================================
class EventDrivenSimulator:
    """Simulator replay tick data tu Parquet files voi EventBus.

    Tinh nang:
    - Replay ticks theo dung timestamp (wall-clock order)
    - LeakageGuard ACTIVE = True (dam bao khong look-ahead bias)
    - Inject calendar events tu MacroEngine theo thoi gian
    - Publish TICK_RECEIVED, NEWS_ALERT vao EventBus
    - Speed target: 1 nam tick data trong < 2 gio (~700K ticks/phut)

    Args:
        event_bus: EventBus instance de publish events
        tick_dir: Thu muc chua Parquet tick files
        leakage_guard_active: Kich hoat LeakageGuard (default: True)
    """

    def __init__(
        self,
        event_bus: EventBus | None = None,
        tick_dir: str | Path = _DEFAULT_TICK_DIR,
        leakage_guard_active: bool = True,
    ) -> None:
        self._bus = event_bus
        self._tick_dir = Path(tick_dir)
        self._leakage_guard_active = leakage_guard_active

        # Performance tracking
        self._ticks_replayed: int = 0
        self._events_injected: int = 0
        self._start_wall: float = 0.0
        self._end_wall: float = 0.0

        # Calendar events de inject
        self._calendar_events: list[dict[str, Any]] = []

        # Running flag
        self._running = False

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    @property
    def leakage_guard_active(self) -> bool:
        """LeakageGuard co dang active khong."""
        return self._leakage_guard_active

    @leakage_guard_active.setter
    def leakage_guard_active(self, value: bool) -> None:
        self._leakage_guard_active = value
        logger.info(f"LeakageGuard: {'ACTIVE' if value else 'INACTIVE'}")

    async def run(
        self,
        symbol: str,
        start: str,
        end: str,
        speed: float = _MAX_REPLAY_SPEED,
    ) -> dict[str, Any]:
        """Replay ticks tu Parquet files.

        Args:
            symbol: Ma symbol (VD: XAUUSD)
            start: Ngay bat dau (YYYY-MM-DD)
            end: Ngay ket thuc (YYYY-MM-DD)
            speed: Toc do replay (seconds delay giua cac tick, 0 = max)

        Returns:
            Dict chua statistics cua lan chay nay

        Raises:
            FileNotFoundError: Neu khong tim thay Parquet file
        """
        self._running = True
        self._ticks_replayed = 0
        self._events_injected = 0
        self._start_wall = asyncio.get_event_loop().time()

        logger.info(
            "[EventDrivenSimulator] Bat dau replay: {symbol} [{start} -> {end}], "
            "LeakageGuard={leakage}",
            symbol=symbol,
            start=start,
            end=end,
            leakage=self._leakage_guard_active,
        )

        # --- Load tick data tu Parquet ---
        ticks = await self._load_ticks(symbol, start, end)
        if not ticks:
            logger.warning(f"Khong co tick data cho {symbol} [{start} -> {end}]")
            self._running = False
            return self._stats()

        # --- Load calendar events cung period ---
        await self._load_calendar_events(symbol, start, end)

        logger.info(
            "Da load {n_ticks} ticks, {n_events} calendar events",
            n_ticks=len(ticks),
            n_events=len(self._calendar_events),
        )

        # --- Replay ticks theo timestamp ---
        cal_idx = 0
        n_cal = len(self._calendar_events)

        for i, tick in enumerate(ticks):
            if not self._running:
                logger.info("EventDrivenSimulator bi stop giua chung")
                break

            tick_ts = tick.get("time", tick.get("timestamp", 0))
            tick_ts_us = int(tick_ts) if tick_ts < 1_000_000_000_000 else int(tick_ts)
            tick_sec = tick_ts_us // 1_000_000

            # --- Inject calendar events truoc tick (neu co) ---
            while cal_idx < n_cal:
                cal_event = self._calendar_events[cal_idx]
                cal_time = cal_event.get("scheduled_time", 0)
                if isinstance(cal_time, datetime):
                    cal_sec = int(cal_time.timestamp())
                else:
                    cal_sec = int(cal_time)

                if cal_sec <= tick_sec:
                    await self._inject_calendar_event(cal_event)
                    cal_idx += 1
                else:
                    break

            # --- Publish tick event ---
            tick_event = self._build_tick_event(symbol, tick, tick_ts_us)
            if self._bus:
                self._bus.publish(tick_event)

            self._ticks_replayed += 1

            # Speed control (neu speed > 0)
            if speed > 0 and i % 100 == 0:
                await asyncio.sleep(speed)

            # Yield control sau moi 10K ticks de tranh block event loop
            if i % 10_000 == 0:
                await asyncio.sleep(0)

        # --- Inject calendar events con lai sau tick cuoi ---
        while cal_idx < n_cal:
            await self._inject_calendar_event(self._calendar_events[cal_idx])
            cal_idx += 1

        self._end_wall = asyncio.get_event_loop().time()
        self._running = False

        elapsed = self._end_wall - self._start_wall
        rate = self._ticks_replayed / elapsed if elapsed > 0 else 0
        logger.info(
            "[EventDrivenSimulator] Hoan thanh: {n} ticks trong {elapsed:.2f}s "
            "({rate:.0f} ticks/s). Event injected: {ev}",
            n=self._ticks_replayed,
            elapsed=elapsed,
            rate=rate,
            ev=self._events_injected,
        )

        return self._stats()

    async def stop(self) -> None:
        """Dung simulator dang chay."""
        self._running = False
        logger.info("EventDrivenSimulator: stop signal nhan")

    # -------------------------------------------------------------------------
    # Internal: Loading
    # -------------------------------------------------------------------------
    async def _load_ticks(
        self,
        symbol: str,
        start: str,
        end: str,
    ) -> list[dict[str, Any]]:
        """Load tick data tu Parquet files.

        Implement xu ly doc Parquet bang PyArrow/pandas.
        Neu file khong ton tai, thu fallback path.

        Args:
            symbol: Ma symbol
            start: Ngay bat dau
            end: Ngay ket thuc

        Returns:
            List dict tick data theo thu tu thoi gian
        """
        try:
            import pandas as pd
            import pyarrow.parquet as pq
        except ImportError:
            logger.error(
                "Thieu thu vien pandas/pyarrow. "
                "Cai dat: pip install pandas pyarrow"
            )
            return []

        pattern = f"{symbol}_{start}_{end}.parquet"
        parquet_path = self._tick_dir / pattern

        # Fallback: tim file theo symbol
        if not parquet_path.exists():
            alt_patterns = [
                self._tick_dir / f"{symbol}.parquet",
                self._tick_dir / f"{symbol}_{start}_to_{end}.parquet",
            ]
            for alt in alt_patterns:
                if alt.exists():
                    parquet_path = alt
                    break
            else:
                logger.warning(
                    "Khong tim thay Parquet file cho {symbol}. "
                    "Da thu: {paths}",
                    symbol=symbol,
                    paths=[str(p) for p in [parquet_path] + alt_patterns],
                )
                return []

        try:
            table = pq.read_table(str(parquet_path))
            df: pd.DataFrame = table.to_pandas()

            # Loc theo thoi gian neu co column time
            if "time" in df.columns and start and end:
                start_ts = int(
                    datetime.strptime(start, "%Y-%m-%d")
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                )
                end_ts = int(
                    datetime.strptime(end, "%Y-%m-%d")
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                )
                # Xu ly ca microseconds
                df = df[
                    (df["time"] >= start_ts * 1_000_000)
                    & (df["time"] <= end_ts * 1_000_000 + 86_400_000_000)
                ]

            # Sort theo thoi gian
            if "time" in df.columns:
                df = df.sort_values("time")

            logger.info(
                "Da doc {n} ticks tu {path}",
                n=len(df),
                path=parquet_path,
            )
            return df.to_dict("records")

        except Exception as exc:
            logger.error(
                "Loi doc Parquet {path}: {exc}",
                path=parquet_path,
                exc=exc,
            )
            return []

    async def _load_calendar_events(
        self,
        symbol: str,
        start: str,
        end: str,
    ) -> None:
        """Load calendar events tu MacroEngine cho period.

        Args:
            symbol: Ma symbol
            start: Ngay bat dau
            end: Ngay ket thuc
        """
        try:
            # Import lazy de tranh circular import
            from core.macro.macro_engine import MacroEngine, MacroEngineConfig

            engine = MacroEngine(
                config=MacroEngineConfig(),
            )

            # Lay events tu CalendarScraper
            if hasattr(engine, "_scraper") and hasattr(engine._scraper, "get_events"):
                events = engine._scraper.get_events(
                    start=start,
                    end=end,
                    currencies=["USD", "XAU"],
                    min_impact="MEDIUM",
                )
                self._calendar_events = list(events) if events else []
            else:
                self._calendar_events = []

        except Exception as exc:
            logger.warning(
                "Khong the load calendar events: {exc}",
                exc=exc,
            )
            self._calendar_events = []

    # -------------------------------------------------------------------------
    # Internal: Event injection
    # -------------------------------------------------------------------------
    def _build_tick_event(
        self,
        symbol: str,
        tick: dict[str, Any],
        timestamp_us: int,
    ) -> TickReceivedEvent:
        """Xay dung TickReceivedEvent tu tick data.

        Args:
            symbol: Ma symbol
            tick: Dict chua tick data
            timestamp_us: Timestamp microseconds

        Returns:
            TickReceivedEvent
        """
        return TickReceivedEvent(
            symbol=symbol,
            timestamp_us=timestamp_us,
            bid=float(tick.get("bid", 0.0)),
            ask=float(tick.get("ask", 0.0)),
            last=float(tick.get("last", tick.get("close", 0.0))),
            volume=float(tick.get("volume", 0.0)),
            flags=int(tick.get("flags", 0)),
            mid_price=(float(tick.get("bid", 0.0)) + float(tick.get("ask", 0.0))) / 2,
        )

    async def _inject_calendar_event(self, cal_event: dict[str, Any]) -> None:
        """Inject mot calendar event vao EventBus.

        Args:
            cal_event: Dict chua thong tin calendar event
        """
        if not self._bus:
            self._events_injected += 1
            return

        try:
            impact_raw = cal_event.get("impact", "MEDIUM")
            impact_map = {
                "LOW": NewsImpact.LOW,
                "MEDIUM": NewsImpact.MEDIUM,
                "HIGH": NewsImpact.HIGH,
            }
            impact = impact_map.get(impact_raw, NewsImpact.MEDIUM)

            scheduled = cal_event.get("scheduled_time")
            if isinstance(scheduled, str):
                try:
                    scheduled = datetime.fromisoformat(scheduled)
                except (ValueError, TypeError):
                    scheduled = datetime.utcnow()

            news_event = NewsAlertEvent(
                event_id=str(cal_event.get("event_id", "")),
                title=str(cal_event.get("title", "Unknown")),
                currency=str(cal_event.get("currency", "USD")),
                impact=impact,
                scheduled_time=scheduled if isinstance(scheduled, datetime) else datetime.utcnow(),
                forecast=cal_event.get("forecast"),
                previous=cal_event.get("previous"),
                state="SCHEDULED",
            )

            self._bus.publish(news_event)
            self._events_injected += 1

            logger.debug(
                "[CalendarInject] {title} ({impact}) @ {time}",
                title=news_event.title,
                impact=impact.value,
                time=scheduled,
            )

        except Exception as exc:
            logger.warning(
                "Loi inject calendar event: {exc}",
                exc=exc,
            )

    # -------------------------------------------------------------------------
    # Internal: Stats
    # -------------------------------------------------------------------------
    def _stats(self) -> dict[str, Any]:
        """Tra ve statistics cho lan chay nay.

        Returns:
            Dict chua statistics
        """
        elapsed = self._end_wall - self._start_wall
        return {
            "ticks_replayed": self._ticks_replayed,
            "events_injected": self._events_injected,
            "elapsed_seconds": round(elapsed, 3),
            "tick_rate": round(self._ticks_replayed / elapsed, 1) if elapsed > 0 else 0,
            "leakage_guard": self._leakage_guard_active,
        }

    def get_progress(self) -> dict[str, Any]:
        """Lay trang thai hien tai cua simulator.

        Returns:
            Dict progress
        """
        elapsed = asyncio.get_event_loop().time() - self._start_wall
        return {
            "ticks_replayed": self._ticks_replayed,
            "events_injected": self._events_injected,
            "running": self._running,
            "elapsed_seconds": round(elapsed, 3),
            "tick_rate": round(self._ticks_replayed / elapsed, 1) if elapsed > 0 else 0,
        }
