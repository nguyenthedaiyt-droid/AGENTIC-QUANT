# =============================================================================
# AGENTIC-QUANT — Economic Calendar Scraper
# Fetch lịch kinh tế từ ForexFactory, lưu vào SQLite
# =============================================================================

from __future__ import annotations

import asyncio
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import aiohttp
from loguru import logger

if TYPE_CHECKING:
    pass


# =============================================================================
# Data Models
# =============================================================================
class ImpactLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    XAU = "XAU"  # Gold


@dataclass
class RawNewsEvent:
    """
    Su kien kinh te sau khi scrape.

    Thuong duoc tao tu ForexFactory JSON response.
    """

    event_id: str
    title: str
    country: str
    currency: str
    impact: ImpactLevel
    scheduled_time: datetime
    forecast: float | None = None
    previous: float | None = None
    actual: float | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_surprise: bool = False
    surprise_z: float | None = None
    scraped_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_high_impact(self) -> bool:
        return self.impact == ImpactLevel.HIGH

    @property
    def is_medium_impact(self) -> bool:
        return self.impact == ImpactLevel.MEDIUM

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "title": self.title,
            "country": self.country,
            "currency": self.currency,
            "impact": self.impact.value,
            "scheduled_time": self.scheduled_time.isoformat(),
            "forecast": self.forecast,
            "previous": self.previous,
            "actual": self.actual,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_surprise": self.is_surprise,
            "surprise_z": self.surprise_z,
            "scraped_at": self.scraped_at.isoformat(),
        }


# =============================================================================
# Economic Calendar DB
# =============================================================================
class EconomicCalendarDB:
    """
    SQLite storage cho economic calendar.

    Schema:
    - economic_calendar: Tat ca su kien kinh te
    - news_historical_volatility: Lịch sử volatility sau mỗi event
    - news_outcomes: Ket qua sau khi event xay ra
    """

    def __init__(self, db_path: str | Path = "data/economic_calendar.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        """Ket noi SQLite."""
        self._conn = sqlite3.connect(
            str(self._db_path),
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def _create_tables(self) -> None:
        """Tao database schema."""
        if not self._conn:
            raise RuntimeError("DB chua connect")

        cur = self._conn.cursor()

        # economic_calendar table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS economic_calendar (
                event_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                country TEXT NOT NULL,
                currency TEXT NOT NULL,
                impact TEXT NOT NULL,
                scheduled_time TEXT NOT NULL,
                forecast REAL,
                previous REAL,
                actual REAL,
                is_surprise INTEGER DEFAULT 0,
                surprise_z REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                scraped_at TEXT NOT NULL
            )
        """)

        # news_historical_volatility table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS news_historical_volatility (
                event_id TEXT PRIMARY KEY,
                currency TEXT NOT NULL,
                event_type TEXT NOT NULL,
                surprise_sigma REAL NOT NULL,
                sample_count INTEGER NOT NULL DEFAULT 1,
                avg_actual REAL,
                avg_forecast REAL,
                avg_surprise REAL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (event_id) REFERENCES economic_calendar(event_id)
            )
        """)

        # news_outcomes table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS news_outcomes (
                outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                classification TEXT NOT NULL,
                price_at_event REAL,
                price_5min_after REAL,
                directional_move REAL,
                surprise_direction TEXT,
                surprise_z REAL,
                regime TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (event_id) REFERENCES economic_calendar(event_id)
            )
        """)

        # Indexes
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_calendar_scheduled
            ON economic_calendar(scheduled_time)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_calendar_currency
            ON economic_calendar(currency)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_calendar_impact
            ON economic_calendar(impact)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_volatility_currency
            ON news_historical_volatility(currency)
        """)

        self._conn.commit()

    def upsert_event(self, event: RawNewsEvent) -> None:
        """Insert hoac update event."""
        if not self._conn:
            raise RuntimeError("DB chua connect")

        cur = self._conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO economic_calendar
            (event_id, title, country, currency, impact, scheduled_time,
             forecast, previous, actual, is_surprise, surprise_z,
             created_at, updated_at, scraped_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.event_id,
            event.title,
            event.country,
            event.currency,
            event.impact.value,
            event.scheduled_time.isoformat(),
            event.forecast,
            event.previous,
            event.actual,
            int(event.is_surprise),
            event.surprise_z,
            event.created_at.isoformat(),
            event.updated_at.isoformat(),
            event.scraped_at.isoformat(),
        ))
        self._conn.commit()

    def upsert_events(self, events: list[RawNewsEvent]) -> None:
        """Batch upsert events."""
        if not self._conn:
            raise RuntimeError("DB chua connect")

        cur = self._conn.cursor()
        rows = [
            (
                e.event_id, e.title, e.country, e.currency, e.impact.value,
                e.scheduled_time.isoformat(), e.forecast, e.previous, e.actual,
                int(e.is_surprise), e.surprise_z,
                e.created_at.isoformat(), e.updated_at.isoformat(),
                e.scraped_at.isoformat(),
            )
            for e in events
        ]
        cur.executemany("""
            INSERT OR REPLACE INTO economic_calendar
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        self._conn.commit()

    def get_upcoming_events(
        self,
        from_time: datetime,
        to_time: datetime,
        currencies: list[str] | None = None,
        min_impact: ImpactLevel = ImpactLevel.MEDIUM,
    ) -> list[RawNewsEvent]:
        """Lay su kien sap toi trong khoang thoi gian."""
        if not self._conn:
            raise RuntimeError("DB chua connect")

        cur = self._conn.cursor()

        # Impact filter: LOW < MEDIUM < HIGH
        impact_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        min_impact_val = impact_order[min_impact.value]

        # Chi lay cac impact >= min_impact
        included_impacts = [k for k, v in impact_order.items() if v >= min_impact_val]
        impact_placeholders = ",".join("?" * len(included_impacts))
        query = f"""
            SELECT * FROM economic_calendar
            WHERE scheduled_time >= ? AND scheduled_time <= ?
            AND impact IN ({impact_placeholders})
        """
        params: list = [
            from_time.isoformat(),
            to_time.isoformat(),
        ]
        params.extend(included_impacts)

        if currencies:
            placeholders = ",".join("?" * len(currencies))
            query += f" AND currency IN ({placeholders})"
            params.extend(currencies)

        query += " ORDER BY scheduled_time ASC"

        cur.execute(query, params)
        rows = cur.fetchall()

        return [self._row_to_event(row) for row in rows]

    def get_next_high_impact(
        self,
        after: datetime | None = None,
    ) -> RawNewsEvent | None:
        """Lay su kien high impact tiep theo."""
        if not self._conn:
            raise RuntimeError("DB chua connect")

        now = after or datetime.now(timezone.utc)
        cur = self._conn.cursor()
        cur.execute("""
            SELECT * FROM economic_calendar
            WHERE scheduled_time >= ? AND impact = 'HIGH'
            ORDER BY scheduled_time ASC
            LIMIT 1
        """, (now.isoformat(),))

        row = cur.fetchone()
        if row:
            return self._row_to_event(row)
        return None

    def update_event_actual(
        self,
        event_id: str,
        actual: float,
        is_surprise: bool,
        surprise_z: float | None,
    ) -> None:
        """Cap nhat actual value sau khi event xay ra."""
        if not self._conn:
            raise RuntimeError("DB chua connect")

        cur = self._conn.cursor()
        cur.execute("""
            UPDATE economic_calendar
            SET actual = ?, is_surprise = ?, surprise_z = ?, updated_at = ?
            WHERE event_id = ?
        """, (
            actual,
            int(is_surprise),
            surprise_z,
            datetime.now(timezone.utc).isoformat(),
            event_id,
        ))
        self._conn.commit()

    def get_surprise_sigma(self, event_id: str) -> float | None:
        """Lay surprise sigma tu lịch sử."""
        if not self._conn:
            raise RuntimeError("DB chua connect")

        cur = self._conn.cursor()
        cur.execute("""
            SELECT surprise_sigma FROM news_historical_volatility
            WHERE event_id = ?
        """, (event_id,))

        row = cur.fetchone()
        return float(row["surprise_sigma"]) if row else None

    def update_surprise_sigma(
        self,
        event_id: str,
        currency: str,
        event_type: str,
        new_surprise: float,
    ) -> None:
        """Cap nhat surprise sigma bang Welford's algorithm."""
        if not self._conn:
            raise RuntimeError("DB chua connect")

        cur = self._conn.cursor()
        cur.execute("""
            SELECT * FROM news_historical_volatility WHERE event_id = ?
        """, (event_id,))

        row = cur.fetchone()
        now = datetime.now(timezone.utc).isoformat()

        if row:
            # Welford's online update
            n = row["sample_count"]
            old_sigma = row["surprise_sigma"]
            old_mean = row["avg_surprise"] or 0.0

            # Running mean
            new_mean = old_mean + (new_surprise - old_mean) / (n + 1)
            # Running variance (simplified: sigma^2)
            new_sigma = old_sigma + (
                (new_surprise - old_mean) * (new_surprise - new_mean) - old_sigma
            ) / (n + 1)
            new_sigma = max(0.01, new_sigma)  # Prevent negative

            cur.execute("""
                UPDATE news_historical_volatility
                SET surprise_sigma = ?, sample_count = ?,
                    avg_surprise = ?, updated_at = ?
                WHERE event_id = ?
            """, (new_sigma, n + 1, new_mean, now, event_id))
        else:
            cur.execute("""
                INSERT INTO news_historical_volatility
                (event_id, currency, event_type, surprise_sigma,
                 sample_count, avg_surprise, updated_at)
                VALUES (?, ?, ?, ?, 1, ?, ?)
            """, (event_id, currency, event_type, abs(new_surprise), new_surprise, now))

        self._conn.commit()

    def _row_to_event(self, row: sqlite3.Row) -> RawNewsEvent:
        impact_map = {"LOW": ImpactLevel.LOW, "MEDIUM": ImpactLevel.MEDIUM, "HIGH": ImpactLevel.HIGH}
        return RawNewsEvent(
            event_id=row["event_id"],
            title=row["title"],
            country=row["country"],
            currency=row["currency"],
            impact=impact_map.get(row["impact"], ImpactLevel.MEDIUM),
            scheduled_time=datetime.fromisoformat(row["scheduled_time"]),
            forecast=row["forecast"],
            previous=row["previous"],
            actual=row["actual"],
            is_surprise=bool(row["is_surprise"]),
            surprise_z=row["surprise_z"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            scraped_at=datetime.fromisoformat(row["scraped_at"]),
        )

    def __enter__(self) -> "EconomicCalendarDB":
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.close()


# =============================================================================
# Calendar Scraper
# =============================================================================
class CalendarScraper:
    """
    Scrapes economic calendar từ ForexFactory.

    Flow:
    1. Fetch ForexFactory JSON
    2. Retry neu fail (3 lan, exponential backoff)
    3. Fallback sang Investing.com neu FF fail
    4. Chuẩn hóa thanh RawNewsEvent
    5. Lưu vào SQLite

    Args:
        currencies: List currencies cần theo dõi (default: ["USD", "XAU"])
        min_impact: Impact tối thiểu để scrape
        db: EconomicCalendarDB instance
    """

    FF_API = "https://nuffptbcxi.execute-api.us-east-1.amazonaws.com/default/ffCalendarEvents"

    def __init__(
        self,
        currencies: list[str] | None = None,
        min_impact: ImpactLevel = ImpactLevel.MEDIUM,
        db: EconomicCalendarDB | None = None,
    ) -> None:
        self._currencies = currencies or ["USD", "XAU"]
        self._min_impact = min_impact
        self._db = db or EconomicCalendarDB()
        self._session: aiohttp.ClientSession | None = None
        self._retry_delays = [2, 4, 8]  # seconds
        self._scraped_count = 0

    async def start(self) -> None:
        """Khoi tao HTTP session."""
        timeout = aiohttp.ClientTimeout(total=30)
        self._session = aiohttp.ClientSession(timeout=timeout)

    async def stop(self) -> None:
        """Dong HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def scrape(self) -> list[RawNewsEvent]:
        """
        Scrapes tat ca events trong 24h toi.

        Returns:
            List RawNewsEvent
        """
        events: list[RawNewsEvent] = []

        # Thu ForexFactory
        ff_events = await self._fetch_forexfactory()
        if ff_events:
            events.extend(ff_events)
        else:
            # Fallback Investing.com
            logger.warning("ForexFactory failed, trying Investing.com...")
            inv_events = await self._fetch_investing()
            events.extend(inv_events)

        # Luu vao DB
        if events:
            self._db.upsert_events(events)

        self._scraped_count += len(events)
        logger.info(
            "Scraped {n} events, total: {total}",
            n=len(events),
            total=self._scraped_count,
        )

        return events

    async def _fetch_forexfactory(self) -> list[RawNewsEvent]:
        """Fetch tu ForexFactory API."""
        if not self._session:
            await self.start()

        for attempt, delay in enumerate(self._retry_delays):
            try:
                async with self._session.get(self.FF_API) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        events = self._parse_forexfactory(data)
                        logger.info(
                            "ForexFactory: fetched {n} events",
                            n=len(events),
                        )
                        return events
                    else:
                        logger.warning(
                            "ForexFactory HTTP {status}, attempt {a}/{total}",
                            status=resp.status,
                            a=attempt + 1,
                            total=len(self._retry_delays),
                        )
            except asyncio.TimeoutError:
                logger.warning(
                    "ForexFactory timeout, attempt {a}/{total}",
                    a=attempt + 1,
                    total=len(self._retry_delays),
                )
            except Exception:
                logger.exception(
                    "ForexFactory error attempt {a}/{total}",
                    a=attempt + 1,
                    total=len(self._retry_delays),
                )

            if attempt < len(self._retry_delays) - 1:
                await asyncio.sleep(delay)

        return []

    async def _fetch_investing(self) -> list[RawNewsEvent]:
        """Fallback: scrape Investing.com HTML."""
        # Investing.com scraping de phong
        # HTML parsing with BeautifulSoup de phong
        # Chi tra ve empty list neu chua implement
        logger.warning("Investing.com fallback chua implement")
        return []

    def _parse_forexfactory(self, data: dict) -> list[RawNewsEvent]:
        """Parse ForexFactory API response."""
        events: list[RawNewsEvent] = []
        impact_filter = {"low": ImpactLevel.LOW, "medium": ImpactLevel.MEDIUM, "high": ImpactLevel.HIGH}

        for item in data.get("channel", {}).get("item", []):
            try:
                # Parse fields
                title = item.get("title", "")
                currency = self._extract_currency(title)
                if currency not in self._currencies:
                    continue

                impact_str = (item.get("impact") or "").lower()
                impact = impact_filter.get(impact_str, ImpactLevel.MEDIUM)
                if impact == ImpactLevel.LOW and self._min_impact == ImpactLevel.MEDIUM:
                    continue

                # Parse datetime
                pub_date = item.get("pubDate", "")
                scheduled = self._parse_ff_date(pub_date)

                # Parse forecast/previous/actual
                forecast, previous, actual = self._extract_values(item.get("description", ""))

                event_id = self._generate_event_id(title, scheduled)

                event = RawNewsEvent(
                    event_id=event_id,
                    title=title,
                    country=item.get("country", ""),
                    currency=currency,
                    impact=impact,
                    scheduled_time=scheduled,
                    forecast=forecast,
                    previous=previous,
                    actual=actual,
                )

                events.append(event)

            except Exception:
                logger.warning("Loi parse event: {item}", item=item.get("title", ""))

        return events

    def _extract_currency(self, title: str) -> str:
        """Extract currency tu event title."""
        # Common patterns: "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD"
        known = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD", "CHF", "XAU"]
        for cur in known:
            if cur in title.upper():
                return cur
        return "USD"

    def _parse_ff_date(self, pub_date: str) -> datetime:
        """Parse ForexFactory date format."""
        try:
            # Format: "Mon, 27 May 2024 08:30:00 GMT"
            return datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            # Try alternate format
            try:
                return datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
            except Exception:
                return datetime.now(timezone.utc)

    def _extract_values(self, description: str) -> tuple[float | None, float | None, float | None]:
        """Extract forecast/previous/actual tu description HTML."""
        import re

        forecast = previous = actual = None

        def extract_number(text: str) -> float | None:
            m = re.search(r"[-+]?[\d,]+\.?\d*", text)
            if m:
                return float(m.group().replace(",", ""))
            return None

        parts = description.split("<br>")
        for part in parts:
            part_upper = part.upper()
            if "FORECAST" in part_upper or "FORECAST" in part:
                forecast = extract_number(part)
            elif "PREVIOUS" in part_upper or "PREVIOUS" in part:
                previous = extract_number(part)
            elif "ACTUAL" in part_upper or "ACTUAL" in part:
                actual = extract_number(part)

        return forecast, previous, actual

    def _generate_event_id(self, title: str, scheduled: datetime) -> str:
        """Generate unique event ID."""
        import hashlib

        key = f"{title}:{scheduled.isoformat()}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    @property
    def scraped_count(self) -> int:
        return self._scraped_count
