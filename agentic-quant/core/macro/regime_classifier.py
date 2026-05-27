# =============================================================================
# AGENTIC-QUANT — Post-News Regime Classifier
# Phan loai thi truong sau khi tin ra: IMPULSIVE/REVERSAL/CHOPPY
# =============================================================================

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from .calendar_scraper import EconomicCalendarDB

if TYPE_CHECKING:
    pass


# =============================================================================
# Post-News Regime
# =============================================================================
class PostNewsRegime(str, Enum):
    """Loai regime sau tin."""

    IMPULSIVE_FOLLOW_THROUGH = "IMPULSIVE_FOLLOW_THROUGH"
    REVERSAL_AFTER_SPIKE = "REVERSAL_AFTER_SPIKE"
    CHOPPY_CONSOLIDATION = "CHOPPY_CONSOLIDATION"
    UNKNOWN = "UNKNOWN"


@dataclass
class NewsOutcome:
    """
    Ket qua sau khi tin ra.

    Attributes:
        event_id: ID su kien
        classification: Regime sau tin
        price_at_event: Gia luc event xay ra
        price_5min_after: Gia 5 phut sau event
        directional_move: (P_5min - P0) / ATR_H1
        surprise_direction: Huong surprise
        surprise_z: Z-score
        regime: PostNewsRegime
    """

    outcome_id: int | None = None
    event_id: str = ""
    classification: PostNewsRegime = PostNewsRegime.UNKNOWN
    price_at_event: float = 0.0
    price_5min_after: float = 0.0
    directional_move: float = 0.0
    surprise_direction: str = "NEUTRAL"
    surprise_z: float | None = None
    regime: PostNewsRegime = PostNewsRegime.UNKNOWN
    created_at: datetime = None

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

    @property
    def is_impulsive(self) -> bool:
        return self.classification == PostNewsRegime.IMPULSIVE_FOLLOW_THROUGH

    @property
    def is_reversal(self) -> bool:
        return self.classification == PostNewsRegime.REVERSAL_AFTER_SPIKE

    @property
    def is_choppy(self) -> bool:
        return self.classification == PostNewsRegime.CHOPPY_CONSOLIDATION


# =============================================================================
# Classifier
# =============================================================================
class PostNewsRegimeClassifier:
    """
    Phan loai thi truong sau khi tin ra.

    Cong thuc:
        directional_move = (P_5min - P0) / ATR_H1

    Loai bo:
        - IMPULSIVE: directional_move / ATR_H1 >= 0.8 AND cung huong voi surprise
        - REVERSAL: directional_move / ATR_H1 <= -0.5
        - CHOPPY: con lai

    Args:
        impulsive_threshold: Nguong IMPULSIVE (default: 0.8)
        reversal_threshold: Nguong REVERSAL (default: -0.5)
        classification_window_minutes: So phut sau event de phan loai (default: 5)
        db_path: Duong dan SQLite DB
    """

    def __init__(
        self,
        impulsive_threshold: float = 0.8,
        reversal_threshold: float = -0.5,
        classification_window_minutes: int = 5,
        db: EconomicCalendarDB | None = None,
        db_path: str | Path | None = None,
    ) -> None:
        self._impulsive_threshold = impulsive_threshold
        self._reversal_threshold = reversal_threshold
        self._window_minutes = classification_window_minutes
        # Use shared DB instance if provided, otherwise create own connection
        self._db: EconomicCalendarDB | None = db
        self._db_path = Path(db_path) if db_path else None
        self._own_conn: sqlite3.Connection | None = None
        if self._db is None and self._db_path:
            self._own_conn = sqlite3.connect(str(self._db_path))

    def classify(
        self,
        event_id: str,
        price_at_event: float,
        price_5min_after: float,
        atr_h1: float,
        surprise_direction: str,
        surprise_z: float | None = None,
    ) -> NewsOutcome:
        """
        Phan loai thi truong sau tin.

        Args:
            event_id: ID su kien
            price_at_event: Gia luc event
            price_5min_after: Gia 5 phut sau event
            atr_h1: ATR tren H1
            surprise_direction: "BULLISH" | "BEARISH" | "NEUTRAL"
            surprise_z: Z-score tu news vectorizer

        Returns:
            NewsOutcome
        """
        # Tinh directional move
        directional_move = (price_5min_after - price_at_event) / max(atr_h1, 0.01)

        # Phan loai
        classification = self._classify_regime(
            directional_move, surprise_direction
        )

        outcome = NewsOutcome(
            event_id=event_id,
            classification=classification,
            price_at_event=price_at_event,
            price_5min_after=price_5min_after,
            directional_move=directional_move,
            surprise_direction=surprise_direction,
            surprise_z=surprise_z,
            regime=classification,
        )

        logger.info(
            "News outcome: {event_id} -> {regime} "
            "(directional_move={dm:.3f}, surprise={dir}, z={z})",
            event_id=event_id,
            regime=classification.value,
            dm=directional_move,
            dir=surprise_direction,
            z=surprise_z,
        )

        return outcome

    def _classify_regime(
        self,
        directional_move: float,
        surprise_direction: str,
    ) -> PostNewsRegime:
        """Phan loai regime."""
        # IMPULSIVE: Vuot nguong cung huong surprise
        if surprise_direction == "BULLISH" and directional_move >= self._impulsive_threshold:
            return PostNewsRegime.IMPULSIVE_FOLLOW_THROUGH
        if surprise_direction == "BEARISH" and directional_move <= -self._impulsive_threshold:
            return PostNewsRegime.IMPULSIVE_FOLLOW_THROUGH

        # REVERSAL: Dao chieu manh (vuot nguong reversal)
        if directional_move <= self._reversal_threshold:
            return PostNewsRegime.REVERSAL_AFTER_SPIKE

        # REVERSAL: Di chuyen nguoc huong surprise
        if surprise_direction == "BULLISH" and directional_move <= -0.2:
            return PostNewsRegime.REVERSAL_AFTER_SPIKE
        if surprise_direction == "BEARISH" and directional_move >= 0.2:
            return PostNewsRegime.REVERSAL_AFTER_SPIKE

        return PostNewsRegime.CHOPPY_CONSOLIDATION

    def save_outcome(self, outcome: NewsOutcome) -> int | None:
        """Luu outcome vao SQLite."""
        if self._db:
            return self._save_outcome_shared(outcome)

        if self._own_conn is None and self._db_path:
            self._own_conn = sqlite3.connect(str(self._db_path))

        if self._own_conn is None:
            logger.warning("Khong co DB connection, skip save")
            return None

        try:
            cur = self._own_conn.cursor()
            cur.execute("""
                INSERT INTO news_outcomes
                (event_id, classification, price_at_event, price_5min_after,
                 directional_move, surprise_direction, surprise_z, regime, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                outcome.event_id,
                outcome.classification.value,
                outcome.price_at_event,
                outcome.price_5min_after,
                outcome.directional_move,
                outcome.surprise_direction,
                outcome.surprise_z,
                outcome.regime.value,
                outcome.created_at.isoformat(),
            ))
            outcome_id = cur.lastrowid
            self._own_conn.commit()
            return outcome_id
        except Exception:
            logger.exception("Loi save outcome")
            return None

    def _save_outcome_shared(self, outcome: NewsOutcome) -> int | None:
        """Luu outcome bang shared DB connection."""
        if not self._db._conn:
            return None
        try:
            cur = self._db._conn.cursor()
            cur.execute("""
                INSERT INTO news_outcomes
                (event_id, classification, price_at_event, price_5min_after,
                 directional_move, surprise_direction, surprise_z, regime, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                outcome.event_id,
                outcome.classification.value,
                outcome.price_at_event,
                outcome.price_5min_after,
                outcome.directional_move,
                outcome.surprise_direction,
                outcome.surprise_z,
                outcome.regime.value,
                outcome.created_at.isoformat(),
            ))
            self._db._conn.commit()
            cur.execute("SELECT last_insert_rowid()")
            return cur.fetchone()[0]
        except Exception:
            logger.exception("Loi save outcome (shared)")
            return None

    def get_outcomes_for_event(self, event_id: str) -> list[NewsOutcome]:
        """Lay tat ca outcomes cho mot event."""
        conn = self._get_conn()
        if conn is None:
            return []

        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT * FROM news_outcomes WHERE event_id = ?
                ORDER BY created_at DESC
            """, (event_id,))

            rows = cur.fetchall()
            if conn != self._db._conn:
                conn.close()

            outcomes = []
            for row in rows:
                outcomes.append(NewsOutcome(
                    outcome_id=row[0],
                    event_id=row[1],
                    classification=PostNewsRegime(row[2]),
                    price_at_event=row[3],
                    price_5min_after=row[4],
                    directional_move=row[5],
                    surprise_direction=row[6],
                    surprise_z=row[7],
                    regime=PostNewsRegime(row[8]),
                    created_at=datetime.fromisoformat(row[9]),
                ))
            return outcomes

        except Exception:
            logger.exception("Loi get outcomes")
            return []

    def get_statistics(self) -> dict:
        """Lay thong ke outcomes."""
        conn = self._get_conn()
        if conn is None:
            return {}

        try:
            cur = conn.cursor()

            cur.execute("""
                SELECT classification, COUNT(*) as count
                FROM news_outcomes
                GROUP BY classification
            """)
            by_type = {row[0]: row[1] for row in cur.fetchall()}

            cur.execute("""
                SELECT AVG(ABS(directional_move)), MAX(ABS(directional_move))
                FROM news_outcomes
            """)
            avg_move, max_move = cur.fetchone() or (0, 0)

            if conn != self._db._conn:
                conn.close()

            return {
                "total_outcomes": sum(by_type.values()),
                "by_type": by_type,
                "avg_directional_move": avg_move or 0,
                "max_directional_move": max_move or 0,
            }
        except Exception:
            logger.exception("Loi get statistics")
            return {}

    def _get_conn(self) -> sqlite3.Connection | None:
        """Lay connection phu hop."""
        if self._db and self._db._conn:
            return self._db._conn
        if self._own_conn:
            return self._own_conn
        if self._db_path and self._db_path.exists():
            try:
                return sqlite3.connect(str(self._db_path))
            except Exception:
                return None
        return None
