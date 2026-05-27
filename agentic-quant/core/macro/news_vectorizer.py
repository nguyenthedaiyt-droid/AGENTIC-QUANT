# =============================================================================
# AGENTIC-QUANT — News Impact Vectorizer
# Tinh I_news, Surprise Factor, Welford's algorithm
# =============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from .calendar_scraper import RawNewsEvent, ImpactLevel, EconomicCalendarDB

if TYPE_CHECKING:
    pass


# =============================================================================
# Welford's Online Variance
# =============================================================================
class WelfordVariance:
    """
    Welford's online algorithm cho running mean and variance.

    Cho phep cap nhat variance khong can luu tat ca data points.
    Dung de tinh running surprise_sigma sau moi event.

    Algorithm:
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        M2 += delta * delta2
        variance = M2 / (n - 1)
        std = sqrt(variance)
    """

    def __init__(self) -> None:
        self._n: int = 0
        self._mean: float = 0.0
        self._M2: float = 0.0

    def update(self, x: float) -> float:
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._M2 += delta * delta2
        if self._n < 2:
            return 0.0
        return math.sqrt(max(0.0, self._M2 / (self._n - 1)))

    def update_many(self, values: list[float]) -> float:
        for v in values:
            self.update(v)
        return self.std

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        if self._n < 2:
            return 0.0
        return math.sqrt(max(0.0, self._M2 / (self._n - 1)))

    @property
    def variance(self) -> float:
        if self._n < 2:
            return 0.0
        return self._M2 / (self._n - 1)

    @property
    def sample_count(self) -> int:
        return max(0, self._n)

    def reset(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._M2 = 0.0

    def __repr__(self) -> str:
        return f"WelfordVariance(n={self.sample_count}, mean={self._mean:.4f}, std={self.std:.4f})"


# =============================================================================
# News Impact Score
# =============================================================================
@dataclass
class NewsImpactScore:
    event_id: str
    impact_base: float
    market_volatility_factor: float = 0.0
    i_news: float = 0.0
    surprise_z: float | None = None
    surprise_factor: float = 1.0
    surprise_direction: str = "NEUTRAL"

    @property
    def is_surprise(self) -> bool:
        return self.surprise_z is not None and abs(self.surprise_z) > 2.0

    @property
    def is_bearish(self) -> bool:
        return self.surprise_direction == "BEARISH"

    @property
    def is_bullish(self) -> bool:
        return self.surprise_direction == "BULLISH"


# =============================================================================
# News Vectorizer
# =============================================================================
class NewsVectorizer:
    BASE_IMPACT: dict[ImpactLevel, float] = {
        ImpactLevel.LOW: 0.2,
        ImpactLevel.MEDIUM: 0.5,
        ImpactLevel.HIGH: 1.0,
    }

    def __init__(
        self,
        alpha: float = 0.4,
        surprise_threshold: float = 2.0,
        db: EconomicCalendarDB | None = None,
    ) -> None:
        self._alpha = alpha
        self._surprise_threshold = surprise_threshold
        self._db = db
        self._welford: dict[str, WelfordVariance] = {}
        self._atr_d1: float = 10.0

    def set_atr_d1(self, atr: float) -> None:
        self._atr_d1 = max(0.01, atr)

    def vectorize(self, event: RawNewsEvent) -> NewsImpactScore:
        base = self.BASE_IMPACT.get(event.impact, 0.5)
        m_bar = self._get_market_volatility(event.currency)
        m_bar_ec = m_bar / max(self._atr_d1, 0.01)
        i_news = base * (1.0 + self._alpha * m_bar_ec)
        i_news = min(i_news, 3.0)

        surprise_z: float | None = None
        surprise_factor = 1.0
        surprise_direction = "NEUTRAL"

        if event.actual is not None and event.forecast is not None:
            surprise_z = self._compute_surprise_z(event)
            if surprise_z is not None:
                surprise_factor = self._compute_surprise_factor(abs(surprise_z))
                surprise_direction = self._compute_direction(
                    surprise_z, event.actual, event.forecast
                )
                self._update_welford(event.currency, event.event_id, surprise_z)

        return NewsImpactScore(
            event_id=event.event_id,
            impact_base=base,
            market_volatility_factor=m_bar_ec,
            i_news=i_news,
            surprise_z=surprise_z,
            surprise_factor=surprise_factor,
            surprise_direction=surprise_direction,
        )

    def _get_market_volatility(self, currency: str) -> float:
        return self._atr_d1 * 0.1

    def _compute_surprise_z(self, event: RawNewsEvent) -> float | None:
        if event.actual is None or event.forecast is None:
            return None
        if event.forecast == 0:
            return None
        sigma = self._get_surprise_sigma(event.event_id)
        diff = event.actual - event.forecast
        if sigma is None or sigma < 0.001:
            # Relative surprise (khong co historical sigma)
            return diff / max(abs(event.forecast), 0.001)
        return diff / sigma

    def _get_surprise_sigma(self, event_id: str) -> float | None:
        if self._db:
            return self._db.get_surprise_sigma(event_id)
        return None

    def _compute_surprise_factor(self, z: float) -> float:
        abs_z = abs(z)
        if abs_z <= 1.0:
            return 1.0
        elif abs_z <= 2.0:
            return 1.0 + 0.25 * (abs_z - 1.0)
        else:
            return 1.25 + 0.5 * (abs_z - 2.0)

    def _compute_direction(self, z: float | None, actual: float, forecast: float) -> str:
        # Fallback: so sanh actual vs forecast truc tiep
        if actual > forecast:
            return "BULLISH"
        elif actual < forecast:
            return "BEARISH"
        return "NEUTRAL"

    def _update_welford(self, currency: str, event_id: str, surprise_z: float) -> None:
        if currency not in self._welford:
            self._welford[currency] = WelfordVariance()
        self._welford[currency].update(surprise_z)
        if self._db:
            self._db.update_surprise_sigma(
                event_id=event_id,
                currency=currency,
                event_type="GENERAL",
                new_surprise=surprise_z,
            )

    def get_welford(self, currency: str) -> WelfordVariance | None:
        return self._welford.get(currency)

    def adjust_i_news(self, i_news: float, active_guardrail: bool, dampening_factor: float = 0.3) -> float:
        if active_guardrail:
            return i_news * dampening_factor
        return i_news
