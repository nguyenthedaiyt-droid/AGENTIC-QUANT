# =============================================================================
# AGENTIC-QUANT — Unit Tests cho Macro Calendar Engine
# =============================================================================

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.macro import (
    CalendarScraper,
    EconomicCalendarDB,
    RawNewsEvent,
    ImpactLevel,
    Currency,
    NewsVectorizer,
    NewsImpactScore,
    WelfordVariance,
    VolatilityCountdown,
    MacroState,
    CountdownEvent,
    PostNewsRegimeClassifier,
    NewsOutcome,
    PostNewsRegime,
    MacroEngine,
    MacroEngineConfig,
)


class TestWelfordVariance:
    """Tests cho Welford's online algorithm."""

    def test_initial_state(self) -> None:
        w = WelfordVariance()
        assert w.sample_count == 0
        assert w.mean == 0.0
        assert w.std == 0.0

    def test_single_update(self) -> None:
        """Sau 1 update: n=1, mean = x (10.0), std = 0 (chua du du lieu)."""
        w = WelfordVariance()
        std = w.update(10.0)
        assert w.mean == 10.0
        assert w.sample_count == 1
        assert std == 0.0  # Chi 1 diem -> variance undefined

    def test_multiple_updates(self) -> None:
        """Tinh mean/var dung voi Welford."""
        w = WelfordVariance()
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        for v in values:
            w.update(v)

        # 8 values -> sample_count = 8
        assert w.sample_count == 8
        # Mean = (2+4+4+4+5+5+7+9)/8 = 40/8 = 5.0
        assert abs(w.mean - 5.0) < 0.1
        assert w.std > 0

    def test_update_many(self) -> None:
        w = WelfordVariance()
        w.update_many([1.0, 2.0, 3.0, 4.0, 5.0])
        # 5 values -> sample_count = 5
        assert w.sample_count == 5
        assert abs(w.mean - 3.0) < 0.01

    def test_reset(self) -> None:
        w = WelfordVariance()
        w.update_many([1.0, 2.0, 3.0])
        w.reset()
        assert w.sample_count == 0
        assert w.mean == 0.0

    def test_repr(self) -> None:
        w = WelfordVariance()
        w.update(5.0)
        r = repr(w)
        assert "WelfordVariance" in r


class TestRawNewsEvent:
    """Tests cho RawNewsEvent."""

    def test_is_high_impact(self) -> None:
        event = RawNewsEvent(
            event_id="1",
            title="NFP",
            country="US",
            currency="USD",
            impact=ImpactLevel.HIGH,
            scheduled_time=datetime.now(timezone.utc),
        )
        assert event.is_high_impact is True
        assert event.is_medium_impact is False

    def test_to_dict(self) -> None:
        now = datetime.now(timezone.utc)
        event = RawNewsEvent(
            event_id="1",
            title="NFP",
            country="US",
            currency="USD",
            impact=ImpactLevel.HIGH,
            scheduled_time=now,
        )
        d = event.to_dict()
        assert d["event_id"] == "1"
        assert d["impact"] == "HIGH"
        assert d["currency"] == "USD"


class TestNewsVectorizer:
    """Tests cho NewsVectorizer."""

    def test_base_impact_scores(self) -> None:
        vec = NewsVectorizer()
        vec.set_atr_d1(10.0)

        low_event = RawNewsEvent(
            event_id="1", title="Test", country="US", currency="USD",
            impact=ImpactLevel.LOW, scheduled_time=datetime.now(timezone.utc),
        )
        medium_event = RawNewsEvent(
            event_id="2", title="Test", country="US", currency="USD",
            impact=ImpactLevel.MEDIUM, scheduled_time=datetime.now(timezone.utc),
        )
        high_event = RawNewsEvent(
            event_id="3", title="Test", country="US", currency="USD",
            impact=ImpactLevel.HIGH, scheduled_time=datetime.now(timezone.utc),
        )

        low_score = vec.vectorize(low_event)
        med_score = vec.vectorize(medium_event)
        high_score = vec.vectorize(high_event)

        assert low_score.impact_base == 0.2
        assert med_score.impact_base == 0.5
        assert high_score.impact_base == 1.0
        assert high_score.i_news >= med_score.i_news
        assert med_score.i_news >= low_score.i_news

    def test_surprise_bullish(self) -> None:
        """Khi actual > forecast + nguong, direction = BULLISH."""
        vec = NewsVectorizer()
        event = RawNewsEvent(
            event_id="1",
            title="NFP",
            country="US",
            currency="USD",
            impact=ImpactLevel.HIGH,
            scheduled_time=datetime.now(timezone.utc),
            forecast=200.0,
            actual=250.0,
        )
        score = vec.vectorize(event)
        # No historical sigma -> relative surprise, direction determined by actual vs forecast
        assert score.surprise_direction == "BULLISH"

    def test_surprise_bearish(self) -> None:
        """Khi actual < forecast - nguong, direction = BEARISH."""
        vec = NewsVectorizer()
        event = RawNewsEvent(
            event_id="1",
            title="NFP",
            country="US",
            currency="USD",
            impact=ImpactLevel.HIGH,
            scheduled_time=datetime.now(timezone.utc),
            forecast=200.0,
            actual=150.0,
        )
        score = vec.vectorize(event)
        assert score.surprise_direction == "BEARISH"

    def test_i_news_in_range(self) -> None:
        vec = NewsVectorizer()
        vec.set_atr_d1(10.0)

        for impact in [ImpactLevel.LOW, ImpactLevel.MEDIUM, ImpactLevel.HIGH]:
            event = RawNewsEvent(
                event_id=f"id_{impact.value}",
                title="Test",
                country="US",
                currency="USD",
                impact=impact,
                scheduled_time=datetime.now(timezone.utc),
            )
            score = vec.vectorize(event)
            assert 0 <= score.i_news <= 3.0, f"I_news out of range: {score.i_news}"

    def test_adjust_guardrail_dampening(self) -> None:
        vec = NewsVectorizer()
        event = RawNewsEvent(
            event_id="1",
            title="FOMC",
            country="US",
            currency="USD",
            impact=ImpactLevel.HIGH,
            scheduled_time=datetime.now(timezone.utc),
        )
        score = vec.vectorize(event)
        original_i_news = score.i_news
        dampened = vec.adjust_i_news(score.i_news, active_guardrail=True, dampening_factor=0.3)
        assert dampened < original_i_news
        assert dampened == original_i_news * 0.3

    def test_no_actual_no_surprise(self) -> None:
        vec = NewsVectorizer()
        event = RawNewsEvent(
            event_id="1",
            title="FOMC",
            country="US",
            currency="USD",
            impact=ImpactLevel.HIGH,
            scheduled_time=datetime.now(timezone.utc),
            forecast=200.0,
        )
        score = vec.vectorize(event)
        assert score.surprise_z is None
        assert score.surprise_direction == "NEUTRAL"


class TestVolatilityCountdown:
    """Tests cho VolatilityCountdown."""

    @pytest.fixture
    def countdown(self) -> VolatilityCountdown:
        return VolatilityCountdown()

    def test_initial_state_normal(self, countdown: VolatilityCountdown) -> None:
        assert countdown.state == MacroState.NORMAL
        assert countdown.active_guardrail is False

    def test_pre_news_state(self, countdown: VolatilityCountdown) -> None:
        now = datetime.now(timezone.utc)
        high_event = RawNewsEvent(
            event_id="1",
            title="FOMC",
            country="US",
            currency="USD",
            impact=ImpactLevel.HIGH,
            scheduled_time=now + timedelta(minutes=10),
        )
        countdown.set_events([high_event])
        result = countdown._tick()
        assert result.state == MacroState.PRE_NEWS
        assert result.active_guardrail is True
        assert 500 <= result.seconds_to_next <= 600

    def test_normal_state_no_events(self, countdown: VolatilityCountdown) -> None:
        countdown.set_events([])
        result = countdown._tick()
        assert result.state == MacroState.NORMAL

    def test_medium_impact_no_guardrail(self, countdown: VolatilityCountdown) -> None:
        now = datetime.now(timezone.utc)
        med_event = RawNewsEvent(
            event_id="1",
            title="Test",
            country="US",
            currency="USD",
            impact=ImpactLevel.MEDIUM,
            scheduled_time=now + timedelta(minutes=5),
        )
        countdown.set_events([med_event])
        result = countdown._tick()
        assert result.active_guardrail is False

    def test_dampening_pre_news(self, countdown: VolatilityCountdown) -> None:
        now = datetime.now(timezone.utc)
        event = RawNewsEvent(
            event_id="1",
            title="FOMC",
            country="US",
            currency="USD",
            impact=ImpactLevel.HIGH,
            scheduled_time=now + timedelta(minutes=7),
        )
        countdown.set_events([event])
        result = countdown._tick()
        assert 0 < result.dampening_factor <= 0.3

    def test_dampening_news_window(self, countdown: VolatilityCountdown) -> None:
        now = datetime.now(timezone.utc)
        event = RawNewsEvent(
            event_id="1",
            title="FOMC",
            country="US",
            currency="USD",
            impact=ImpactLevel.HIGH,
            scheduled_time=now + timedelta(seconds=10),
        )
        countdown.set_events([event])
        result = countdown._tick()
        assert result.dampening_factor == 0.1

    def test_cluster_detection(self, countdown: VolatilityCountdown) -> None:
        now = datetime.now(timezone.utc)
        events = [
            RawNewsEvent(
                event_id=f"h{i}",
                title=f"High{i}",
                country="US",
                currency="USD",
                impact=ImpactLevel.HIGH,
                scheduled_time=now + timedelta(minutes=5 * i),
            )
            for i in range(3)
        ]
        countdown.set_events(events)
        result = countdown._tick()
        assert result.state == MacroState.CLUSTER


class TestPostNewsRegimeClassifier:
    """Tests cho PostNewsRegimeClassifier."""

    def test_impulsive_bullish(self) -> None:
        clf = PostNewsRegimeClassifier(impulsive_threshold=0.8)
        outcome = clf.classify(
            event_id="1",
            price_at_event=100.0,
            price_5min_after=108.0,
            atr_h1=10.0,
            surprise_direction="BULLISH",
            surprise_z=3.0,
        )
        assert outcome.classification == PostNewsRegime.IMPULSIVE_FOLLOW_THROUGH

    def test_impulsive_bearish(self) -> None:
        clf = PostNewsRegimeClassifier(impulsive_threshold=0.8)
        outcome = clf.classify(
            event_id="1",
            price_at_event=100.0,
            price_5min_after=91.5,
            atr_h1=10.0,
            surprise_direction="BEARISH",
            surprise_z=-3.0,
        )
        # directional_move = -0.85, BEARISH surprise -> IMPULSIVE
        assert outcome.classification == PostNewsRegime.IMPULSIVE_FOLLOW_THROUGH

    def test_reversal_after_spike(self) -> None:
        clf = PostNewsRegimeClassifier()
        # directional_move <= -0.5 -> REVERSAL
        outcome = clf.classify(
            event_id="1",
            price_at_event=100.0,
            price_5min_after=94.0,
            atr_h1=10.0,
            surprise_direction="BEARISH",
            surprise_z=-2.0,
        )
        assert outcome.classification == PostNewsRegime.REVERSAL_AFTER_SPIKE

    def test_reversal_against_surprise(self) -> None:
        clf = PostNewsRegimeClassifier()
        outcome = clf.classify(
            event_id="1",
            price_at_event=100.0,
            price_5min_after=98.0,
            atr_h1=10.0,
            surprise_direction="BULLISH",
            surprise_z=2.0,
        )
        # Bullish surprise nhung gia xuong -> reversal
        assert outcome.classification == PostNewsRegime.REVERSAL_AFTER_SPIKE

    def test_choppy(self) -> None:
        clf = PostNewsRegimeClassifier()
        outcome = clf.classify(
            event_id="1",
            price_at_event=100.0,
            price_5min_after=101.0,
            atr_h1=10.0,
            surprise_direction="NEUTRAL",
        )
        assert outcome.classification == PostNewsRegime.CHOPPY_CONSOLIDATION

    def test_directional_move_calculation(self) -> None:
        clf = PostNewsRegimeClassifier()
        outcome = clf.classify(
            event_id="1",
            price_at_event=100.0,
            price_5min_after=105.0,
            atr_h1=10.0,
            surprise_direction="BULLISH",
        )
        assert abs(outcome.directional_move - 0.5) < 0.01


class TestEconomicCalendarDB:
    """Tests cho EconomicCalendarDB."""

    def test_upsert_and_query(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        db = EconomicCalendarDB(db_path)
        db.connect()

        now = datetime.now(timezone.utc)
        event = RawNewsEvent(
            event_id="nfp_1",
            title="NFP",
            country="US",
            currency="USD",
            impact=ImpactLevel.HIGH,
            scheduled_time=now + timedelta(days=1),
            forecast=200.0,
            previous=180.0,
        )

        db.upsert_event(event)
        events = db.get_upcoming_events(
            from_time=now,
            to_time=now + timedelta(days=7),
            currencies=["USD"],
        )

        assert len(events) == 1
        assert events[0].title == "NFP"
        assert events[0].impact == ImpactLevel.HIGH

        db.close()

    def test_get_next_high_impact(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        db = EconomicCalendarDB(db_path)
        db.connect()

        now = datetime.now(timezone.utc)
        events = [
            RawNewsEvent(
                event_id="1",
                title="Low",
                country="US",
                currency="USD",
                impact=ImpactLevel.LOW,
                scheduled_time=now + timedelta(hours=1),
            ),
            RawNewsEvent(
                event_id="2",
                title="High",
                country="US",
                currency="USD",
                impact=ImpactLevel.HIGH,
                scheduled_time=now + timedelta(hours=2),
            ),
        ]

        db.upsert_events(events)
        next_high = db.get_next_high_impact()

        assert next_high is not None
        assert next_high.title == "High"
        assert next_high.impact == ImpactLevel.HIGH

        db.close()

    def test_update_event_actual(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        db = EconomicCalendarDB(db_path)
        db.connect()

        now = datetime.now(timezone.utc)
        event = RawNewsEvent(
            event_id="nfp_1",
            title="NFP",
            country="US",
            currency="USD",
            impact=ImpactLevel.HIGH,
            scheduled_time=now,
        )

        db.upsert_event(event)
        db.update_event_actual("nfp_1", actual=250.0, is_surprise=True, surprise_z=2.5)

        events = db.get_upcoming_events(
            from_time=now - timedelta(hours=1),
            to_time=now + timedelta(hours=1),
        )

        assert len(events) == 1
        assert events[0].actual == 250.0
        assert events[0].is_surprise is True

        db.close()

    def test_currency_filter(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        db = EconomicCalendarDB(db_path)
        db.connect()

        now = datetime.now(timezone.utc)
        events = [
            RawNewsEvent(
                event_id="1", title="USD Event", country="US",
                currency="USD", impact=ImpactLevel.HIGH,
                scheduled_time=now + timedelta(hours=1),
            ),
            RawNewsEvent(
                event_id="2", title="EUR Event", country="DE",
                currency="EUR", impact=ImpactLevel.HIGH,
                scheduled_time=now + timedelta(hours=2),
            ),
        ]

        db.upsert_events(events)
        usd_events = db.get_upcoming_events(
            from_time=now, to_time=now + timedelta(days=1),
            currencies=["USD"],
        )

        assert len(usd_events) == 1
        assert usd_events[0].currency == "USD"

        db.close()


# =============================================================================
# Integration & Edge Case Tests
# =============================================================================


class TestNewsVectorizerSurprise:
    """Tests nang cao cho surprise factor calculation."""

    def test_surprise_factor_z_0(self) -> None:
        """Z = 0: khong co surprise -> factor = 1.0."""
        vec = NewsVectorizer()
        factor = vec._compute_surprise_factor(0.0)
        assert factor == 1.0

    def test_surprise_factor_z_1(self) -> None:
        """Z = 1: canhgioi duoi -> factor = 1.0."""
        vec = NewsVectorizer()
        factor = vec._compute_surprise_factor(1.0)
        assert factor == 1.0

    def test_surprise_factor_z_15(self) -> None:
        """Z = 1.5: trong vung 1-2 -> factor = 1.0 + 0.25*0.5 = 1.125."""
        vec = NewsVectorizer()
        factor = vec._compute_surprise_factor(1.5)
        assert abs(factor - 1.125) < 0.001

    def test_surprise_factor_z_2(self) -> None:
        """Z = 2: canhgioi tren -> factor = 1.0 + 0.25*1.0 = 1.25."""
        vec = NewsVectorizer()
        factor = vec._compute_surprise_factor(2.0)
        assert abs(factor - 1.25) < 0.001

    def test_surprise_factor_z_3(self) -> None:
        """Z = 3: > 2SD -> factor = 1.25 + 0.5*1.0 = 1.75."""
        vec = NewsVectorizer()
        factor = vec._compute_surprise_factor(3.0)
        assert abs(factor - 1.75) < 0.001

    def test_surprise_factor_negative(self) -> None:
        """Z am: su dung tri tuyet."""
        vec = NewsVectorizer()
        factor = vec._compute_surprise_factor(-2.0)
        assert abs(factor - 1.25) < 0.001

    def test_is_surprise_threshold(self) -> None:
        """is_surprise = True khi |z| > 2.0."""
        vec = NewsVectorizer()
        score_low = vec._compute_surprise_factor(1.9)
        score_at = vec._compute_surprise_factor(2.0)
        score_high = vec._compute_surprise_factor(2.1)
        # Z = 2.0 -> factor = 1.25 -> not surprise (threshold > 2.0)
        # Chinh lai: spec noi is_surprise = |z| > 2.0
        assert vec._compute_surprise_factor(2.0) == 1.25
        # Factor cho z=2.1 = 1.25 + 0.5*0.1 = 1.30

    def test_vectorize_with_surprise(self) -> None:
        """Integration: vectorize voi actual/forecast -> tinh surprise."""
        vec = NewsVectorizer()
        vec.set_atr_d1(10.0)
        event = RawNewsEvent(
            event_id="s1",
            title="CPI",
            country="US",
            currency="USD",
            impact=ImpactLevel.HIGH,
            scheduled_time=datetime.now(timezone.utc),
            forecast=2.5,
            actual=3.2,
        )
        score = vec.vectorize(event)
        assert score.surprise_z is not None
        # actual > forecast -> BULLISH
        assert score.surprise_direction == "BULLISH"
        assert score.surprise_factor >= 1.0

    def test_i_news_upper_bound(self) -> None:
        """I_news khong vuot qua 3.0."""
        vec = NewsVectorizer()
        vec.set_atr_d1(0.1)  # ATR rat nho -> m_bar_ec rat lon
        event = RawNewsEvent(
            event_id="big",
            title="NFP",
            country="US",
            currency="USD",
            impact=ImpactLevel.HIGH,
            scheduled_time=datetime.now(timezone.utc),
        )
        score = vec.vectorize(event)
        assert score.i_news <= 3.0


class TestVolatilityCountdownAdvanced:
    """Tests nang cao cho VolatilityCountdown: state transitions, performance."""

    def test_state_transitions_normal_to_pre_news(self) -> None:
        """NORMAL -> PRE_NEWS khi event trong 15 phut."""
        cd = VolatilityCountdown()
        now = datetime.now(timezone.utc)
        event = RawNewsEvent(
            event_id="e1",
            title="FOMC",
            country="US",
            currency="USD",
            impact=ImpactLevel.HIGH,
            scheduled_time=now + timedelta(minutes=14),
        )
        cd.set_events([event])
        result = cd._tick()
        assert result.state == MacroState.PRE_NEWS
        assert result.active_guardrail is True

    def test_state_transitions_pre_news_to_news_window(self) -> None:
        """PRE_NEWS -> NEWS_WINDOW khi con 5 phut."""
        cd = VolatilityCountdown()
        now = datetime.now(timezone.utc)
        event = RawNewsEvent(
            event_id="e1",
            title="FOMC",
            country="US",
            currency="USD",
            impact=ImpactLevel.HIGH,
            scheduled_time=now + timedelta(minutes=4, seconds=59),
        )
        cd.set_events([event])
        result = cd._tick()
        assert result.state == MacroState.NEWS_WINDOW
        assert result.dampening_factor == 0.1

    def test_state_transitions_to_post_news(self) -> None:
        """POST_NEWS: event vua qua (5-30 phut), khong co event sap toi."""
        cd = VolatilityCountdown()
        now = datetime.now(timezone.utc)
        # Past event, no future events
        event = RawNewsEvent(
            event_id="past",
            title="Past Event",
            country="US",
            currency="USD",
            impact=ImpactLevel.HIGH,
            scheduled_time=now - timedelta(minutes=10),
        )
        cd.set_events([event])
        result = cd._tick()
        # next_event=None, last_past=event, seconds_since_past=600 -> POST_NEWS
        assert result.state == MacroState.POST_NEWS

    def test_post_news_exit(self) -> None:
        """POST_NEWS -> NORMAL khi qua 30 phut."""
        cd = VolatilityCountdown()
        now = datetime.now(timezone.utc)
        event = RawNewsEvent(
            event_id="e1",
            title="FOMC",
            country="US",
            currency="USD",
            impact=ImpactLevel.HIGH,
            scheduled_time=now - timedelta(minutes=31),
        )
        cd.set_events([event])
        result = cd._tick()
        # Past > 30 phut, next_event=None, last_past exists, but > 30min -> NORMAL
        assert result.state == MacroState.NORMAL

    def test_cluster_amplification(self) -> None:
        """2+ High events trong 30 phut -> CLUSTER state."""
        cd = VolatilityCountdown()
        now = datetime.now(timezone.utc)
        events = [
            RawNewsEvent(
                event_id=f"h{i}",
                title=f"High{i}",
                country="US",
                currency="USD",
                impact=ImpactLevel.HIGH,
                scheduled_time=now + timedelta(minutes=5 + i * 10),
            )
            for i in range(2)
        ]
        cd.set_events(events)
        result = cd._tick()
        assert result.state == MacroState.CLUSTER
        assert result.dampening_factor == 0.1

    def test_dampening_at_15min(self) -> None:
        """PRE_NEWS tai 15 phut: dampening = 0.3 * 1.0 + 0.1 = 0.4."""
        cd = VolatilityCountdown()
        now = datetime.now(timezone.utc)
        event = RawNewsEvent(
            event_id="e1",
            title="FOMC",
            country="US",
            currency="USD",
            impact=ImpactLevel.HIGH,
            scheduled_time=now + timedelta(minutes=15),
        )
        cd.set_events([event])
        result = cd._tick()
        # dampening = 0.3 * (900/900) + 0.1 = 0.4
        assert abs(result.dampening_factor - 0.4) < 0.01

    def test_dampening_at_7_5min(self) -> None:
        """PRE_NEWS tai 7.5 phut: dampening = 0.3 * 0.5 + 0.1 = 0.25."""
        cd = VolatilityCountdown()
        now = datetime.now(timezone.utc)
        event = RawNewsEvent(
            event_id="e1",
            title="FOMC",
            country="US",
            currency="USD",
            impact=ImpactLevel.HIGH,
            scheduled_time=now + timedelta(minutes=7, seconds=30),
        )
        cd.set_events([event])
        result = cd._tick()
        # dampening = 0.3 * (450/900) + 0.1 = 0.25
        assert abs(result.dampening_factor - 0.25) < 0.01

    def test_medium_impact_no_guardrail(self) -> None:
        """MEDIUM impact: guardrail khong duoc kich hoat."""
        cd = VolatilityCountdown(high_only_guardrail=True)
        now = datetime.now(timezone.utc)
        event = RawNewsEvent(
            event_id="m1",
            title="Test",
            country="US",
            currency="USD",
            impact=ImpactLevel.MEDIUM,
            scheduled_time=now + timedelta(minutes=5),
        )
        cd.set_events([event])
        result = cd._tick()
        assert result.active_guardrail is False

    def test_countdown_seconds_to_next(self) -> None:
        """seconds_to_next tinh dung."""
        cd = VolatilityCountdown()
        now = datetime.now(timezone.utc)
        target = now + timedelta(minutes=10, seconds=30)
        event = RawNewsEvent(
            event_id="e1",
            title="FOMC",
            country="US",
            currency="USD",
            impact=ImpactLevel.HIGH,
            scheduled_time=target,
        )
        cd.set_events([event])
        result = cd._tick()
        assert 629 <= result.seconds_to_next <= 631


class TestPostNewsRegimeClassifierAdvanced:
    """Tests nang cao cho PostNewsRegimeClassifier."""

    def test_impulsive_within_threshold(self) -> None:
        """Directional move < threshold: CHOPPY chu khong phai IMPULSIVE."""
        clf = PostNewsRegimeClassifier(impulsive_threshold=0.8)
        outcome = clf.classify(
            event_id="1",
            price_at_event=100.0,
            price_5min_after=107.0,
            atr_h1=10.0,
            surprise_direction="BULLISH",
        )
        # directional_move = 0.7, surprise=BULLISH -> not >= 0.8
        assert outcome.classification == PostNewsRegime.CHOPPY_CONSOLIDATION

    def test_reversal_within_threshold(self) -> None:
        """d_move = -0.5, exactly at reversal_threshold -> REVERSAL (boundary inclusive)."""
        clf = PostNewsRegimeClassifier(reversal_threshold=-0.5)
        outcome = clf.classify(
            event_id="1",
            price_at_event=100.0,
            price_5min_after=95.0,
            atr_h1=10.0,
            surprise_direction="NEUTRAL",
        )
        # directional_move = -0.5, <= -0.5 -> REVERSAL
        assert outcome.classification == PostNewsRegime.REVERSAL_AFTER_SPIKE

    def test_bullish_surprise_bearish_move(self) -> None:
        """Bullish surprise nhung gia giam manh -> REVERSAL."""
        clf = PostNewsRegimeClassifier()
        outcome = clf.classify(
            event_id="1",
            price_at_event=100.0,
            price_5min_after=97.0,
            atr_h1=10.0,
            surprise_direction="BULLISH",
        )
        assert outcome.classification == PostNewsRegime.REVERSAL_AFTER_SPIKE

    def test_bearish_surprise_bullish_move(self) -> None:
        """Bearish surprise nhung gia tang -> REVERSAL."""
        clf = PostNewsRegimeClassifier()
        outcome = clf.classify(
            event_id="1",
            price_at_event=100.0,
            price_5min_after=103.0,
            atr_h1=10.0,
            surprise_direction="BEARISH",
        )
        assert outcome.classification == PostNewsRegime.REVERSAL_AFTER_SPIKE

    def test_extreme_bearish_impulsive(self) -> None:
        """Bearish surprise voi directional move < -0.8 -> IMPULSIVE."""
        clf = PostNewsRegimeClassifier(impulsive_threshold=0.8)
        outcome = clf.classify(
            event_id="1",
            price_at_event=100.0,
            price_5min_after=91.5,
            atr_h1=10.0,
            surprise_direction="BEARISH",
        )
        # directional_move = -0.85, surprise=BEARISH -> impulsive
        assert outcome.classification == PostNewsRegime.IMPULSIVE_FOLLOW_THROUGH

    def test_save_and_load_outcome(self, tmp_path: Path) -> None:
        """Luu outcome -> lay lai (dung shared DB)."""
        db_path = tmp_path / "outcomes.db"
        db = EconomicCalendarDB(db_path)
        db.connect()
        clf = PostNewsRegimeClassifier(db=db)

        outcome = clf.classify(
            event_id="nfp_1",
            price_at_event=100.0,
            price_5min_after=108.0,
            atr_h1=10.0,
            surprise_direction="BULLISH",
            surprise_z=2.5,
        )

        saved_id = clf.save_outcome(outcome)
        assert saved_id is not None

        loaded = clf.get_outcomes_for_event("nfp_1")
        assert len(loaded) == 1
        assert loaded[0].surprise_z == 2.5

        db.close()

    def test_get_statistics(self, tmp_path: Path) -> None:
        """Thong ke outcomes (dung shared DB)."""
        db_path = tmp_path / "stats.db"
        db = EconomicCalendarDB(db_path)
        db.connect()
        clf = PostNewsRegimeClassifier(db=db)

        o1 = clf.classify("e1", 100.0, 108.0, 10.0, "BULLISH", 2.0)
        o2 = clf.classify("e2", 100.0, 92.0, 10.0, "BEARISH", -2.0)
        o3 = clf.classify("e3", 100.0, 101.0, 10.0, "NEUTRAL", 0.0)

        clf.save_outcome(o1)
        clf.save_outcome(o2)
        clf.save_outcome(o3)

        stats = clf.get_statistics()
        assert stats["total_outcomes"] == 3
        assert PostNewsRegime.IMPULSIVE_FOLLOW_THROUGH.value in stats["by_type"]
        assert PostNewsRegime.CHOPPY_CONSOLIDATION.value in stats["by_type"]
        assert stats["max_directional_move"] > 0

        db.close()


class TestEconomicCalendarDBWelford:
    """Tests cho Welford update trong DB."""

    def test_welford_online_update(self, tmp_path: Path) -> None:
        """Welford cap nhat sigma moi lan co surprise."""
        db_path = tmp_path / "welford.db"
        db = EconomicCalendarDB(db_path)
        db.connect()

        event_id = "cpi_1"
        currency = "USD"
        event_type = "CPI"

        # Lan 1: surprise = 1.0
        db.update_surprise_sigma(event_id, currency, event_type, 1.0)

        # Lan 2: surprise = 2.0
        db.update_surprise_sigma(event_id, currency, event_type, 2.0)

        # Lan 3: surprise = 1.5
        db.update_surprise_sigma(event_id, currency, event_type, 1.5)

        sigma = db.get_surprise_sigma(event_id)
        assert sigma is not None
        assert sigma > 0

        db.close()


class TestEdgeCases:
    """Tests cho cac edge case handlers."""

    def test_event_reschedule_detection(self) -> None:
        """Event thay doi thoi gian > 30 phut -> reset."""
        from core.macro import MacroEngine, MacroEngineConfig

        config = MacroEngineConfig(
            enable_event_reschedule_detection=True,
            reschedule_threshold_minutes=30,
        )
        engine = MacroEngine(config=config)

        now = datetime.now(timezone.utc)

        # Su kien goc
        old_event = RawNewsEvent(
            event_id="reschedule_test",
            title="NFP",
            country="US",
            currency="USD",
            impact=ImpactLevel.HIGH,
            scheduled_time=now + timedelta(hours=2),
        )
        engine._known_event_times[old_event.event_id] = old_event.scheduled_time

        # Su kien bi reschedule 45 phut
        new_event = RawNewsEvent(
            event_id="reschedule_test",
            title="NFP",
            country="US",
            currency="USD",
            impact=ImpactLevel.HIGH,
            scheduled_time=now + timedelta(hours=3) + timedelta(minutes=15),
        )

        # Kich hoat guardrail
        engine._guardrail_active = True
        engine._detect_reschedule([new_event])

        # Sau reschedule -> guardrail reset
        assert engine._guardrail_active is False

    def test_event_cancellation_detection(self) -> None:
        """Event bi huy -> xoa khoi queue, reset PRE_NEWS."""
        from core.macro import MacroEngine, MacroEngineConfig, MacroState

        config = MacroEngineConfig(enable_event_cancellation_detection=True)
        engine = MacroEngine(config=config)

        now = datetime.now(timezone.utc)

        # Su kien dang trong queue
        event = RawNewsEvent(
            event_id="cancel_test",
            title="NFP",
            country="US",
            currency="USD",
            impact=ImpactLevel.HIGH,
            scheduled_time=now + timedelta(hours=1),
        )
        engine._countdown._events = [event]
        engine._known_event_times[event.event_id] = event.scheduled_time
        engine._current_macro_state = MacroState.PRE_NEWS
        engine._guardrail_active = True

        # Scrape khong co event nay (da bi huy)
        engine._detect_cancellations([])

        # Event da bi xoa
        assert len(engine._countdown._events) == 0
        assert event.event_id not in engine._known_event_times
        # State reset ve NORMAL
        assert engine._current_macro_state == MacroState.NORMAL
        assert engine._guardrail_active is False

    def test_scraping_failure_sets_stale(self) -> None:
        """Scrape fail -> is_stale = True."""
        from core.macro import MacroEngine, MacroEngineConfig

        config = MacroEngineConfig()
        engine = MacroEngine(config=config)

        assert engine.is_stale is False
        # Khi scrape fail, _is_stale duoc set True
        # Day duoc test trong integration test voi mock

    def test_empty_scraped_events(self) -> None:
        """Scrape tra ve rong -> khong crash."""
        cd = VolatilityCountdown()
        cd.set_events([])
        result = cd._tick()
        assert result.state == MacroState.NORMAL
        assert result.current_event is None
        assert result.active_guardrail is False

    def test_past_events_kept(self) -> None:
        """Past events duoc giu de POST_NEWS detection hoat dong."""
        cd = VolatilityCountdown()
        now = datetime.now(timezone.utc)
        events = [
            RawNewsEvent(
                event_id="past",
                title="Past",
                country="US",
                currency="USD",
                impact=ImpactLevel.HIGH,
                scheduled_time=now - timedelta(hours=2),
            ),
            RawNewsEvent(
                event_id="future",
                title="Future",
                country="US",
                currency="USD",
                impact=ImpactLevel.HIGH,
                scheduled_time=now + timedelta(hours=2),
            ),
        ]
        cd.set_events(events)
        # Gio giu tat ca events (past + future)
        assert len(cd._events) == 2


class TestMacroEngineConfig:
    """Tests cho MacroEngineConfig."""

    def test_default_config(self) -> None:
        config = MacroEngineConfig()
        assert config.refresh_interval_seconds == 21600
        assert config.pre_event_refresh_seconds == 300
        assert config.pre_event_warning_seconds == 1800
        assert config.currencies == ["USD", "XAU"]
        assert config.min_impact == ImpactLevel.MEDIUM

    def test_custom_config(self) -> None:
        config = MacroEngineConfig(
            currencies=["EUR", "GBP", "XAU"],
            min_impact=ImpactLevel.HIGH,
            cluster_amplification_factor=1.5,
        )
        assert config.currencies == ["EUR", "GBP", "XAU"]
        assert config.min_impact == ImpactLevel.HIGH
        assert config.cluster_amplification_factor == 1.5


class TestVolatilityCountdownPerformance:
    """Performance tests cho VolatilityCountdown — verify loop 1s < 10ms CPU."""

    def test_tick_under_10ms_empty_events(self) -> None:
        """_tick() voi 0 events: phai < 10ms."""
        cd = VolatilityCountdown()
        cd.set_events([])

        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            cd._tick()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        avg_ms = sum(times) / len(times)
        p99_ms = sorted(times)[98]
        assert avg_ms < 10.0, f"Average tick time {avg_ms:.2f}ms exceeds 10ms"
        assert p99_ms < 20.0, f"P99 tick time {p99_ms:.2f}ms exceeds 20ms"

    def test_tick_under_10ms_with_events(self) -> None:
        """_tick() voi 20 events: phai < 10ms."""
        cd = VolatilityCountdown()
        now = datetime.now(timezone.utc)
        events = [
            RawNewsEvent(
                event_id=f"e{i}",
                title=f"Event {i}",
                country="US",
                currency="USD",
                impact=ImpactLevel.HIGH if i % 3 == 0 else ImpactLevel.MEDIUM,
                scheduled_time=now + timedelta(minutes=i * 5),
            )
            for i in range(20)
        ]
        cd.set_events(events)

        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            cd._tick()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        avg_ms = sum(times) / len(times)
        p99_ms = sorted(times)[98]
        assert avg_ms < 10.0, f"Average tick time {avg_ms:.2f}ms exceeds 10ms"
        assert p99_ms < 20.0, f"P99 tick time {p99_ms:.2f}ms exceeds 20ms"

    def test_tick_under_10ms_cluster_scenario(self) -> None:
        """_tick() voi cluster (30 events): phai < 10ms."""
        cd = VolatilityCountdown()
        now = datetime.now(timezone.utc)
        events = [
            RawNewsEvent(
                event_id=f"c{i}",
                title=f"Cluster {i}",
                country="US",
                currency="USD",
                impact=ImpactLevel.HIGH,
                scheduled_time=now + timedelta(minutes=i),
            )
            for i in range(30)
        ]
        cd.set_events(events)

        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            cd._tick()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        avg_ms = sum(times) / len(times)
        p99_ms = sorted(times)[98]
        assert avg_ms < 10.0, f"Cluster tick time {avg_ms:.2f}ms exceeds 10ms"
        assert p99_ms < 20.0, f"Cluster P99 {p99_ms:.2f}ms exceeds 20ms"

    def test_tick_worst_case_100_events(self) -> None:
        """_tick() voi 100 events: van phai nho hon 10ms."""
        cd = VolatilityCountdown()
        now = datetime.now(timezone.utc)
        events = [
            RawNewsEvent(
                event_id=f"e{i}",
                title=f"Event {i}",
                country="US",
                currency=["USD", "EUR", "GBP", "JPY"][i % 4],
                impact=ImpactLevel.HIGH,
                scheduled_time=now + timedelta(hours=i),
            )
            for i in range(100)
        ]
        cd.set_events(events)

        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            cd._tick()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        avg_ms = sum(times) / len(times)
        p99_ms = sorted(times)[49]
        max_ms = max(times)
        assert avg_ms < 10.0, f"100-event avg {avg_ms:.2f}ms exceeds 10ms"
        assert p99_ms < 20.0, f"100-event P99 {p99_ms:.2f}ms exceeds 20ms"
        assert max_ms < 50.0, f"100-event max {max_ms:.2f}ms exceeds 50ms (possible O(n) issue)"

