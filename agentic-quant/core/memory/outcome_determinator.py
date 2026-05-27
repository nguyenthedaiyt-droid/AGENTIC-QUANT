# =============================================================================
# AGENTIC-QUANT — Outcome Determination Engine
# Background worker triggered by BAR_CLOSE(M1) events
# =============================================================================

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

from core.memory.long_term.sqlite_history_store import SQLiteHistoryStore
from core.memory.short_term.redis_cache_manager import RedisCacheManager
from core.memory.models import ModelAPrediction, PredictionOutcome, Zone
from core.memory.models.enums import OutcomeType, Timeframe, ZoneStatus

if TYPE_CHECKING:
    from core.utils.events import EventBus


# =============================================================================
# Constants
# =============================================================================
MAX_HORIZON_MS: int = 240 * 60 * 1000  # 240 minutes in ms
ZONE_HOLD_BARS: int = 3  # Zone phai duoc giu trong 3 bars de la HOLD
ZONE_TOUCH_TOLERANCE_PIPS: float = 2.0  # Tolerance pip de kiem tra zone touch


# =============================================================================
# Price Series Helper
# =============================================================================
@dataclass
class PriceBar:
    """Mot bar OHLCV."""

    close_time: int  # Unix ms
    open: float
    high: float
    low: float
    close: float


@dataclass
class PriceSeries:
    """Chuoi gia sau mot prediction."""

    symbol: str
    bars: list[PriceBar] = field(default_factory=list)
    prediction_bar_close_time: int = 0  # Unix ms

    def high_since_prediction(self) -> float:
        """Gia cao nhat kể tu prediction bar close."""
        if not self.bars:
            return 0.0
        return max(b.high for b in self.bars)

    def low_since_prediction(self) -> float:
        """Gia thap nhat ke tu prediction bar close."""
        if not self.bars:
            return 0.0
        return min(b.low for b in self.bars)

    def elapsed_ms(self, current_time: int) -> int:
        """So ms da troi qua kể tu prediction."""
        return current_time - self.prediction_bar_close_time

    def is_timeout(self, current_time: int) -> bool:
        """Kiem tra xem da vuot horizon chua."""
        return self.elapsed_ms(current_time) > MAX_HORIZON_MS

    def bar_count(self) -> int:
        """So bar kể tu prediction."""
        return len(self.bars)


# =============================================================================
# Outcome Determination Result
# =============================================================================
@dataclass
class OutcomeResult:
    """Ket qua xac dinh outcome cho mot prediction."""

    prediction_id: str
    symbol: str
    bar_close_time: int
    outcome: OutcomeType
    outcome_time: int  # Unix ms
    elapsed_minutes: float
    high_since_prediction: float
    low_since_prediction: float

    # Chi tiet zone hold
    zone_hold_bars: int = 0
    zone_touched: bool = False


# =============================================================================
# Outcome Determinator
# =============================================================================
class OutcomeDeterminator:
    """
    Xac dinh ket qua prediction.

    Triggered boi BAR_CLOSE(M1) event:
    1. Query predictions WHERE outcome_determined = 0
    2. For each prediction:
       - BSL hit: high_since_prediction >= predicted_bsl_level
       - SSL hit: low_since_prediction <= predicted_ssl_level
       - Timeout: elapsed > MAX_HORIZON (240 min)
       - Zone hold: zone touched AND held for 3 bars
    3. Update predictions in SQLite
    4. Publish OUTCOME_CONFIRMED event to EventBus

    Integration:
    - Phase 1: BAR_CLOSE event trigger
    - Phase 2: RegimeClassifier (macro_regime context)
    - Phase 3: SQLiteHistoryStore (read/write predictions)
    - Redis: Zone Registry (zone hold detection)
    """

    def __init__(
        self,
        store: SQLiteHistoryStore,
        redis: RedisCacheManager,
        event_bus: EventBus | None = None,
    ) -> None:
        self._store = store
        self._redis = redis
        self._event_bus = event_bus
        self._bars_cache: dict[str, list[PriceBar]] = {}  # symbol -> bars

    def _get_or_create_series(self, symbol: str) -> PriceSeries:
        """Lay hoac tao price series cho symbol."""
        if symbol not in self._bars_cache:
            self._bars_cache[symbol] = []
        return PriceSeries(symbol=symbol, bars=self._bars_cache[symbol])

    async def on_bar_close(self, bar: PriceBar) -> list[OutcomeResult]:
        """
        Xu ly BAR_CLOSE(M1) event - trigger outcome determination.

        Args:
            bar: Bar vua dong

        Returns:
            List of OutcomeResult cho cac predictions da duoc xac dinh
        """
        # Cap nhat local bars cache
        series = self._get_or_create_series(bar.symbol)
        series.bars.append(bar)
        series.prediction_bar_close_time = bar.close_time

        # Gioi han so bars trong cache (chi giu 300 bars ~ 5h)
        if len(series.bars) > 300:
            self._bars_cache[bar.symbol] = series.bars[-300:]

        current_time = bar.close_time
        results: list[OutcomeResult] = []

        # Lay pending predictions
        pending = await self._store.get_pending_predictions(symbol=bar.symbol, limit=100)

        for pred_row in pending:
            result = await self._determine_outcome(pred_row, series, current_time)
            if result:
                results.append(result)
                await self._update_and_publish(result)

        if results:
            logger.info(
                f"OutcomeDeterminator: {len(results)} outcomes determined for {bar.symbol}"
            )

        return results

    async def _determine_outcome(
        self,
        pred_row: dict[str, Any],
        series: PriceSeries,
        current_time: int,
    ) -> OutcomeResult | None:
        """
        Xac dinh outcome cho mot prediction.

        Priority:
        1. Zone Hold (if applicable)
        2. BSL hit (before SSL)
        3. SSL hit
        4. Timeout (240 min)
        5. Lateral (price still in range after timeout)
        """
        pred_id = pred_row.get("prediction_id", f"pred_{pred_row.get('symbol')}_{pred_row.get('bar_close_time')}")
        symbol = pred_row.get("symbol", "")
        bar_close_time = int(pred_row.get("bar_close_time", 0))
        bsl_level = float(pred_row.get("predicted_bsl_level", 0))
        ssl_level = float(pred_row.get("predicted_ssl_level", 0))

        # Loc bars ke tu prediction
        bars_since = [b for b in series.bars if b.close_time > bar_close_time]
        if not bars_since:
            return None

        high_since = max(b.high for b in bars_since)
        low_since = min(b.low for b in bars_since)
        elapsed_ms = current_time - bar_close_time
        elapsed_min = elapsed_ms / 60000.0

        # === Zone Hold Check ===
        zone_hold_result = await self._check_zone_hold(symbol, bar_close_time, current_time)
        if zone_hold_result:
            return OutcomeResult(
                prediction_id=pred_id,
                symbol=symbol,
                bar_close_time=bar_close_time,
                outcome=OutcomeType.ZONE_HOLD,
                outcome_time=current_time,
                elapsed_minutes=elapsed_min,
                high_since_prediction=high_since,
                low_since_prediction=low_since,
                zone_hold_bars=zone_hold_result["bars_held"],
                zone_touched=zone_hold_result["touched"],
            )

        # === BSL Hit Check ===
        if bsl_level > 0 and high_since >= bsl_level:
            logger.debug(
                f"Outcome: BSL_HIT for {symbol} @ {current_time} "
                f"(high={high_since:.2f} >= bsl={bsl_level:.2f})"
            )
            return OutcomeResult(
                prediction_id=pred_id,
                symbol=symbol,
                bar_close_time=bar_close_time,
                outcome=OutcomeType.BSL_HIT,
                outcome_time=current_time,
                elapsed_minutes=elapsed_min,
                high_since_prediction=high_since,
                low_since_prediction=low_since,
            )

        # === SSL Hit Check ===
        if ssl_level > 0 and low_since <= ssl_level:
            logger.debug(
                f"Outcome: SSL_HIT for {symbol} @ {current_time} "
                f"(low={low_since:.2f} <= ssl={ssl_level:.2f})"
            )
            return OutcomeResult(
                prediction_id=pred_id,
                symbol=symbol,
                bar_close_time=bar_close_time,
                outcome=OutcomeType.SSL_HIT,
                outcome_time=current_time,
                elapsed_minutes=elapsed_min,
                high_since_prediction=high_since,
                low_since_prediction=low_since,
            )

        # === Timeout Check ===
        if elapsed_ms > MAX_HORIZON_MS:
            # Da vuot horizon -> xac dinh la LATERAL hoac TIMEOUT
            logger.debug(
                f"Outcome: TIMEOUT for {symbol} @ {current_time} "
                f"(elapsed={elapsed_min:.1f}min)"
            )
            return OutcomeResult(
                prediction_id=pred_id,
                symbol=symbol,
                bar_close_time=bar_close_time,
                outcome=OutcomeType.TIMEOUT,
                outcome_time=current_time,
                elapsed_minutes=elapsed_min,
                high_since_prediction=high_since,
                low_since_prediction=low_since,
            )

        return None

    async def _check_zone_hold(
        self,
        symbol: str,
        bar_close_time: int,
        current_time: int,
    ) -> dict[str, Any] | None:
        """
        Kiem tra xem co zone nao duoc giu (touched + held 3 bars) khong.

        Tra ve dict neu zone hold, None neu khong.
        """
        try:
            zone_keys = await self._redis.get_all_zone_keys(symbol)
            if not zone_keys:
                return None

            bars_since_count = len([
                b for b in self._bars_cache.get(symbol, [])
                if b.close_time > bar_close_time
            ])

            # Chi kiem tra neu co it nhat ZONE_HOLD_BARS bars
            if bars_since_count < ZONE_HOLD_BARS:
                return None

            all_bars = self._bars_cache.get(symbol, [])
            bars_since = [b for b in all_bars if b.close_time > bar_close_time]

            for zone_key in zone_keys:
                zone_data = await self._redis.get_zone(zone_key)
                if not zone_data:
                    continue

                zone = Zone.from_dict(zone_data)
                if not zone.is_active():
                    continue

                zone_top = zone.top
                zone_bottom = zone.bottom

                # Dem so bars ma price da cham zone (trong khoang zone)
                bars_in_zone = 0
                touched = False
                for bar in bars_since:
                    if zone_bottom <= bar.close <= zone_top:
                        bars_in_zone += 1
                        touched = True

                # Zone hold = price da cham zone va giu it nhat ZONE_HOLD_BARS bars
                if touched and bars_in_zone >= ZONE_HOLD_BARS:
                    return {
                        "zone_id": zone.id,
                        "bars_held": bars_in_zone,
                        "touched": touched,
                        "zone_top": zone_top,
                        "zone_bottom": zone_bottom,
                    }

            return None
        except Exception as e:
            logger.warning(f"Zone hold check failed: {e}")
            return None

    async def _update_and_publish(self, result: OutcomeResult) -> None:
        """Update SQLite va publish event."""
        # === Update SQLite ===
        outcome_str = result.outcome.value if hasattr(result.outcome, "value") else str(result.outcome)
        await self._store.update_prediction_outcome(
            prediction_id=result.prediction_id,
            outcome=PredictionOutcome(outcome_str),
            outcome_time=result.outcome_time,
        )

        # === Publish OUTCOME_CONFIRMED event ===
        if self._event_bus:
            try:
                from core.utils.events.types import OutcomeConfirmedEvent

                event = OutcomeConfirmedEvent(
                    prediction_id=result.prediction_id,
                    symbol=result.symbol,
                    outcome=outcome_str,
                    outcome_timestamp=datetime.fromtimestamp(
                        result.outcome_time / 1000, tz=timezone.utc
                    ),
                    actual_bsl_hit=result.outcome == OutcomeType.BSL_HIT,
                    actual_ssl_hit=result.outcome == OutcomeType.SSL_HIT,
                    actual_zone_hold=result.outcome == OutcomeType.ZONE_HOLD,
                    elapsed_minutes=result.elapsed_minutes,
                    max_horizon_minutes=240.0,
                )
                self._event_bus.publish(event)
            except ImportError:
                logger.debug("EventBus not available for OUTCOME_CONFIRMED publish")

    # =============================================================================
    # Bulk Processing (for backfill)
    # =============================================================================
    async def determine_outcomes_for_symbol(
        self,
        symbol: str,
        price_bars: list[PriceBar],
    ) -> list[OutcomeResult]:
        """
        Bulk determine outcomes (dung cho backfill/retrain).
        Du lieu price_bars phai bao gom tat ca bars can thiet.
        """
        if not price_bars:
            return []

        # Load bars vao cache
        self._bars_cache[symbol] = price_bars
        series = self._get_or_create_series(symbol)

        pending = await self._store.get_pending_predictions(symbol=symbol, limit=1000)
        results: list[OutcomeResult] = []

        for pred_row in pending:
            # Tim bar close time cua prediction trong series
            pred_bar_time = int(pred_row.get("bar_close_time", 0))
            matching_bars = [b for b in price_bars if b.close_time >= pred_bar_time]

            if not matching_bars:
                continue

            series.bars = matching_bars
            series.prediction_bar_close_time = pred_bar_time

            current_time = matching_bars[-1].close_time
            result = await self._determine_outcome(pred_row, series, current_time)
            if result:
                results.append(result)
                await self._update_and_publish(result)

        return results
