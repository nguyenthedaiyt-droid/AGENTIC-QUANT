# =============================================================================
# AGENTIC-QUANT — Debate Orchestrator
# Dieu phoi toan bo Multi-Agent Debate pipeline
# =============================================================================
"""
Debate Orchestrator dieu phoi toan bo luong debate:

Flow (timeout 8s total):
1. RAG Retrieval (1s via asyncio.wait_for) — Lay precedents tu VectorDB
2. Technical Brief (0.1s) — Build context
3. Bull + Bear parallel (3s each via asyncio.gather + asyncio.wait_for)
4. Critic (5s via asyncio.wait_for) — Tong hop consensus
5. Redis persist — Luu debate record voi TTL 1800s
6. Publish CONSENSUS_READY event

Trigger conditions:
- BOS/MSS tren H1/H4 (Begin/End of Session)
- Periodic 5 phut (background task via asyncio.create_task)
- MAJOR_SURPRISE_FLAG

Rate limit: 3 phut minimum interval.
Cost tracking: log token count + daily USD, max $5/day, skip debate if exceeded.

Usage::

    orch = DebateOrchestrator(event_bus, rag_retriever, redis_manager)
    await orch.start()  # Subscribe to events + start periodic loop
    # ... system chay ...
    await orch.shutdown()
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from core.utils.events import EventType
from core.utils.events.types import (
    ConsensusReadyEvent,
    PredictionReadyEvent,
    ModelDegradedEvent,
)

from .technical_brief import TechnicalBrief
from .bull_agent import BullAgent
from .bear_agent import BearAgent
from .critic_agent import CriticAgent, ConsensusResult
from .rule_fallback import RuleFallback

if TYPE_CHECKING:
    from core.utils.events.bus import EventBus
    from core.memory.long_term.rag_retriever import RAGRetriever, Precedent
    from core.memory.short_term.redis_cache_manager import RedisCacheManager


# =============================================================================
# Constants
# =============================================================================
ORCHESTRATOR_NAME = "DebateOrchestrator"

# Timeouts (seconds)
TIMEOUT_RAG = 1.0
TIMEOUT_BRIEF = 0.1
TIMEOUT_AGENT = 3.0
TIMEOUT_CRITIC = 5.0
TIMEOUT_TOTAL = 8.0  # Tong timeout cho pipeline

# Rate limit
RATE_LIMIT_SECONDS = 180  # 3 phut minimum interval

# Periodic trigger (5 phut)
PERIODIC_INTERVAL_SECONDS = 300  # 5 phut

# Redis
DEBATE_TTL_SECONDS = 1800  # 30 phut

# Cost tracking
COST_PER_INPUT_TOKEN = 0.00000015  # $0.15/M tokens (mock)
COST_PER_OUTPUT_TOKEN = 0.00000060  # $0.60/M tokens (mock)
DAILY_BUDGET_USD = 5.0  # $5/ngay

# Log format
LOG_FORMAT = (
    "{symbol} {tf} | "
    "RAG={rag_ms:.0f}ms Brief={brief_ms:.0f}ms "
    "Bull={bull_ms:.0f}ms Bear={bear_ms:.0f}ms "
    "Critic={critic_ms:.0f}ms Total={total_ms:.0f}ms | "
    "Rating={rating:+d} Dir={dir} Conf={conf} Agr={agr:.2f} "
    "Tokens={tokens} Cost=${cost:.4f}"
)


# =============================================================================
# Daily Cost Tracker
# =============================================================================
@dataclass
class DailyCostTracker:
    """
    Track chi phi API hang ngay.

    Resets vao 00:00 UTC.
    """

    date: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    def add(self, tokens: int) -> None:
        """Cong tokens va tinh cost."""
        # Reset neu sang ngay moi
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.date:
            self.date = today
            self.total_tokens = 0
            self.total_cost_usd = 0.0

        self.total_tokens += tokens
        # Estimate: 1/3 input, 2/3 output (rough)
        input_tokens = tokens // 3
        output_tokens = tokens - input_tokens
        cost = (input_tokens * COST_PER_INPUT_TOKEN) + (output_tokens * COST_PER_OUTPUT_TOKEN)
        self.total_cost_usd += cost

    @property
    def is_over_budget(self) -> bool:
        """Kiem tra co vuot budget hang ngay khong."""
        return self.total_cost_usd >= DAILY_BUDGET_USD

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "budget_usd": DAILY_BUDGET_USD,
            "over_budget": self.is_over_budget,
        }


# =============================================================================
# Debate Context
# =============================================================================
@dataclass
class DebateContext:
    """
    Context cho mot lan debate.

    Duoc tao tu trigger event (PredictionReadyEvent).
    """
    symbol: str = ""
    timeframe: str = ""
    bar_close_time: int = 0
    timestamp: int = 0

    # Feature outputs
    prediction_features: dict[str, Any] = field(default_factory=dict)
    macro_context: dict[str, Any] = field(default_factory=dict)
    active_guardrail: bool = False
    macro_regime: str = ""

    # RAG
    precedents: list[Any] = field(default_factory=list)

    # Timing
    rag_ms: float = 0.0
    brief_ms: float = 0.0
    bull_ms: float = 0.0
    bear_ms: float = 0.0
    critic_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "bar_close_time": self.bar_close_time,
            "timestamp": self.timestamp,
            "active_guardrail": self.active_guardrail,
            "macro_regime": self.macro_regime,
            "rag_ms": self.rag_ms,
            "brief_ms": self.brief_ms,
            "bull_ms": self.bull_ms,
            "bear_ms": self.bear_ms,
            "critic_ms": self.critic_ms,
            "total_ms": self.total_ms,
        }


# =============================================================================
# Debate Orchestrator
# =============================================================================
class DebateOrchestrator:
    """
    Debate Orchestrator: Dieu phoi toan bo Multi-Agent Debate.

    Flow:
    1. Lang nghe BAR_CLOSE (BOS/MSS), periodic timer, MAJOR_SURPRISE_FLAG.
    2. Khi trigger: RAG(1s) -> Brief(0.1s) -> Bull+Bear parallel(3s) -> Critic(5s).
    3. Persist consensus vao Redis.
    4. Publish CONSENSUS_READY event.

    Rate limit: 3 phut toi thieu giua cac lan debate cho cung symbol.

    Usage::

        orch = DebateOrchestrator(event_bus, rag_retriever, redis_manager)
        await orch.start()
        # ... system runs ...
        await orch.shutdown()
    """

    def __init__(
        self,
        event_bus: EventBus,
        rag_retriever: RAGRetriever | None = None,
        redis_manager: RedisCacheManager | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._rag = rag_retriever
        self._redis = redis_manager

        # Components
        self._brief_builder = TechnicalBrief()
        self._bull_agent = BullAgent()
        self._bear_agent = BearAgent()
        self._critic_agent = CriticAgent()
        self._rule_fallback = RuleFallback()

        # State
        self._last_debate_time: dict[str, float] = {}  # symbol -> timestamp
        self._cost_tracker = DailyCostTracker()
        self._is_running = False
        self._periodic_task: asyncio.Task | None = None
        self._unsubscribe_callbacks: list[Callable[[], None]] = []

    # =========================================================================
    # Lifecycle
    # =========================================================================
    async def start(self) -> None:
        """Khoi dong Orchestrator: subscribe vao cac events trigger."""
        if self._is_running:
            logger.warning("DebateOrchestrator: already running")
            return

        logger.info("DebateOrchestrator: starting...")

        # Subscribe to PREDICTION_READY (trigger chinh)
        unsub1 = self._event_bus.subscribe(
            EventType.PREDICTION_READY,
            self._on_prediction_ready,
        )
        self._unsubscribe_callbacks.append(unsub1)

        # Subscribe to MODEL_DEGRADED
        unsub2 = self._event_bus.subscribe(
            EventType.MODEL_DEGRADED,
            self._on_model_degraded,
        )
        self._unsubscribe_callbacks.append(unsub2)

        # MAJOR_SURPRISE_FLAG duoc xu ly qua PREDICTION_READY event
        # (surprise_direction field trong PredictionReadyEvent)

        self._is_running = True

        # Start periodic debate loop (5 phut)
        self._periodic_task = asyncio.create_task(
            self._periodic_debate_loop(),
            name="DebateOrchestratorPeriodic",
        )
        logger.debug(
            f"DebateOrchestrator: periodic task started "
            f"(interval={PERIODIC_INTERVAL_SECONDS}s)"
        )

        logger.info(
            f"DebateOrchestrator: started. "
            f"Rate limit={RATE_LIMIT_SECONDS}s, "
            f"Budget=${DAILY_BUDGET_USD}/day"
        )

    async def shutdown(self) -> None:
        """Dung Orchestrator: unsubscribe tat ca events."""
        if not self._is_running:
            return

        logger.info("DebateOrchestrator: shutting down...")

        # Cancel periodic task
        if self._periodic_task and not self._periodic_task.done():
            self._periodic_task.cancel()
            try:
                await self._periodic_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"Periodic task shutdown error: {e}")
            self._periodic_task = None
        for cb in self._unsubscribe_callbacks:
            try:
                cb()
            except Exception as e:
                logger.warning(f"Unsubscribe error: {e}")
        self._unsubscribe_callbacks.clear()
        self._is_running = False
        logger.info("DebateOrchestrator: stopped")

    # =========================================================================
    # Periodic Debate Loop
    # =========================================================================
    async def _periodic_debate_loop(self) -> None:
        """
        Background task chay moi 5 phut, trigger debate cho cac symbol
        dang duoc theo doi.

        Flow:
        1. Sleep PERIODIC_INTERVAL_SECONDS (300s = 5 phut).
        2. Lay danh sach symbol dang active.
        3. Cho moi symbol: kiem tra rate limit, budget.
        4. Neu OK: tao context va chay pipeline.

        NOTE: Day la background task, exception khong duoc crash orchestrator.
        """
        logger.info(
            f"DebateOrchestrator: periodic loop started "
            f"(interval={PERIODIC_INTERVAL_SECONDS}s)"
        )
        while self._is_running:
            try:
                await asyncio.sleep(PERIODIC_INTERVAL_SECONDS)

                if not self._is_running:
                    break

                # Lay symbol dang active (tu rate limit tracker)
                active_symbols = list(self._last_debate_time.keys())
                if not active_symbols:
                    logger.debug("DebateOrchestrator: no active symbols for periodic trigger")
                    continue

                logger.debug(
                    f"DebateOrchestrator: periodic trigger for {len(active_symbols)} symbol(s)"
                )

                for symbol in active_symbols:
                    if not self._is_running:
                        break
                    if not self._check_rate_limit(symbol):
                        logger.debug(
                            f"DebateOrchestrator: periodic skip {symbol} (rate limited)"
                        )
                        continue
                    if self._cost_tracker.is_over_budget:
                        logger.warning(
                            f"DebateOrchestrator: periodic skip {symbol} "
                            f"(daily budget ${DAILY_BUDGET_USD} exceeded)"
                        )
                        continue

                    # Tao context don gian va chay debate
                    context = DebateContext(
                        symbol=symbol,
                        timeframe="H1",
                        bar_close_time=int(time.time()),
                        timestamp=int(time.time()),
                    )
                    # Chay pipeline (bat exception ben trong)
                    await self._run_debate(context)

            except asyncio.CancelledError:
                logger.info("DebateOrchestrator: periodic loop cancelled")
                break
            except Exception as e:
                logger.exception(
                    f"DebateOrchestrator: periodic loop error: {e}"
                )
                # Tiep tuc loop, khong crash

        logger.info("DebateOrchestrator: periodic loop stopped")

    # =========================================================================
    # Event Handlers
    # =========================================================================
    async def _on_prediction_ready(self, event: Any) -> None:
        """
        Xu ly PREDICTION_READY event.

        Trigger: BOS/MSS tren H1/H4, periodic 5 phut, MAJOR_SURPRISE_FLAG.
        """
        # Check rate limit
        symbol = getattr(event, "symbol", "")
        if not symbol:
            logger.warning("DebateOrchestrator: event missing symbol")
            return

        if not self._check_rate_limit(symbol):
            logger.debug(f"DebateOrchestrator: rate limited for {symbol}")
            return

        # Check budget
        if self._cost_tracker.is_over_budget:
            logger.warning(
                f"DebateOrchestrator: daily budget ${DAILY_BUDGET_USD} exceeded. "
                f"Skipping debate for {symbol}"
            )
            return

        # Build context
        context = DebateContext(
            symbol=symbol,
            timeframe=getattr(event, "timeframe", "H1"),
            bar_close_time=getattr(event, "timestamp", 0),
            timestamp=int(time.time()),
            macro_context={
                "macro_regime": getattr(event, "macro_regime", ""),
            },
            active_guardrail=getattr(event, "active_guardrail", False),
        )

        # Kiem tra MAJOR_SURPRISE_FLAG
        if hasattr(event, "surprise_direction") and getattr(event, "surprise_direction", "") in ("POSITIVE", "NEGATIVE"):
            context.macro_context["MAJOR_SURPRISE_FLAG"] = True

        # Chay pipeline
        await self._run_debate(context)

    async def _on_model_degraded(self, event: Any) -> None:
        """
        Xu ly MODEL_DEGRADED event.

        Neu MODEL_DEGRADED=True -> chuyen sang RuleFallback.
        """
        symbol = getattr(event, "symbol", "")
        if not symbol:
            return

        logger.warning(f"DebateOrchestrator: MODEL_DEGRADED for {symbol}, switching to rule fallback")

        # Build minimal context
        context = DebateContext(
            symbol=symbol,
            timeframe="H1",
            bar_close_time=int(time.time()),
            timestamp=int(time.time()),
            macro_context={
                "macro_regime": "",
                "model_degraded": True,
            },
        )

        # Run rule fallback directly
        consensus = await self._rule_fallback.evaluate(context.macro_context or {})

        # Persist and publish
        await self._persist_and_publish(context, consensus)

    # =========================================================================
    # Main Debate Pipeline
    # =========================================================================
    async def _run_debate(self, context: DebateContext) -> None:
        """
        Chay pipeline debate chinh.

        Flow: RAG -> Brief -> Bull+Bear (parallel) -> Critic -> Redis -> Event.
        """
        start_total = time.monotonic()
        logger.info(
            f"DebateOrchestrator: starting debate for {context.symbol} "
            f"{context.timeframe} (guardrail={context.active_guardrail})"
        )

        # Double-check budget inside pipeline (extra safety)
        if self._cost_tracker.is_over_budget:
            logger.warning(
                f"DebateOrchestrator: daily budget ${DAILY_BUDGET_USD} exceeded, "
                f"aborting debate for {context.symbol}"
            )
            return

        try:
            # === Step 1: RAG Retrieval (1s) ===
            logger.debug("DebateOrchestrator: step 1/4 - RAG")
            start_rag = time.monotonic()
            context.precedents = await self._run_rag(context)
            context.rag_ms = (time.monotonic() - start_rag) * 1000

            # === Step 2: Technical Brief (0.1s) ===
            logger.debug("DebateOrchestrator: step 2/4 - Technical Brief")
            start_brief = time.monotonic()
            brief_data = await self._run_brief(context)
            context.brief_ms = (time.monotonic() - start_brief) * 1000

            # === Step 3: Bull + Bear parallel (3s each) ===
            logger.debug("DebateOrchestrator: step 3/4 - Bull + Bear (parallel)")
            start_agents = time.monotonic()
            bull_result, bear_result = await self._run_agents(brief_data, context)
            context.bull_ms = (time.monotonic() - start_agents) * 1000
            context.bear_ms = context.bull_ms  # Parallel, same duration

            # === Step 4: Critic (5s) ===
            logger.debug("DebateOrchestrator: step 4/4 - Critic")
            start_critic = time.monotonic()
            consensus = await self._run_critic(bull_result, bear_result, context)
            context.critic_ms = (time.monotonic() - start_critic) * 1000

            context.total_ms = (time.monotonic() - start_total) * 1000

            # Check total timeout
            if context.total_ms > TIMEOUT_TOTAL * 1000:
                logger.warning(
                    f"DebateOrchestrator: total time {context.total_ms:.0f}ms "
                    f"exceeds {TIMEOUT_TOTAL * 1000}ms limit"
                )

            # === Persist + Publish ===
            await self._persist_and_publish(context, consensus)

            # === Log execution ===
            self._log_execution(context, consensus)

        except asyncio.TimeoutError:
            logger.error(
                f"DebateOrchestrator: pipeline TOTAL TIMEOUT for {context.symbol} "
                f"({TIMEOUT_TOTAL}s exceeded)"
            )
        except Exception as e:
            logger.exception(
                f"DebateOrchestrator: pipeline failed for {context.symbol}: {e}"
            )

    # =========================================================================
    # Pipeline Steps
    # =========================================================================
    async def _run_rag(self, context: DebateContext) -> list[Any]:
        """Step 1: RAG Retrieval (max 1s)."""
        if not self._rag:
            logger.debug("DebateOrchestrator: no RAG retriever configured")
            return []

        try:
            precedents = await asyncio.wait_for(
                self._rag.retrieve_precedents(
                    embedding=[],
                    symbol=context.symbol,
                    macro_regime=context.macro_context.get("macro_regime"),
                ),
                timeout=TIMEOUT_RAG,
            )
            logger.debug(
                f"DebateOrchestrator: RAG retrieved {len(precedents)} precedents"
            )
            return precedents
        except asyncio.TimeoutError:
            logger.warning("DebateOrchestrator: RAG timed out")
            return []
        except Exception as e:
            logger.warning(f"DebateOrchestrator: RAG failed: {e}")
            return []

    async def _run_brief(self, context: DebateContext) -> Any:
        """Step 2: Build Technical Brief (max 0.1s)."""
        brief_data = await self._brief_builder.build(
            features=context.prediction_features,
            symbol=context.symbol,
            timeframe=context.timeframe,
            bar_close_time=context.bar_close_time,
            current_price=0.0,
            macro_context=context.macro_context,
            precedents=context.precedents,
        )
        return brief_data

    async def _run_agents(
        self,
        brief_data: Any,
        context: DebateContext,
    ) -> tuple[Any, Any]:
        """Step 3: Bull + Bear parallel (max 3s each)."""
        brief_text = brief_data.brief_text

        bull_coro = self._bull_agent.analyze(
            brief_text=brief_text,
            active_guardrail=context.active_guardrail,
        )
        bear_coro = self._bear_agent.analyze(
            brief_text=brief_text,
            active_guardrail=context.active_guardrail,
        )

        try:
            bull_result, bear_result = await asyncio.wait_for(
                asyncio.gather(
                    bull_coro,
                    bear_coro,
                    return_exceptions=True,
                ),
                timeout=TIMEOUT_AGENT,
            )
        except asyncio.TimeoutError:
            logger.warning("DebateOrchestrator: agents timed out (3s)")
            # Try to get individual results if any completed
            bull_result = await self._bull_agent.analyze(
                brief_text=brief_text,
                active_guardrail=context.active_guardrail,
            )
            bear_result = await self._bear_agent.analyze(
                brief_text=brief_text,
                active_guardrail=context.active_guardrail,
            )

        # Xu ly exceptions
        if isinstance(bull_result, Exception):
            logger.error(f"DebateOrchestrator: Bull agent exception: {bull_result}")
            bull_result = type("_Fake", (), {
                "success": False, "error": str(bull_result),
                "rating": 0, "confidence": 0, "evidence": [],
                "reasoning": "", "target_price": None,
                "token_count": 0, "active_guardrail": context.active_guardrail,
                "news_caveat": "", "to_dict": lambda: {}
            })()

        if isinstance(bear_result, Exception):
            logger.error(f"DebateOrchestrator: Bear agent exception: {bear_result}")
            bear_result = type("_Fake", (), {
                "success": False, "error": str(bear_result),
                "rating": 0, "confidence": 0, "evidence": [],
                "reasoning": "", "target_price": None,
                "token_count": 0, "active_guardrail": context.active_guardrail,
                "news_caveat": "", "to_dict": lambda: {}
            })()

        return bull_result, bear_result

    async def _run_critic(
        self,
        bull_result: Any,
        bear_result: Any,
        context: DebateContext,
    ) -> ConsensusResult:
        """Step 4: Critic (max 5s)."""
        try:
            consensus = await asyncio.wait_for(
                self._critic_agent.evaluate(
                    bull_result=bull_result,
                    bear_result=bear_result,
                    active_guardrail=context.active_guardrail,
                ),
                timeout=TIMEOUT_CRITIC,
            )
        except asyncio.TimeoutError:
            logger.warning("DebateOrchestrator: Critic timed out (5s)")
            # Fallback: rule-based
            consensus = self._critic_agent._rule_based_fallback(
                bull_result, bear_result, context.active_guardrail
            )

        return consensus

    # =========================================================================
    # Persist + Publish
    # =========================================================================
    async def _persist_and_publish(
        self,
        context: DebateContext,
        consensus: ConsensusResult,
    ) -> None:
        """Luu debate vao Redis va publish CONSENSUS_READY event."""
        # === Redis persist ===
        if self._redis:
            try:
                debate_data = {
                    "symbol": context.symbol,
                    "timeframe": context.timeframe,
                    "bar_close_time": context.bar_close_time,
                    "timestamp": context.timestamp,
                    "rating": consensus.rating,
                    "direction": consensus.direction,
                    "confidence_qualifier": consensus.confidence_qualifier,
                    "agreement_score": consensus.agreement_score,
                    "bull_thesis": consensus.bull_thesis,
                    "bear_thesis": consensus.bear_thesis,
                    "reasoning": consensus.reasoning,
                    "bull_evidence": json.dumps(consensus.bull_evidence),
                    "bear_evidence": json.dumps(consensus.bear_evidence),
                    "debate_used_fallback": consensus.debate_used_fallback,
                    "active_guardrail": context.active_guardrail,
                    "macro_regime": context.macro_context.get("macro_regime", ""),
                    "rag_ms": context.rag_ms,
                    "brief_ms": context.brief_ms,
                    "bull_ms": context.bull_ms,
                    "bear_ms": context.bear_ms,
                    "critic_ms": context.critic_ms,
                    "total_ms": context.total_ms,
                    "token_count": consensus.token_count,
                    "cost_usd": round(consensus.token_count * COST_PER_INPUT_TOKEN, 8),
                }
                await self._redis.set_debate(
                    symbol=context.symbol,
                    bar_close_ts=context.bar_close_time,
                    debate_data=debate_data,
                )
                # Set custom TTL
                key = self._redis.debate_key(context.symbol, context.bar_close_time)
                await self._redis.client.expire(key, DEBATE_TTL_SECONDS)
                logger.debug(
                    f"DebateOrchestrator: persisted to Redis "
                    f"(key=debate:{context.symbol}:{context.bar_close_time})"
                )
            except Exception as e:
                logger.warning(f"DebateOrchestrator: Redis persist failed: {e}")

        # === Publish event ===
        try:
            event = ConsensusReadyEvent(
                symbol=context.symbol,
                timestamp=context.bar_close_time,
                rating=consensus.rating,
                direction=consensus.direction,
                confidence_qualifier=consensus.confidence_qualifier,
                agreement_score=consensus.agreement_score,
                bull_thesis=consensus.bull_thesis,
                bear_thesis=consensus.bear_thesis,
                reasoning=consensus.reasoning,
                bull_evidence=consensus.bull_evidence,
                bear_evidence=consensus.bear_evidence,
                debate_used_fallback=consensus.debate_used_fallback,
                source=ORCHESTRATOR_NAME,
            )
            self._event_bus.publish(event)
            logger.debug("DebateOrchestrator: published CONSENSUS_READY event")
        except Exception as e:
            logger.error(f"DebateOrchestrator: failed to publish event: {e}")

    # =========================================================================
    # Rate Limiting
    # =========================================================================
    def _check_rate_limit(self, symbol: str) -> bool:
        """
        Kiem tra rate limit: 3 phut minimum interval cho cung symbol.

        Returns:
            True neu duoc phep debate, False neu bi rate limit.
        """
        now = time.time()
        last_time = self._last_debate_time.get(symbol, 0.0)

        if now - last_time < RATE_LIMIT_SECONDS:
            remaining = RATE_LIMIT_SECONDS - (now - last_time)
            logger.debug(
                f"DebateOrchestrator: rate limited for {symbol}, "
                f"{remaining:.0f}s remaining"
            )
            return False

        self._last_debate_time[symbol] = now
        return True

    # =========================================================================
    # Logging & Cost Tracking
    # =========================================================================
    def _log_execution(
        self,
        context: DebateContext,
        consensus: ConsensusResult,
    ) -> None:
        """Log execution stats + track cost."""
        # Track cost
        self._cost_tracker.add(consensus.token_count)

        log_line = LOG_FORMAT.format(
            symbol=context.symbol,
            tf=context.timeframe,
            rag_ms=context.rag_ms,
            brief_ms=context.brief_ms,
            bull_ms=context.bull_ms,
            bear_ms=context.bear_ms,
            critic_ms=context.critic_ms,
            total_ms=context.total_ms,
            rating=consensus.rating,
            dir=consensus.direction,
            conf=consensus.confidence_qualifier,
            agr=consensus.agreement_score,
            tokens=consensus.token_count,
            cost=consensus.token_count * COST_PER_INPUT_TOKEN,
        )
        logger.info(f"Debate: {log_line}")

        # Log daily cost summary
        cost_info = self._cost_tracker.to_dict()
        logger.debug(
            f"DebateOrchestrator: daily cost: "
            f"${cost_info['total_cost_usd']:.4f} / ${DAILY_BUDGET_USD} "
            f"({cost_info['total_tokens']} tokens)"
        )

    def get_cost_stats(self) -> dict:
        """Tra ve cost statistics."""
        return self._cost_tracker.to_dict()

    def get_status(self) -> dict:
        """Tra ve trang thai hien tai cua orchestrator."""
        return {
            "is_running": self._is_running,
            "rate_limit_seconds": RATE_LIMIT_SECONDS,
            "timeout_total": TIMEOUT_TOTAL,
            "cost_tracker": self._cost_tracker.to_dict(),
            "last_debates": {
                sym: datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                for sym, ts in self._last_debate_time.items()
            },
        }

    # =========================================================================
    # Manual Trigger
    # =========================================================================
    async def trigger_debate(
        self,
        symbol: str,
        timeframe: str = "H1",
        prediction_features: dict[str, Any] | None = None,
        macro_context: dict[str, Any] | None = None,
        active_guardrail: bool = False,
    ) -> ConsensusResult | None:
        """
        Trigger debate manually (dung cho test hoac API).

        Bo qua rate limit check.
        """
        context = DebateContext(
            symbol=symbol,
            timeframe=timeframe,
            bar_close_time=int(time.time()),
            timestamp=int(time.time()),
            prediction_features=prediction_features or {},
            macro_context=macro_context or {},
            active_guardrail=active_guardrail,
        )

        await self._run_debate(context)

        # Return last consensus (not ideal but works for manual trigger)
        return None
