# =============================================================================
# AGENTIC-QUANT — RAG Retriever
# Retrieve similar precedents from VectorDB for Multi-Agent Debate
# =============================================================================

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

from core.memory.long_term.vectordb_adapter import BaseVectorDB, DebateHit
from core.memory.short_term.redis_cache_manager import RedisCacheManager

if TYPE_CHECKING:
    pass


# =============================================================================
# Constants
# =============================================================================
RAG_CACHE_TTL_SECONDS = 300  # 5 minutes
RAG_CACHE_BUCKET_SECONDS = 300  # timestamp // 300 -> 5-min bucket
RAG_INITIAL_THRESHOLD = 0.80  # Cosine similarity threshold
RAG_FALLBACK_THRESHOLD = 0.75  # Lowered threshold if < 3 results
RAG_DEFAULT_K = 3  # So luong precedents tra ve


# =============================================================================
# Precedent Data Class
# =============================================================================
@dataclass
class Precedent:
    """
    Mot precedent tra ve tu RAG retrieval.
    Chua debate record + metadata tu RAG.
    """

    symbol: str
    bar_close_time: int
    macro_regime: str
    session: str

    # Consensus info
    rating: int  # -4 to +4
    direction: str  # BULLISH | BEARISH | NEUTRAL
    agreement_score: float  # [0, 1]

    # Market context
    outcome: str  # Ket qua thuc te
    precedents_count: int

    # Scores
    cosine_sim: float
    recency_days: int
    re_rank_score: float

    # Full debate data
    bull_summary: str = ""
    bear_summary: str = ""

    @property
    def is_bullish(self) -> bool:
        return self.direction == "BULLISH"

    @property
    def is_bearish(self) -> bool:
        return self.direction == "BEARISH"

    def age_days(self) -> int:
        """So ngay tu khi debate xay ra."""
        return self.recency_days

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "bar_close_time": self.bar_close_time,
            "macro_regime": self.macro_regime,
            "session": self.session,
            "rating": self.rating,
            "direction": self.direction,
            "agreement_score": self.agreement_score,
            "outcome": self.outcome,
            "precedents_count": self.precedents_count,
            "cosine_sim": self.cosine_sim,
            "recency_days": self.recency_days,
            "re_rank_score": self.re_rank_score,
            "bull_summary": self.bull_summary,
            "bear_summary": self.bear_summary,
        }


# =============================================================================
# RAG Retriever
# =============================================================================
class RAGRetriever:
    """
    Retrieve similar precedents tu VectorDB cho Multi-Agent Debate.

    retrieve_precedents() flow:
    1. Pre-filter by payload: symbol matches AND actual_outcome IS NOT NULL
    2. ANN search with cosine_sim >= 0.80
    3. If < 3 results: lower threshold to 0.75
    4. Re-rank: 0.7 * cosine_sim + 0.3 * recency_weight
    5. Return top-3

    Cache in Redis: rag:{symbol}:{timestamp_rounded_to_5min}
      - TTL: 300s (5 min)
      - On cache hit: return cached results immediately

    Integration:
    - Phase 3 (VectorDB): Query debate_archive collection
    - Phase 3 (Redis): Cache results 5 min
    - Phase 5 (Multi-Agent Debate): Su dung precedents de enhance debate
    """

    def __init__(
        self,
        vectordb: BaseVectorDB,
        redis: RedisCacheManager,
        cache_ttl_seconds: int = RAG_CACHE_TTL_SECONDS,
    ) -> None:
        self._vectordb = vectordb
        self._redis = redis
        self._cache_ttl = cache_ttl_seconds

    # =========================================================================
    # Main Retrieval
    # =========================================================================
    async def retrieve_precedents(
        self,
        embedding: list[float],
        symbol: str,
        macro_regime: str | None = None,
        k: int = RAG_DEFAULT_K,
    ) -> list[Precedent]:
        """
        Lay k precedents tuong tu nhat tu VectorDB.

        Args:
            embedding: e_USV vector cua debate hien tai
            symbol: Symbol can tim
            macro_regime: Loc theo macro regime (None = tat ca)
            k: So luong precedents tra ve

        Returns:
            List of Precedent objects, sorted by re_rank_score descending
        """
        # === Check Redis cache ===
        cache_key = self._build_cache_key(symbol, macro_regime)
        cached = await self._get_from_cache(cache_key)
        if cached:
            logger.debug(f"RAGRetriever: cache HIT for {cache_key}")
            return cached

        # === ANN search voi threshold 0.80 ===
        hits = await self._vectordb.search_similar_debates(
            embedding=embedding,
            symbol=symbol,
            macro_regime=macro_regime,
            k=k,
        )

        # === Threshold fallback: 0.80 -> 0.75 ===
        valid_hits = [h for h in hits if h.cosine_sim >= RAG_INITIAL_THRESHOLD]

        if len(valid_hits) < 3 and len(hits) >= 3:
            logger.debug(
                f"RAGRetriever: threshold fallback {RAG_INITIAL_THRESHOLD} -> {RAG_FALLBACK_THRESHOLD}"
            )
            valid_hits = [h for h in hits if h.cosine_sim >= RAG_FALLBACK_THRESHOLD]

        # === Convert to Precedents ===
        precedents = [self._hit_to_precedent(h) for h in valid_hits[:k]]

        # === Re-rank (done in VectorDB adapter, but re-verify) ===
        precedents.sort(key=lambda p: p.re_rank_score, reverse=True)

        # === Filter: chi tra ve precedents co outcome da xac dinh ===
        precedents = [p for p in precedents if p.outcome]

        # === Store in Redis cache ===
        await self._store_in_cache(cache_key, precedents)

        logger.debug(
            f"RAGRetriever: retrieved {len(precedents)} precedents for {symbol} "
            f"(regime={macro_regime or 'all'}, top_score={precedents[0].re_rank_score if precedents else 0:.3f})"
        )

        return precedents[:k]

    # =========================================================================
    # Cache Operations
    # =========================================================================
    def _build_cache_key(
        self,
        symbol: str,
        macro_regime: str | None,
    ) -> str:
        """
        Build Redis cache key.
        Format: rag:{symbol}:{timestamp // 300}
        Dung timestamp bucket de tranh over-caching nhung van co hieu qua.
        """
        now_bucket = int(datetime.now(timezone.utc).timestamp()) // RAG_CACHE_BUCKET_SECONDS
        regime_part = macro_regime or "ALL"
        return f"rag:{symbol}:{regime_part}:{now_bucket}"

    async def _get_from_cache(self, cache_key: str) -> list[Precedent] | None:
        """Doc tu Redis cache."""
        try:
            cached_json = await self._redis.client.get(cache_key)
            if not cached_json:
                return None

            import json
            data = json.loads(cached_json.decode() if isinstance(cached_json, bytes) else cached_json)
            return [Precedent(**p) for p in data]
        except Exception as e:
            logger.warning(f"RAGRetriever: cache read failed for {cache_key}: {e}")
            return None

    async def _store_in_cache(
        self,
        cache_key: str,
        precedents: list[Precedent],
    ) -> None:
        """Luu vao Redis cache voi TTL 5 min."""
        try:
            import json
            data = json.dumps([p.to_dict() for p in precedents])
            await self._redis.client.set(cache_key, data.encode(), ex=self._cache_ttl)
        except Exception as e:
            logger.warning(f"RAGRetriever: cache write failed for {cache_key}: {e}")

    # =========================================================================
    # Cache Invalidation
    # =========================================================================
    async def invalidate_cache(self, symbol: str) -> None:
        """
        Xoa cache cho symbol (dung khi co du lieu moi nhat).
        """
        try:
            pattern = f"rag:{symbol}:*"
            deleted = 0
            async for key in self._redis.client.scan_iter(match=pattern, count=100):
                await self._redis.client.delete(key)
                deleted += 1
            if deleted > 0:
                logger.debug(f"RAGRetriever: invalidated {deleted} cache entries for {symbol}")
        except Exception as e:
            logger.warning(f"RAGRetriever: cache invalidation failed for {symbol}: {e}")

    # =========================================================================
    # Internal Helpers
    # =========================================================================
    def _hit_to_precedent(self, hit: DebateHit) -> Precedent:
        """Chuyen DebateHit thanh Precedent."""
        debate = hit.debate_record

        # Extract summaries
        bull_summary = ""
        bear_summary = ""
        if debate.bull:
            parts = []
            if debate.bull.target_price:
                parts.append(f"Target: {debate.bull.target_price:.2f}")
            if debate.bull.confidence:
                parts.append(f"Confidence: {debate.bull.confidence:.2f}")
            bull_summary = " | ".join(parts)

        if debate.bear:
            parts = []
            if debate.bear.target_price:
                parts.append(f"Target: {debate.bear.target_price:.2f}")
            if debate.bear.confidence:
                parts.append(f"Confidence: {debate.bear.confidence:.2f}")
            bear_summary = " | ".join(parts)

        consensus = debate.consensus

        return Precedent(
            symbol=debate.symbol,
            bar_close_time=debate.bar_close_time,
            macro_regime=debate.macro_regime,
            session=debate.session,

            rating=consensus.rating if consensus else 0,
            direction=consensus.preferred_direction if consensus else "NEUTRAL",
            agreement_score=consensus.agreement_score if consensus else 0.0,

            outcome=getattr(debate, "outcome", "") or "",
            precedents_count=debate.precedents_count,

            cosine_sim=hit.cosine_sim,
            recency_days=hit.days_since,
            re_rank_score=hit.re_rank_score,

            bull_summary=bull_summary,
            bear_summary=bear_summary,
        )

    # =========================================================================
    # Statistics
    # =========================================================================
    async def get_cache_stats(self, symbol: str | None = None) -> dict[str, Any]:
        """Tra ve cache statistics."""
        try:
            pattern = f"rag:{symbol or '*'}:*"
            keys: list[str] = []
            async for key in self._redis.client.scan_iter(match=pattern, count=1000):
                keys.append(key.decode() if isinstance(key, bytes) else key)
            return {
                "cache_keys": len(keys),
                "symbol": symbol,
            }
        except Exception as e:
            return {"error": str(e)}
