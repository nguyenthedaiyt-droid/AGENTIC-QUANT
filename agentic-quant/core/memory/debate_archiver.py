# =============================================================================
# AGENTIC-QUANT — Debate Archival Worker
# Background asyncio task: archive debates from Redis to VectorDB every 5 minutes
# =============================================================================

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from core.memory.long_term.sqlite_history_store import SQLiteHistoryStore
from core.memory.long_term.vectordb_adapter import BaseVectorDB, DebateHit
from core.memory.short_term.redis_cache_manager import RedisCacheManager

if TYPE_CHECKING:
    pass


# =============================================================================
# Constants
# =============================================================================
ARCHIVE_CHECK_INTERVAL_SECONDS = 300  # 5 minutes
DEBATE_ARCHIVE_TTL_THRESHOLD_SECONDS = 600  # Archive when TTL < 600s (10 min)
MAX_ARCHIVE_BATCH_SIZE = 50
RETRY_MAX_ATTEMPTS = 3


# =============================================================================
# e_USV Projection Helper
# =============================================================================
def compute_e_usv(debate_data: dict) -> list[float]:
    """
    Tinh e_USV projection tu debate features.

    e_USV la projection vector duoc su dung de:
    1. Search similar debates in VectorDB (RAG)
    2. Insert into debate_archive collection

    Projection duoc tinh tu:
    - p_bsl, p_ssl, p_lateral (from Model A)
    - consensus_rating, agreement_score
    - regime, session encoding
    - precedent_count

    Luu y: Day la simplified projection. Trong production,
    LSTM model se tao ra e_USV vector that su.
    """
    import hashlib
    import struct

    # Simple projection: hash-based deterministic vector
    # Trong production, thay bang actual LSTM projection
    hash_input = (
        f"{debate_data.get('symbol', '')}"
        f"{debate_data.get('bar_close_time', 0)}"
        f"{debate_data.get('macro_regime', 'NORMAL')}"
    )
    hash_bytes = hashlib.sha256(hash_input.encode()).digest()

    # Tao 256-dim vector tu hash
    vector: list[float] = []
    for i in range(256):
        idx = i % len(hash_bytes)
        val = struct.unpack_from("!f", hash_bytes, (idx * 4) % 28)[0] if i < 64 else 0.0
        # Normalize ve [-1, 1]
        if i < 64:
            vector.append(val)
        else:
            # Pad with regime/session encoding
            regime_val = hash(hash_input) % 1000 / 500.0 - 1.0
            vector.append(regime_val if i < 128 else 0.0)

    # Normalize vector
    import math
    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 0:
        vector = [v / norm for v in vector]

    return vector[:256]


# =============================================================================
# Archive Result
# =============================================================================
@dataclass
class ArchiveResult:
    """Ket qua archive mot debate."""

    debate_key: str
    symbol: str
    bar_close_time: int
    success: bool
    vectordb_id: str | None = None
    error: str | None = None
    used_fallback: bool = False


# =============================================================================
# Debate Archiver
# =============================================================================
class DebateArchiver:
    """
    Background worker: archive debates from Redis to VectorDB.

    Chay moi 5 phut:
    1. Scan Redis for debate keys where archived = False AND TTL < 600s
    2. For each qualifying debate:
       - Compute e_USV projection from debate features
       - Insert into VectorDB debate_archive collection
       - Set archived = True in Redis
    3. Retry logic: if VectorDB unavailable -> queue in SQLite pending_archive

    Depends on:
    - 3.1 (Redis): Read debate keys
    - 3.5 (VectorDB): Insert embeddings
    - 3.3 (SQLite): Fallback pending_archive queue
    """

    def __init__(
        self,
        redis: RedisCacheManager,
        vectordb: BaseVectorDB,
        store: SQLiteHistoryStore,
        check_interval_seconds: int = ARCHIVE_CHECK_INTERVAL_SECONDS,
    ) -> None:
        self._redis = redis
        self._vectordb = vectordb
        self._store = store
        self._interval = check_interval_seconds
        self._running = False
        self._task: asyncio.Task[None] | None = None

    # =========================================================================
    # Lifecycle
    # =========================================================================
    async def start(self) -> None:
        """Bat dau archival worker (background task)."""
        if self._running:
            logger.warning("DebateArchiver already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"DebateArchiver started (interval={self._interval}s)")

    async def stop(self) -> None:
        """Dung archival worker."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("DebateArchiver stopped")

    async def _run_loop(self) -> None:
        """Main loop: chay archive moi _interval giay."""
        while self._running:
            try:
                await asyncio.sleep(self._interval)
                if self._running:
                    await self.archive_ready_debates()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"DebateArchiver loop error: {e}")

    # =========================================================================
    # Archive Logic
    # =========================================================================
    async def archive_ready_debates(self, symbol: str | None = None) -> list[ArchiveResult]:
        """
        Tim va archive tat ca debates san sang.

        Args:
            symbol: Neu duoc cung cap, chi archive debates cho symbol nay

        Returns:
            List of ArchiveResult cho tat ca debates da xu ly
        """
        from core.memory.models import DebateRecord

        results: list[ArchiveResult] = []

        try:
            pending = await self._redis.get_pending_debates_for_archive(symbol=symbol)
            logger.debug(f"DebateArchiver: found {len(pending)} debates to check")

            for sym, bar_ts, debate_data in pending[:MAX_ARCHIVE_BATCH_SIZE]:
                result = await self._archive_single(sym, bar_ts, debate_data)
                results.append(result)

            archived_count = sum(1 for r in results if r.success)
            logger.info(
                f"DebateArchiver: archived {archived_count}/{len(results)} debates"
            )
        except Exception as e:
            logger.error(f"DebateArchiver: failed to check debates: {e}")

        return results

    async def _archive_single(
        self,
        symbol: str,
        bar_close_time: int,
        debate_data: dict,
    ) -> ArchiveResult:
        """
        Archive mot debate don.
        Neu VectorDB fail, fallback sang SQLite pending_archive.
        """
        debate_key = self._redis.debate_key(symbol, bar_close_time)

        try:
            # === Compute e_USV projection ===
            embedding = compute_e_usv(debate_data)

            # === Build DebateRecord ===
            debate_record = DebateRecord.from_dict(debate_data)
            debate_record.embedding = embedding

            # === Insert vao VectorDB ===
            vectordb_id = await self._vectordb.insert_debate(debate_record, embedding)

            # === Mark as archived in Redis ===
            await self._redis.mark_debate_archived(symbol, bar_close_time)

            logger.debug(f"DebateArchiver: archived {debate_key} -> {vectordb_id}")
            return ArchiveResult(
                debate_key=debate_key,
                symbol=symbol,
                bar_close_time=bar_close_time,
                success=True,
                vectordb_id=vectordb_id,
            )

        except Exception as e:
            logger.warning(f"DebateArchiver: VectorDB insert failed for {debate_key}: {e}")

            # === Fallback: queue in SQLite ===
            try:
                await self._store.insert_pending_archive(
                    content_type="debate",
                    content_id=debate_key,
                    payload=debate_data,
                )
                logger.debug(f"DebateArchiver: queued {debate_key} in pending_archive")
                return ArchiveResult(
                    debate_key=debate_key,
                    symbol=symbol,
                    bar_close_time=bar_close_time,
                    success=True,  # Queued = success (will retry)
                    error=str(e),
                    used_fallback=True,
                )
            except Exception as sqe:
                logger.error(f"DebateArchiver: SQLite fallback also failed: {sqe}")
                return ArchiveResult(
                    debate_key=debate_key,
                    symbol=symbol,
                    bar_close_time=bar_close_time,
                    success=False,
                    error=str(e),
                    used_fallback=False,
                )

    # =========================================================================
    # Retry Pending (from SQLite fallback)
    # =========================================================================
    async def retry_pending_archive(self) -> int:
        """
        Retry pending archives tu SQLite fallback queue.
        Tra ve so luong thanh cong.
        """
        items = await self._store.get_pending_archive_items(content_type="debate", limit=50)
        success_count = 0

        for item in items:
            record_id = item.get("record_id")
            content_id = item.get("content_id")
            payload_str = item.get("payload_json", "{}")

            import json
            try:
                payload = json.loads(payload_str)
                # Parse symbol and bar_close_time tu content_id
                # content_id format: debate:{symbol}:{bar_close_ts}
                parts = content_id.split(":")
                if len(parts) >= 3:
                    sym = parts[1]
                    bar_ts = int(parts[2])
                    result = await self._archive_single(sym, bar_ts, payload)
                    if result.success and not result.used_fallback:
                        await self._store.mark_archive_completed(record_id)
                        success_count += 1
                    elif result.success and result.used_fallback:
                        await self._store.mark_archive_completed(record_id)
                        success_count += 1
                else:
                    await self._store.mark_archive_failed(record_id, f"Invalid content_id: {content_id}")
            except Exception as e:
                await self._store.mark_archive_failed(record_id, str(e))

        logger.info(f"DebateArchiver: retry completed, {success_count}/{len(items)} succeeded")
        return success_count
