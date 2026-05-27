# =============================================================================
# AGENTIC-QUANT — SQLite History Store
# Async CRUD cho tat ca bang SQLite, async write queue, version-based migrations
# =============================================================================

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite
from loguru import logger

from core.memory.models import (
    ModelAPrediction,
    PredictionOutcome,
    Zone,
)
from core.memory.models.enums import ZoneStatus

if TYPE_CHECKING:
    pass


# =============================================================================
# Default Paths & Constants
# =============================================================================
DEFAULT_DB_PATH = "data/memory.db"
MIGRATIONS_DIR = "scripts/migrations"
CURRENT_SCHEMA_VERSION = 2
WRITE_QUEUE_MAXSIZE = 10000
WRITE_BATCH_INTERVAL_MS = 100


# =============================================================================
# Write Queue Item
# =============================================================================
@dataclass
class _WriteJob:
    """Mot write operation trong queue."""

    sql: str
    params: tuple[Any, ...]
    retry_count: int = 0

    @property
    def max_retries(self) -> int:
        return 3


# =============================================================================
# SQLite History Store
# =============================================================================
class SQLiteHistoryStore:
    """
    Async SQLite CRUD cho Phase 3 Memory Engine.

    Features:
    - Async write queue: buffer writes, flush batch every 100ms
    - Version-based migrations (PRAGMA user_version)
    - WAL mode, optimized PRAGMA settings
    - All 5 core tables: predictions, zone_history, model_performance,
      system_metrics, pending_archive

    Usage::

        store = SQLiteHistoryStore()
        await store.connect()
        await store.insert_prediction(pred)
        await store.close()
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        write_queue_maxsize: int = WRITE_QUEUE_MAXSIZE,
        write_batch_interval_ms: int = WRITE_BATCH_INTERVAL_MS,
    ) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: aiosqlite.Connection | None = None

        self._write_queue: asyncio.Queue[_WriteJob] = asyncio.Queue(
            maxsize=write_queue_maxsize
        )
        self._write_batch_interval = write_batch_interval_ms / 1000.0
        self._flush_task: asyncio.Task[None] | None = None
        self._running = False

    # =========================================================================
    # Lifecycle
    # =========================================================================
    async def connect(self) -> None:
        """Ket noi, chay migrations, khoi dong flush task."""
        if self._conn:
            return

        self._conn = await aiosqlite.connect(
            str(self._db_path),
            isolation_level=None,
        )
        self._conn.row_factory = aiosqlite.Row

        await self._configure()
        await self._run_migrations()

        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info(f"SQLite connected: {self._db_path}")

    async def close(self) -> None:
        """Dong ket noi, dung flush task."""
        self._running = False

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        # Flush remaining jobs
        await self.flush_all()

        if self._conn:
            await self._conn.close()
            self._conn = None
        logger.info("SQLite disconnected")

    async def _configure(self) -> None:
        """Cai dat SQLite PRAGMA."""
        if not self._conn:
            raise RuntimeError("DB not connected")

        await self._conn.execute("PRAGMA journal_mode = WAL")
        await self._conn.execute("PRAGMA synchronous = NORMAL")
        await self._conn.execute("PRAGMA cache_size = 10000")
        await self._conn.execute("PRAGMA mmap_size = 268435456")  # 256MB
        await self._conn.execute("PRAGMA temp_store = MEMORY")
        await self._conn.execute("PRAGMA foreign_keys = ON")

    # =========================================================================
    # Migrations
    # =========================================================================
    async def _run_migrations(self) -> None:
        """Chay migrations theo thu tu neu can."""
        if not self._conn:
            raise RuntimeError("DB not connected")

        cursor = await self._conn.execute("PRAGMA user_version")
        row = await cursor.fetchone()
        current_version = row[0] if row else 0

        logger.info(f"SQLite schema version: {current_version} (target: {CURRENT_SCHEMA_VERSION})")

        migrations_dir = Path(MIGRATIONS_DIR)
        for version in range(current_version + 1, CURRENT_SCHEMA_VERSION + 1):
            sql_file = migrations_dir / f"migration_{version:03d}_initial_schema.sql"
            if not sql_file.exists():
                sql_file = migrations_dir / f"migration_{version:03d}_add_indexes.sql"

            if sql_file.exists():
                logger.info(f"Running migration {version:03d}: {sql_file.name}")
                with open(sql_file, encoding="utf-8") as f:
                    sql_content = f.read()

                # Use thread executor for sync sqlite3 to avoid aiosqlite's
                # threading/transaction complexity.
                def _exec_migration() -> None:
                    import sqlite3
                    conn = sqlite3.connect(str(self._db_path), isolation_level=None)
                    try:
                        conn.executescript(sql_content)
                    finally:
                        conn.close()

                await asyncio.to_thread(_exec_migration)
                await self._conn.execute(f"PRAGMA user_version = {version}")
                logger.info(f"Migration {version:03d} completed")
            else:
                logger.warning(f"Migration {version:03d} SQL file not found: {sql_file}")

    @staticmethod
    def _split_sql_statements(sql: str) -> list[str]:
        """Tach SQL thanh cac statement, loai bo comments va empty lines."""
        import re
        # Remove SQL comments (-- style)
        cleaned = re.sub(r"--[^\n]*\n", "\n", sql)
        # Remove blank lines
        cleaned = re.sub(r"\n\s*\n", "\n", cleaned)
        statements = []
        for stmt in cleaned.split(";"):
            stmt = stmt.strip()
            if stmt and not stmt.startswith("--"):
                statements.append(stmt)
        return statements

    # =========================================================================
    # Write Queue
    # =========================================================================
    async def _enqueue_write(self, sql: str, params: tuple[Any, ...]) -> None:
        """Enqueue mot write operation (non-blocking)."""
        job = _WriteJob(sql=sql, params=params)
        try:
            self._write_queue.put_nowait(job)
        except asyncio.QueueFull:
            logger.error("SQLite write queue full, flushing...")
            await self._flush_all()
            self._write_queue.put_nowait(job)

    async def _flush_loop(self) -> None:
        """Background loop: flush queue every 100ms."""
        while self._running:
            try:
                await asyncio.sleep(self._write_batch_interval)
                await self._flush_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Flush loop error: {e}")

    async def _flush_batch(self) -> None:
        """Flush mot batch of jobs trong mot transaction."""
        if not self._conn or self._write_queue.empty():
            return

        jobs: list[_WriteJob] = []
        while not self._write_queue.empty() and len(jobs) < 500:
            try:
                jobs.append(self._write_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        if not jobs:
            return

        try:
            await self._conn.execute("BEGIN")
            for job in jobs:
                await self._conn.execute(job.sql, job.params)
            await self._conn.execute("COMMIT")
        except Exception as e:
            logger.error(f"Batch write failed: {e}")
            # Retry individually
            await self._conn.execute("ROLLBACK")
            for job in jobs:
                if job.retry_count < job.max_retries:
                    job.retry_count += 1
                    try:
                        await self._conn.execute(job.sql, job.params)
                    except Exception:
                        logger.error(f"Write retry failed after {job.retry_count} attempts")
                else:
                    logger.error(f"Write dropped after {job.max_retries} retries: {job.sql[:100]}")

    async def flush(self) -> None:
        """Flush tat ca trong queue ngay lap tuc (doi cho flush xong)."""
        if not self._conn:
            return
        await self._flush_batch()

    async def flush_all(self) -> None:
        """Flush tat ca con lai trong queue."""
        while not self._write_queue.empty():
            await self._flush_batch()

    # =========================================================================
    # Predictions CRUD
    # =========================================================================
    async def insert_prediction(self, pred: ModelAPrediction) -> None:
        """
        Chen prediction moi vao bang predictions.
        Su dung async queue, khong blocking.
        """
        now_str = datetime.now(timezone.utc).isoformat()
        bar_str = datetime.fromtimestamp(
            pred.bar_close_time / 1000, tz=timezone.utc
        ).isoformat()

        zone_pred_json = None
        if hasattr(pred, "zone_predictions") and pred.zone_predictions:
            zone_pred_json = json.dumps([zp.to_dict() if hasattr(zp, "to_dict") else zp for zp in pred.zone_predictions])

        sql = """
        INSERT OR REPLACE INTO predictions (
            prediction_id, symbol, bar_close_time, bar_close_time_str,
            p_bsl, p_ssl, p_lateral, predicted_bsl_level, predicted_ssl_level,
            bsl_tf, ssl_tf, confidence_qualifier, model_version, inference_latency_ms,
            session_id, macro_regime, i_news, active_guardrail,
            zone_predictions_json, consensus_rating, consensus_direction,
            consensus_agreement, consensus_conviction_price, debate_used_fallback,
            debate_latency_ms, rag_precedents_count, rag_precedents_json,
            outcome_determined, outcome, outcome_time,
            ic_at_prediction, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = (
            getattr(pred, "prediction_id", None) or f"pred_{pred.symbol}_{pred.bar_close_time}",
            pred.symbol,
            pred.bar_close_time,
            bar_str,
            pred.p_bsl,
            pred.p_ssl,
            pred.p_lateral,
            pred.predicted_bsl_level,
            pred.predicted_ssl_level,
            pred.bsl_tf.value if hasattr(pred.bsl_tf, "value") else str(pred.bsl_tf),
            pred.ssl_tf.value if hasattr(pred.ssl_tf, "value") else str(pred.ssl_tf),
            pred.confidence_qualifier.value if hasattr(pred.confidence_qualifier, "value") else str(pred.confidence_qualifier),
            pred.model_version,
            pred.inference_latency_ms,
            getattr(pred, "session_id", "ASIAN"),
            getattr(pred, "macro_regime", "NORMAL"),
            getattr(pred, "i_news", 0.0),
            getattr(pred, "active_guardrail", False),
            zone_pred_json,
            getattr(pred, "consensus_rating", None),
            getattr(pred, "consensus_direction", None),
            getattr(pred, "consensus_agreement", None),
            getattr(pred, "consensus_conviction_price", None),
            getattr(pred, "debate_used_fallback", False),
            getattr(pred, "debate_latency_ms", None),
            getattr(pred, "rag_precedents_count", 0),
            getattr(pred, "rag_precedents_json", None),
            0,  # outcome_determined = False
            None,
            None,
            getattr(pred, "ic_at_prediction", None),
            now_str,
        )
        await self._enqueue_write(sql, params)

    async def update_prediction_outcome(
        self,
        prediction_id: str,
        outcome: PredictionOutcome | str,
        outcome_time: int,
    ) -> None:
        """Cap nhat ket qua prediction sau khi xac dinh."""
        outcome_val = outcome.value if hasattr(outcome, "value") else str(outcome)
        sql = """
        UPDATE predictions
        SET outcome_determined = 1, outcome = ?, outcome_time = ?
        WHERE prediction_id = ?
        """
        await self._enqueue_write(sql, (outcome_val, outcome_time, prediction_id))

    async def update_prediction_outcome_by_bar_time(
        self,
        symbol: str,
        bar_close_time: int,
        outcome: PredictionOutcome | str,
        outcome_time: int,
    ) -> None:
        """Cap nhat outcome bang symbol + bar_close_time (khi khong co prediction_id)."""
        outcome_val = outcome.value if hasattr(outcome, "value") else str(outcome)
        sql = """
        UPDATE predictions
        SET outcome_determined = 1, outcome = ?, outcome_time = ?
        WHERE symbol = ? AND bar_close_time = ? AND outcome_determined = 0
        """
        await self._enqueue_write(sql, (outcome_val, outcome_time, symbol, bar_close_time))

    async def get_pending_predictions(
        self,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Lay predictions chua xac dinh outcome."""
        if not self._conn:
            raise RuntimeError("DB not connected")

        if symbol:
            sql = """
            SELECT * FROM predictions
            WHERE outcome_determined = 0 AND symbol = ?
            ORDER BY bar_close_time DESC
            LIMIT ?
            """
            cursor = await self._conn.execute(sql, (symbol, limit))
        else:
            sql = """
            SELECT * FROM predictions
            WHERE outcome_determined = 0
            ORDER BY bar_close_time DESC
            LIMIT ?
            """
            cursor = await self._conn.execute(sql, (limit,))

        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_prediction_by_id(self, prediction_id: str) -> dict[str, Any] | None:
        """Lay prediction bang ID."""
        if not self._conn:
            raise RuntimeError("DB not connected")
        cursor = await self._conn.execute(
            "SELECT * FROM predictions WHERE prediction_id = ?", (prediction_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_predictions_by_symbol_time(
        self,
        symbol: str,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Lay predictions theo symbol va khoang thoi gian."""
        if not self._conn:
            raise RuntimeError("DB not connected")

        if start_time and end_time:
            sql = """
            SELECT * FROM predictions
            WHERE symbol = ? AND bar_close_time BETWEEN ? AND ?
            ORDER BY bar_close_time DESC LIMIT ?
            """
            cursor = await self._conn.execute(sql, (symbol, start_time, end_time, limit))
        elif start_time:
            sql = """
            SELECT * FROM predictions
            WHERE symbol = ? AND bar_close_time >= ?
            ORDER BY bar_close_time DESC LIMIT ?
            """
            cursor = await self._conn.execute(sql, (symbol, start_time, limit))
        elif end_time:
            sql = """
            SELECT * FROM predictions
            WHERE symbol = ? AND bar_close_time <= ?
            ORDER BY bar_close_time DESC LIMIT ?
            """
            cursor = await self._conn.execute(sql, (symbol, end_time, limit))
        else:
            sql = """
            SELECT * FROM predictions
            WHERE symbol = ?
            ORDER BY bar_close_time DESC LIMIT ?
            """
            cursor = await self._conn.execute(sql, (symbol, limit))

        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    # =========================================================================
    # Zone History CRUD
    # =========================================================================
    async def insert_zone_history(self, zone: Zone) -> None:
        """Chen zone moi vao bang zone_history."""
        now_str = datetime.now(timezone.utc).isoformat()
        zone_type_str = zone.zone_type.value if hasattr(zone.zone_type, "value") else str(zone.zone_type)
        tf_str = zone.timeframe.value if hasattr(zone.timeframe, "value") else str(zone.timeframe)
        htf_str = zone.htf_tf.value if zone.htf_tf and hasattr(zone.htf_tf, "value") else zone.htf_tf

        sql = """
        INSERT OR REPLACE INTO zone_history (
            zone_id, symbol, timeframe, zone_type, top, bottom, ce,
            formed_time, htf_tf, p_hold, w_zone, iii_formation, touch_count,
            initial_status, result, result_time, result_bar_close_time,
            session_id, macro_regime, prediction_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = (
            zone.id,
            zone.symbol,
            tf_str,
            zone_type_str,
            zone.top,
            zone.bottom,
            zone.ce,
            zone.formed_time,
            htf_str,
            zone.p_hold,
            zone.w_zone,
            zone.iii_formation,
            zone.touch_count,
            zone.status.value if hasattr(zone.status, "value") else str(zone.status),
            getattr(zone, "result", None),
            getattr(zone, "result_time", None),
            getattr(zone, "result_bar_close_time", None),
            getattr(zone, "session_id", "ASIAN"),
            getattr(zone, "macro_regime", "NORMAL"),
            getattr(zone, "prediction_id", None),
            now_str,
        )
        await self._enqueue_write(sql, params)

    async def update_zone_history_result(
        self,
        zone_id: str,
        result: str,
        result_time: int,
        result_bar_close_time: int,
    ) -> None:
        """Cap nhat ket qua zone (HOLD, PARTIAL_HOLD, MITIGATED_EARLY, BROKEN)."""
        sql = """
        UPDATE zone_history
        SET result = ?, result_time = ?, result_bar_close_time = ?
        WHERE zone_id = ?
        """
        await self._enqueue_write(sql, (result, result_time, result_bar_close_time, zone_id))

    async def get_zone_history(
        self,
        symbol: str,
        zone_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Lay lich su zone."""
        if not self._conn:
            raise RuntimeError("DB not connected")

        if zone_type:
            sql = """
            SELECT * FROM zone_history
            WHERE symbol = ? AND zone_type = ?
            ORDER BY formed_time DESC LIMIT ?
            """
            cursor = await self._conn.execute(sql, (symbol, zone_type, limit))
        else:
            sql = """
            SELECT * FROM zone_history
            WHERE symbol = ?
            ORDER BY formed_time DESC LIMIT ?
            """
            cursor = await self._conn.execute(sql, (symbol, limit))

        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    # =========================================================================
    # Model Performance CRUD
    # =========================================================================
    async def insert_model_performance(self, metrics: dict[str, Any]) -> None:
        """Chen model performance metrics."""
        now_str = datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO model_performance (
            symbol, model_name, evaluation_window_start, evaluation_window_end,
            window_label, ic_bsl, ic_ssl, ic_composite, brier_score, ece,
            f1_hold, precision_hold, recall_hold, optimal_threshold,
            feature_drift_score, drifted_features_json, model_version,
            n_total, n_bsl_hit, n_ssl_hit, n_lateral, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = (
            metrics.get("symbol"),
            metrics.get("model_name"),
            metrics.get("evaluation_window_start"),
            metrics.get("evaluation_window_end"),
            metrics.get("window_label"),
            metrics.get("ic_bsl"),
            metrics.get("ic_ssl"),
            metrics.get("ic_composite"),
            metrics.get("brier_score"),
            metrics.get("ece"),
            metrics.get("f1_hold"),
            metrics.get("precision_hold"),
            metrics.get("recall_hold"),
            metrics.get("optimal_threshold"),
            metrics.get("feature_drift_score"),
            json.dumps(metrics.get("drifted_features", [])),
            metrics.get("model_version"),
            metrics.get("n_total", 0),
            metrics.get("n_bsl_hit", 0),
            metrics.get("n_ssl_hit", 0),
            metrics.get("n_lateral", 0),
            now_str,
        )
        await self._enqueue_write(sql, params)

    async def get_model_performance_trend(
        self,
        symbol: str,
        model_name: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Lay performance trend cua model."""
        if not self._conn:
            raise RuntimeError("DB not connected")
        sql = """
        SELECT * FROM model_performance
        WHERE symbol = ? AND model_name = ?
        ORDER BY evaluation_window_end DESC LIMIT ?
        """
        cursor = await self._conn.execute(sql, (symbol, model_name, limit))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    # =========================================================================
    # System Metrics CRUD
    # =========================================================================
    async def insert_system_metrics(self, metrics: dict[str, Any]) -> None:
        """Chen system metrics."""
        now_str = datetime.now(timezone.utc).isoformat()
        now_ms = metrics.get("timestamp", int(datetime.now(timezone.utc).timestamp() * 1000))
        sql = """
        INSERT INTO system_metrics (
            timestamp, timestamp_str, component, latency_avg_ms, latency_p95_ms,
            latency_p99_ms, messages_per_sec, throughput_per_sec, memory_mb,
            redis_memory_mb, redis_memory_pct, redis_hit_rate, redis_key_count,
            itq_queue_depth, feed_failure, calendar_stale, model_degraded,
            ic_rolling_20, brier_score_50, ece_rolling_20, tags_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = (
            now_ms,
            now_str,
            metrics.get("component", "system"),
            metrics.get("latency_avg_ms"),
            metrics.get("latency_p95_ms"),
            metrics.get("latency_p99_ms"),
            metrics.get("messages_per_sec"),
            metrics.get("throughput_per_sec"),
            metrics.get("memory_mb"),
            metrics.get("redis_memory_mb"),
            metrics.get("redis_memory_pct"),
            metrics.get("redis_hit_rate"),
            metrics.get("redis_key_count"),
            metrics.get("itq_queue_depth"),
            metrics.get("feed_failure", 0),
            metrics.get("calendar_stale", 0),
            metrics.get("model_degraded", 0),
            metrics.get("ic_rolling_20"),
            metrics.get("brier_score_50"),
            metrics.get("ece_rolling_20"),
            json.dumps(metrics.get("tags", {})),
        )
        await self._enqueue_write(sql, params)

    # =========================================================================
    # Pending Archive CRUD
    # =========================================================================
    async def insert_pending_archive(
        self,
        content_type: str,
        content_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Chen vao pending_archive queue (fallback khi VectorDB down)."""
        now_str = datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT OR REPLACE INTO pending_archive (
            content_type, content_id, payload_json, attempts, last_attempt,
            last_error, max_attempts, status, created_at, updated_at
        ) VALUES (?, ?, ?, 0, NULL, NULL, 5, 'pending', ?, ?)
        """
        await self._enqueue_write(
            sql,
            (content_type, content_id, json.dumps(payload), now_str, now_str),
        )

    async def get_pending_archive_items(
        self,
        content_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Lay cac pending archive items de retry."""
        if not self._conn:
            raise RuntimeError("DB not connected")

        if content_type:
            sql = """
            SELECT * FROM pending_archive
            WHERE status = 'pending' AND content_type = ?
            AND attempts < max_attempts
            ORDER BY created_at ASC LIMIT ?
            """
            cursor = await self._conn.execute(sql, (content_type, limit))
        else:
            sql = """
            SELECT * FROM pending_archive
            WHERE status = 'pending' AND attempts < max_attempts
            ORDER BY created_at ASC LIMIT ?
            """
            cursor = await self._conn.execute(sql, (limit,))

        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def mark_archive_completed(self, record_id: int) -> None:
        """Danh dau archive item da hoan thanh."""
        sql = "UPDATE pending_archive SET status = 'completed' WHERE record_id = ?"
        await self._enqueue_write(sql, (record_id,))

    async def mark_archive_failed(
        self,
        record_id: int,
        error: str,
    ) -> None:
        """Danh dau archive item that bai, tang retry count."""
        sql = """
        UPDATE pending_archive
        SET attempts = attempts + 1, last_attempt = ?, last_error = ?, status = 'failed'
        WHERE record_id = ?
        """
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        await self._enqueue_write(sql, (now_ms, error, record_id))

    # =========================================================================
    # Direct Read Helpers (synchronous-like, non-queued)
    # =========================================================================
    async def get_db_info(self) -> dict[str, Any]:
        """Tra ve database info (size, WAL size, page count)."""
        if not self._conn:
            raise RuntimeError("DB not connected")
        cursor = await self._conn.execute(
            "SELECT page_count * page_size as size_bytes FROM pragma_page_count(), pragma_page_size()"
        )
        row = await cursor.fetchone()
        return {
            "size_bytes": row[0] if row else 0,
            "size_mb": round((row[0] if row else 0) / (1024 * 1024), 2),
            "path": str(self._db_path),
        }

    @property
    def write_queue_size(self) -> int:
        """So luong write jobs trong queue."""
        return self._write_queue.qsize()

    async def flush(self) -> None:
        """Manual flush - dam bao tat ca writes da duoc execute."""
        await self.flush_all()
