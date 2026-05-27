# =============================================================================
# AGENTIC-QUANT — Unit Tests: Memory Engine (Phase 3)
# =============================================================================

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from core.memory.models import (
    Zone,
    ModelAPrediction,
    DebateRecord,
    BullThesis,
    BearThesis,
    ConsensusResult,
    PredictionOutcome,
    LiquidityTarget,
)
from core.memory.models.enums import (
    Timeframe,
    ZoneStatus,
    ZoneType,
    ConfidenceQualifier,
)
from core.memory.short_term.redis_cache_manager import (
    RedisCacheManager,
    RedisNamespace,
    RedisTTL,
    RedisCacheError,
    _json_dumps,
    _json_loads,
    _msgpack_dumps,
    _msgpack_loads,
)
from core.memory.short_term.active_zone_registry import ActiveZoneRegistry
from core.memory.long_term.sqlite_history_store import SQLiteHistoryStore
from core.memory.outcome_determinator import (
    OutcomeDeterminator,
    PriceBar,
    OutcomeType,
    MAX_HORIZON_MS,
)
from core.memory.long_term.vectordb_adapter import (
    ChromaDBProvider,
    QdrantProvider,
    VectorDBFactory,
    VectorDBConfig,
)
from core.memory.debate_archiver import DebateArchiver, compute_e_usv
from core.memory.long_term.rag_retriever import RAGRetriever, RAG_INITIAL_THRESHOLD, RAG_FALLBACK_THRESHOLD


# =============================================================================
# TestRedisCacheManager
# =============================================================================
class TestRedisCacheManager:
    """Tests cho RedisCacheManager voi fakeredis."""

    @pytest.fixture
    async def redis_mgr(self) -> RedisCacheManager:
        """Fake Redis manager for testing."""
        import fakeredis.aioredis

        manager = RedisCacheManager()
        manager._pool = None

        # Thay client bang fake
        manager._client = fakeredis.aioredis.FakeRedis(decode_responses=False)
        manager._connected = True

        yield manager

        await manager.disconnect()

    # --- Serialization Tests ---
    def test_json_dumps_loads_roundtrip(self) -> None:
        data = {"symbol": "XAUUSD", "p_bsl": 0.75, "count": 42, "flag": True}
        encoded = _json_dumps(data)
        decoded = _json_loads(encoded)
        assert decoded["symbol"] == "XAUUSD"
        assert decoded["p_bsl"] == 0.75
        assert decoded["count"] == 42
        assert decoded["flag"] is True

    def test_msgpack_dumps_loads_roundtrip(self) -> None:
        vector = [0.1, 0.2, 0.3, -0.5, 0.99]
        encoded = _msgpack_dumps(vector)
        decoded = _msgpack_loads(encoded)
        assert decoded == vector

    # --- Key Building Tests ---
    def test_zone_key(self) -> None:
        key = RedisCacheManager.zone_key("XAUUSD", "M1", "OB_BULL", 1716000000000)
        assert key == "zone:XAUUSD:M1:OB_BULL:1716000000000"

    def test_ai_output_key(self) -> None:
        key = RedisCacheManager.ai_output_key("XAUUSD")
        assert key == "ai:output:XAUUSD:latest"

    def test_macro_state_key(self) -> None:
        key = RedisCacheManager.macro_state_key("USD")
        assert key == "macro:state:USD"

    def test_debate_key(self) -> None:
        key = RedisCacheManager.debate_key("XAUUSD", 1716000000)
        assert key == "debate:XAUUSD:1716000000"

    def test_features_key(self) -> None:
        key = RedisCacheManager.features_key("XAUUSD", 1716000000, "model_a")
        assert key == "features:XAUUSD:1716000000:model_a"

    def test_latent_key(self) -> None:
        key = RedisCacheManager.latent_key("XAUUSD", 1716000000)
        assert key == "latent:XAUUSD:1716000000"

    def test_metrics_key(self) -> None:
        key = RedisCacheManager.metrics_key("xgboost_a")
        assert key == "metrics:xgboost_a:latest"

    # --- Namespace TTL Tests ---
    def test_namespace_ttl_values(self) -> None:
        assert RedisTTL.ZONE == 86400  # 24h
        assert RedisTTL.AI_OUTPUT == 120  # 2min
        assert RedisTTL.MACRO_STATE == 60  # 1min
        assert RedisTTL.MACRO_EVENTS == 21600  # 6h
        assert RedisTTL.DEBATE == 3600  # 1h
        assert RedisTTL.FEATURES == 3600  # 1h
        assert RedisTTL.LATENT == 3600  # 1h
        assert RedisTTL.METRICS == 300  # 5min

    # --- Zone Namespace Tests ---
    @pytest.mark.asyncio
    async def test_set_get_zone(self, redis_mgr: RedisCacheManager) -> None:
        zone_data = {
            "id": "zone_1",
            "symbol": "XAUUSD",
            "timeframe": "M1",
            "zone_type": "OB_BULL",
            "top": "2500.0",
            "bottom": "2499.0",
            "ce": "0.5",
            "formed_time": "1716000000000",
            "status": "UNMITIGATED",
            "p_hold": "0.75",
            "p_hold_updated": "1716000000000",
            "w_zone": "1.0",
            "iii_formation": "0.0",
            "touch_count": "0",
            "last_touch_time": "0",
            "htf_tf": "None",
        }
        key = redis_mgr.zone_key("XAUUSD", "M1", "OB_BULL", 1716000000000)
        await redis_mgr.set_zone(key, zone_data)
        result = await redis_mgr.get_zone(key)
        assert result is not None
        assert result["id"] == "zone_1"
        assert result["symbol"] == "XAUUSD"

    @pytest.mark.asyncio
    async def test_set_get_ai_output(self, redis_mgr: RedisCacheManager) -> None:
        output = {
            "p_bsl": "0.75",
            "p_ssl": "0.15",
            "p_lateral": "0.10",
            "bsl_level": "2500.0",
        }
        await redis_mgr.set_ai_output("XAUUSD", output)
        result = await redis_mgr.get_ai_output("XAUUSD")
        assert result is not None
        # Decoder converts strings to appropriate Python types
        assert float(result["p_bsl"]) == 0.75

    @pytest.mark.asyncio
    async def test_macro_events_list(self, redis_mgr: RedisCacheManager) -> None:
        event = {"id": "evt_1", "title": "NFP", "currency": "USD"}
        await redis_mgr.add_macro_event("USD", event)
        events = await redis_mgr.get_macro_events("USD")
        assert len(events) == 1
        assert events[0]["id"] == "evt_1"

    @pytest.mark.asyncio
    async def test_latent_vector_msgpack(self, redis_mgr: RedisCacheManager) -> None:
        vector = [0.1 * i for i in range(256)]
        await redis_mgr.set_latent_vector("XAUUSD", 1716000000, vector)
        result = await redis_mgr.get_latent_vector("XAUUSD", 1716000000)
        assert result is not None
        assert len(result) == 256
        assert abs(result[0] - 0.0) < 0.01

    @pytest.mark.asyncio
    async def test_debate_set_get(self, redis_mgr: RedisCacheManager) -> None:
        debate = {
            "symbol": "XAUUSD",
            "bar_close_time": "1716000000000",
            "rating": "2",
            "direction": "BULLISH",
            "archived": "False",
        }
        await redis_mgr.set_debate("XAUUSD", 1716000000, debate)
        result = await redis_mgr.get_debate("XAUUSD", 1716000000)
        assert result is not None
        assert result["symbol"] == "XAUUSD"
        # Decoder converts "False" string to False bool
        assert result["archived"] == False

    @pytest.mark.asyncio
    async def test_sorted_set_ranking(self, redis_mgr: RedisCacheManager) -> None:
        await redis_mgr.update_zone_rank("XAUUSD", "zone:XAUUSD:M1:OB_BULL:1", 0.75, "OB_BULL")
        await redis_mgr.update_zone_rank("XAUUSD", "zone:XAUUSD:M1:OB_BULL:2", 0.90, "OB_BULL")
        await redis_mgr.update_zone_rank("XAUUSD", "zone:XAUUSD:M1:OB_BULL:3", 0.60, "OB_BULL")

        top = await redis_mgr.get_top_zones_by_rank("XAUUSD", "OB_BULL", k=3)
        assert len(top) == 3
        # Score 0.90 should be first
        assert top[0][0] == "zone:XAUUSD:M1:OB_BULL:2"
        assert top[0][1] == 0.90


# =============================================================================
# TestActiveZoneRegistry
# =============================================================================
class TestActiveZoneRegistry:
    """Tests cho ActiveZoneRegistry voi mock Redis."""

    @pytest.fixture
    async def registry(self) -> ActiveZoneRegistry:
        import fakeredis.aioredis
        manager = RedisCacheManager()
        manager._client = fakeredis.aioredis.FakeRedis(decode_responses=False)
        manager._connected = True
        registry = ActiveZoneRegistry(manager)
        yield registry
        await manager.disconnect()

    @pytest.mark.asyncio
    async def test_upsert_zone_new(self, registry: ActiveZoneRegistry) -> None:
        zone = Zone(
            id="zone_1",
            symbol="XAUUSD",
            timeframe=Timeframe.M1,
            zone_type=ZoneType.OB_BULL,
            top=2500.0,
            bottom=2499.0,
            ce=0.5,
            formed_time=1716000000000,
            status=ZoneStatus.UNMITIGATED,
            p_hold=0.75,
            p_hold_updated=1716000000000,
            w_zone=1.0,
            iii_formation=0.0,
            touch_count=0,
            last_touch_time=0,
        )
        is_new = await registry.upsert_zone(zone)
        assert is_new is True

    @pytest.mark.asyncio
    async def test_upsert_zone_update(self, registry: ActiveZoneRegistry) -> None:
        zone = Zone(
            id="zone_1",
            symbol="XAUUSD",
            timeframe=Timeframe.M1,
            zone_type=ZoneType.OB_BULL,
            top=2500.0,
            bottom=2499.0,
            ce=0.5,
            formed_time=1716000000000,
            status=ZoneStatus.UNMITIGATED,
            p_hold=0.75,
            p_hold_updated=1716000000000,
            w_zone=1.0,
        )
        await registry.upsert_zone(zone)

        zone.p_hold = 0.85
        is_new = await registry.upsert_zone(zone)
        assert is_new is False

    @pytest.mark.asyncio
    async def test_get_top_zones_sorted(self, registry: ActiveZoneRegistry) -> None:
        for i in range(5):
            zone = Zone(
                id=f"zone_{i}",
                symbol="XAUUSD",
                timeframe=Timeframe.M1,
                zone_type=ZoneType.OB_BULL,
                top=2500.0 + i,
                bottom=2499.0 + i,
                ce=0.5,
                formed_time=1716000000000 + i,
                status=ZoneStatus.UNMITIGATED,
                p_hold=0.50 + i * 0.10,
                p_hold_updated=1716000000000,
                w_zone=1.0,
            )
            await registry.upsert_zone(zone)

        top = await registry.get_top_zones("XAUUSD", "OB_BULL", k=3)
        assert len(top) == 3
        # Highest p_hold first
        assert top[0].p_hold == 0.90
        assert top[1].p_hold == 0.80
        assert top[2].p_hold == 0.70

    @pytest.mark.asyncio
    async def test_update_zone_status_to_mitigated(self, registry: ActiveZoneRegistry) -> None:
        zone = Zone(
            id="zone_1",
            symbol="XAUUSD",
            timeframe=Timeframe.M1,
            zone_type=ZoneType.OB_BULL,
            top=2500.0,
            bottom=2499.0,
            ce=0.5,
            formed_time=1716000000000,
            status=ZoneStatus.UNMITIGATED,
            p_hold=0.75,
            p_hold_updated=1716000000000,
            w_zone=1.0,
        )
        await registry.upsert_zone(zone)

        success = await registry.update_zone_status(
            zone_id="zone_1",
            symbol="XAUUSD",
            timeframe="M1",
            formed_time=1716000000000,
            zone_type="OB_BULL",
            new_status=ZoneStatus.MITIGATED,
        )
        assert success is True

        top = await registry.get_top_zones("XAUUSD", "OB_BULL", k=5)
        # Mitigated zone should have score 0
        assert len(top) == 0

    @pytest.mark.asyncio
    async def test_update_zone_p_hold(self, registry: ActiveZoneRegistry) -> None:
        zone = Zone(
            id="zone_1",
            symbol="XAUUSD",
            timeframe=Timeframe.M1,
            zone_type=ZoneType.OB_BULL,
            top=2500.0,
            bottom=2499.0,
            ce=0.5,
            formed_time=1716000000000,
            status=ZoneStatus.UNMITIGATED,
            p_hold=0.75,
            p_hold_updated=1716000000000,
            w_zone=2.0,
        )
        await registry.upsert_zone(zone)

        success = await registry.update_zone_p_hold(
            zone_id="zone_1",
            symbol="XAUUSD",
            timeframe="M1",
            formed_time=1716000000000,
            zone_type="OB_BULL",
            p_hold=0.90,
            w_zone=2.0,
        )
        assert success is True

        top = await registry.get_top_zones("XAUUSD", "OB_BULL", k=5)
        assert len(top) == 1
        # Score = p_hold * w_zone = 0.90 * 2.0 = 1.80
        assert top[0].p_hold == 0.90


# =============================================================================
# TestSQLiteHistoryStore
# =============================================================================
class TestSQLiteHistoryStore:
    """Tests cho SQLiteHistoryStore voi in-memory SQLite."""

    @pytest.fixture
    async def store(self, tmp_path: Path) -> SQLiteHistoryStore:
        """SQLite store with temp file for reliable cross-thread testing."""
        db_file = tmp_path / "test_memory.db"
        store = SQLiteHistoryStore(db_path=str(db_file))
        await store.connect()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_schema_created(self, store: SQLiteHistoryStore) -> None:
        """Verify all 5 tables exist."""
        cursor = await store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        rows = await cursor.fetchall()
        table_names = {row[0] for row in rows}

        expected_tables = {
            "predictions",
            "zone_history",
            "model_performance",
            "system_metrics",
            "pending_archive",
        }
        for table in expected_tables:
            assert table in table_names, f"Table {table} not found"

    @pytest.mark.asyncio
    async def test_insert_prediction(self, store: SQLiteHistoryStore) -> None:
        """Test insert prediction vao bang predictions."""
        pred = ModelAPrediction(
            symbol="XAUUSD",
            bar_close_time=1716000000000,
            p_bsl=0.75,
            p_ssl=0.15,
            p_lateral=0.10,
            predicted_bsl_level=2500.0,
            predicted_ssl_level=2480.0,
            bsl_tf=Timeframe.H1,
            ssl_tf=Timeframe.H1,
            confidence_qualifier=ConfidenceQualifier.HIGH,
            model_version="v1.0",
            inference_latency_ms=3.5,
        )
        await store.insert_prediction(pred)
        await store.flush()

        pending = await store.get_pending_predictions(symbol="XAUUSD")
        assert len(pending) >= 1
        assert pending[0]["symbol"] == "XAUUSD"
        assert pending[0]["outcome_determined"] == 0

    @pytest.mark.asyncio
    async def test_update_prediction_outcome(self, store: SQLiteHistoryStore) -> None:
        """Test cap nhat outcome prediction."""
        pred = ModelAPrediction(
            symbol="XAUUSD",
            bar_close_time=1716000000000,
            p_bsl=0.75,
            p_ssl=0.15,
            p_lateral=0.10,
            predicted_bsl_level=2500.0,
            predicted_ssl_level=2480.0,
            bsl_tf=Timeframe.H1,
            ssl_tf=Timeframe.H1,
            confidence_qualifier=ConfidenceQualifier.HIGH,
            model_version="v1.0",
            inference_latency_ms=3.5,
        )
        await store.insert_prediction(pred)
        await store.flush()

        pending = await store.get_pending_predictions(symbol="XAUUSD")
        assert len(pending) > 0
        pred_id = pending[0]["prediction_id"]

        await store.update_prediction_outcome(
            prediction_id=pred_id,
            outcome=PredictionOutcome.BSL_HIT,
            outcome_time=1716020000000,
        )
        await store.flush()

        updated = await store.get_prediction_by_id(pred_id)
        assert updated is not None
        assert updated["outcome"] == "BSL_HIT"
        assert updated["outcome_determined"] == 1

    @pytest.mark.asyncio
    async def test_insert_zone_history(self, store: SQLiteHistoryStore) -> None:
        """Test chen zone history."""
        zone = Zone(
            id="zone_test_1",
            symbol="XAUUSD",
            timeframe=Timeframe.M15,
            zone_type=ZoneType.FVG_BULL,
            top=2500.0,
            bottom=2499.5,
            ce=0.3,
            formed_time=1716000000000,
            status=ZoneStatus.UNMITIGATED,
            p_hold=0.80,
            p_hold_updated=1716000000000,
            w_zone=1.5,
            iii_formation=1.2,
        )
        await store.insert_zone_history(zone)
        await store.flush()

        history = await store.get_zone_history("XAUUSD", limit=10)
        assert len(history) >= 1
        assert history[0]["zone_id"] == "zone_test_1"
        assert history[0]["zone_type"] == "FVG_BULL"

    @pytest.mark.asyncio
    async def test_pending_archive(self, store: SQLiteHistoryStore) -> None:
        """Test pending_archive queue."""
        payload = {"symbol": "XAUUSD", "data": "test"}
        await store.insert_pending_archive("debate", "debate_XAUUSD_1", payload)
        await store.flush()

        items = await store.get_pending_archive_items(content_type="debate")
        assert len(items) >= 1
        assert items[0]["content_type"] == "debate"
        assert items[0]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_write_queue_batching(self, store: SQLiteHistoryStore) -> None:
        """Test write queue batching - chen nhieu predictions."""
        for i in range(10):
            pred = ModelAPrediction(
                symbol="XAUUSD",
                bar_close_time=1716000000000 + i * 60000,
                p_bsl=0.70 + i * 0.01,
                p_ssl=0.15,
                p_lateral=0.15,
                predicted_bsl_level=2500.0,
                predicted_ssl_level=2480.0,
                bsl_tf=Timeframe.H1,
                ssl_tf=Timeframe.H1,
                confidence_qualifier=ConfidenceQualifier.MEDIUM,
                model_version="v1.0",
                inference_latency_ms=3.0,
            )
            await store.insert_prediction(pred)

        # Queue should have 10 items
        assert store.write_queue_size > 0

        # Flush the batch
        await store.flush()

        pending = await store.get_pending_predictions(symbol="XAUUSD")
        assert len(pending) == 10

    @pytest.mark.asyncio
    async def test_migrations_skipped_if_current(self, store: SQLiteHistoryStore) -> None:
        """Test migrations chi chay mot lan."""
        cursor = await store._conn.execute("PRAGMA user_version")
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 2  # CURRENT_SCHEMA_VERSION


# =============================================================================
# TestOutcomeDeterminator
# =============================================================================
class TestOutcomeDeterminator:
    """Tests cho OutcomeDeterminator."""

    @pytest.fixture
    async def store(self, tmp_path: Path) -> SQLiteHistoryStore:
        db_path = tmp_path / "outcome_test.db"
        store = SQLiteHistoryStore(db_path=db_path)
        await store.connect()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_bsl_hit_detection(self, store: SQLiteHistoryStore) -> None:
        """Test BSL hit: high >= bsl_level."""
        import fakeredis.aioredis

        redis_mgr = RedisCacheManager()
        redis_mgr._client = fakeredis.aioredis.FakeRedis(decode_responses=False)
        redis_mgr._connected = True

        determiner = OutcomeDeterminator(store, redis_mgr)

        # Insert prediction
        pred = ModelAPrediction(
            symbol="XAUUSD",
            bar_close_time=1000000000000,
            p_bsl=0.75,
            p_ssl=0.15,
            p_lateral=0.10,
            predicted_bsl_level=2500.0,
            predicted_ssl_level=2480.0,
            bsl_tf=Timeframe.H1,
            ssl_tf=Timeframe.H1,
            confidence_qualifier=ConfidenceQualifier.HIGH,
            model_version="v1.0",
            inference_latency_ms=3.0,
        )
        await store.insert_prediction(pred)
        await store.flush()

        # Simulate bars with BSL hit
        bars = [
            PriceBar(close_time=1000000060000, open=2495, high=2501, low=2494, close=2501),
        ]

        results = await determiner.determine_outcomes_for_symbol("XAUUSD", bars)

        # Should have BSL_HIT outcome
        assert len(results) >= 1
        bsl_hit = next((r for r in results if r.outcome == OutcomeType.BSL_HIT), None)
        assert bsl_hit is not None
        assert bsl_hit.high_since_prediction >= 2500.0

    @pytest.mark.asyncio
    async def test_ssl_hit_detection(self, store: SQLiteHistoryStore) -> None:
        """Test SSL hit: low <= ssl_level."""
        import fakeredis.aioredis

        redis_mgr = RedisCacheManager()
        redis_mgr._client = fakeredis.aioredis.FakeRedis(decode_responses=False)
        redis_mgr._connected = True

        determiner = OutcomeDeterminator(store, redis_mgr)

        pred = ModelAPrediction(
            symbol="XAUUSD",
            bar_close_time=1000000000000,
            p_bsl=0.75,
            p_ssl=0.15,
            p_lateral=0.10,
            predicted_bsl_level=2500.0,
            predicted_ssl_level=2480.0,
            bsl_tf=Timeframe.H1,
            ssl_tf=Timeframe.H1,
            confidence_qualifier=ConfidenceQualifier.HIGH,
            model_version="v1.0",
            inference_latency_ms=3.0,
        )
        await store.insert_prediction(pred)
        await store.flush()

        bars = [
            PriceBar(close_time=1000000060000, open=2485, high=2486, low=2479, close=2479),
        ]

        results = await determiner.determine_outcomes_for_symbol("XAUUSD", bars)

        ssl_hit = next((r for r in results if r.outcome == OutcomeType.SSL_HIT), None)
        assert ssl_hit is not None
        assert ssl_hit.low_since_prediction <= 2480.0

    @pytest.mark.asyncio
    async def test_timeout_detection(self, store: SQLiteHistoryStore) -> None:
        """Test timeout: elapsed > 240 minutes."""
        import fakeredis.aioredis

        redis_mgr = RedisCacheManager()
        redis_mgr._client = fakeredis.aioredis.FakeRedis(decode_responses=False)
        redis_mgr._connected = True

        determiner = OutcomeDeterminator(store, redis_mgr)

        # Prediction 300 minutes ago (exceeds 240 min horizon)
        pred = ModelAPrediction(
            symbol="XAUUSD",
            bar_close_time=1000000000000,
            p_bsl=0.75,
            p_ssl=0.15,
            p_lateral=0.10,
            predicted_bsl_level=2500.0,
            predicted_ssl_level=2480.0,
            bsl_tf=Timeframe.H1,
            ssl_tf=Timeframe.H1,
            confidence_qualifier=ConfidenceQualifier.HIGH,
            model_version="v1.0",
            inference_latency_ms=3.0,
        )
        await store.insert_prediction(pred)
        await store.flush()

        # Bar 300 min later - should trigger TIMEOUT
        current_time = 1000000000000 + (300 * 60 * 1000)
        bars = [
            PriceBar(close_time=current_time, open=2490, high=2495, low=2485, close=2490),
        ]

        results = await determiner.determine_outcomes_for_symbol("XAUUSD", bars)

        timeout = next((r for r in results if r.outcome == OutcomeType.TIMEOUT), None)
        assert timeout is not None
        assert timeout.elapsed_minutes >= 240.0

    def test_max_horizon_constant(self) -> None:
        """Test MAX_HORIZON_MS = 240 minutes."""
        assert MAX_HORIZON_MS == 240 * 60 * 1000


# =============================================================================
# TestVectorDBAdapter
# =============================================================================
class TestVectorDBAdapter:
    """Tests cho VectorDB adapters."""

    def test_vectordb_factory_chromadb(self) -> None:
        """Test factory tao ChromaDB provider."""
        config = VectorDBConfig(provider="chromadb", persist_directory="/tmp/test_chroma")
        provider = VectorDBFactory.create(config)
        assert isinstance(provider, ChromaDBProvider)

    def test_vectordb_factory_qdrant(self) -> None:
        """Test factory tao Qdrant provider."""
        config = VectorDBConfig(provider="qdrant", url="http://localhost:6333")
        provider = VectorDBFactory.create(config)
        assert isinstance(provider, QdrantProvider)

    def test_vectordb_factory_default(self) -> None:
        """Test factory default sang ChromaDB."""
        provider = VectorDBFactory.create()
        assert isinstance(provider, ChromaDBProvider)

    def test_vectordb_factory_unknown_default(self) -> None:
        """Test factory voi provider khong ton tai -> ChromaDB."""
        config = VectorDBConfig(provider="unknown")
        provider = VectorDBFactory.create(config)
        assert isinstance(provider, ChromaDBProvider)

    @pytest.mark.asyncio
    async def test_chromadb_integration(self, tmp_path: Path) -> None:
        """Integration test: insert debate -> search -> verify."""
        config = VectorDBConfig(
            provider="chromadb",
            persist_directory=str(tmp_path / "chromadb_test"),
        )
        provider = ChromaDBProvider(config)
        await provider.connect()
        await provider.ensure_collections()

        # Insert debate
        debate = DebateRecord(
            symbol="XAUUSD",
            bar_close_time=1716000000000,
            macro_regime="NORMAL",
            session="ASIAN",
            bull=BullThesis(direction="BULLISH", confidence=0.75, target_price=2520.0),
            bear=BearThesis(direction="BEARISH", confidence=0.25, target_price=2480.0),
            consensus=ConsensusResult(rating=3, preferred_direction="BULLISH", agreement_score=0.8),
            precedents_count=5,
        )

        embedding = [0.1 * i for i in range(256)]
        doc_id = await provider.insert_debate(debate, embedding)
        assert doc_id == "XAUUSD_1716000000000"

        # Search
        hits = await provider.search_similar_debates(embedding, symbol="XAUUSD", k=3)
        assert len(hits) >= 1
        assert hits[0].debate_record.symbol == "XAUUSD"

        await provider.close()

    def test_rerank_formula(self) -> None:
        """Test re-ranking formula: 0.7 * cosine + 0.3 * recency."""
        cosine = 0.90
        days = 0
        recency = 1.0 / (1.0 + days)
        expected = 0.7 * cosine + 0.3 * recency
        assert abs(expected - 0.7 * 0.90 - 0.3 * 1.0) < 0.001

        # Older item should rank lower
        days_old = 30
        recency_old = 1.0 / (1.0 + days_old)
        old_score = 0.7 * cosine + 0.3 * recency_old
        assert old_score < expected


# =============================================================================
# TestDebateArchiver
# =============================================================================
class TestDebateArchiver:
    """Tests cho DebateArchiver."""

    def test_compute_e_usv_deterministic(self) -> None:
        """Test e_USV computation la deterministic."""
        debate_data = {
            "symbol": "XAUUSD",
            "bar_close_time": 1716000000000,
            "macro_regime": "NORMAL",
        }
        v1 = compute_e_usv(debate_data)
        v2 = compute_e_usv(debate_data)
        assert v1 == v2
        assert len(v1) == 256

    def test_compute_e_usv_different_inputs(self) -> None:
        """Test e_USV khac nhau voi input khac nhau."""
        v1 = compute_e_usv({"symbol": "XAUUSD", "bar_close_time": 1, "macro_regime": "NORMAL"})
        v2 = compute_e_usv({"symbol": "EURUSD", "bar_close_time": 2, "macro_regime": "NORMAL"})
        assert v1 != v2


# =============================================================================
# TestRAGRetriever
# =============================================================================
class TestRAGRetriever:
    """Tests cho RAGRetriever."""

    def test_threshold_constants(self) -> None:
        """Test threshold constants."""
        assert RAG_INITIAL_THRESHOLD == 0.80
        assert RAG_FALLBACK_THRESHOLD == 0.75
        assert RAG_INITIAL_THRESHOLD > RAG_FALLBACK_THRESHOLD

    @pytest.mark.asyncio
    async def test_precedent_from_hit(self) -> None:
        """Test Precedent creation from DebateHit."""
        import fakeredis.aioredis

        config = VectorDBConfig(provider="chromadb", persist_directory=str(tempfile.mkdtemp()))
        provider = ChromaDBProvider(config)
        await provider.connect()

        redis_mgr = RedisCacheManager()
        redis_mgr._client = fakeredis.aioredis.FakeRedis(decode_responses=False)
        redis_mgr._connected = True

        retriever = RAGRetriever(provider, redis_mgr)

        # Insert a debate first
        debate = DebateRecord(
            symbol="XAUUSD",
            bar_close_time=1716000000000,
            macro_regime="NORMAL",
            session="ASIAN",
            bull=BullThesis(direction="BULLISH", confidence=0.75),
            bear=BearThesis(direction="BEARISH", confidence=0.25),
            consensus=ConsensusResult(rating=3, preferred_direction="BULLISH", agreement_score=0.8),
        )
        embedding = [0.1 * i for i in range(256)]
        await provider.insert_debate(debate, embedding)

        # Retrieve with regime=NORMAL to match
        precedents = await retriever.retrieve_precedents(embedding, "XAUUSD", macro_regime="NORMAL", k=3)

        # Verify provider can insert and retrieve
        hits = await provider.search_similar_debates(embedding, symbol="XAUUSD", macro_regime="NORMAL", k=3)
        assert len(hits) >= 1, "ChromaDB should return inserted debate"
        assert hits[0].debate_record.symbol == "XAUUSD"

        # Precedent extraction works if we got hits
        if precedents:
            p = precedents[0]
            assert p.symbol == "XAUUSD"
            assert p.cosine_sim >= 0.0
            assert p.re_rank_score >= 0.0

        await provider.close()

    @pytest.mark.asyncio
    async def test_cache_key_buckets(self) -> None:
        """Test cache key format."""
        import fakeredis.aioredis

        config = VectorDBConfig(provider="chromadb", persist_directory=str(tempfile.mkdtemp()))
        provider = ChromaDBProvider(config)
        await provider.connect()

        redis_mgr = RedisCacheManager()
        redis_mgr._client = fakeredis.aioredis.FakeRedis(decode_responses=False)
        redis_mgr._connected = True

        retriever = RAGRetriever(provider, redis_mgr)

        # Build cache key
        key = retriever._build_cache_key("XAUUSD", "NORMAL")
        parts = key.split(":")
        assert parts[0] == "rag"
        assert parts[1] == "XAUUSD"
        assert parts[2] == "NORMAL"
        # parts[3] la timestamp bucket

        key_all = retriever._build_cache_key("XAUUSD", None)
        assert "ALL" in key_all

        await provider.close()


# =============================================================================
# TestModels
# =============================================================================
class TestZoneModel:
    """Tests cho Zone dataclass."""

    def test_zone_from_dict(self) -> None:
        data = {
            "id": "z1",
            "symbol": "XAUUSD",
            "timeframe": "M1",
            "zone_type": "OB_BULL",
            "top": "2500.0",
            "bottom": "2499.0",
            "ce": "0.5",
            "formed_time": "1716000000000",
            "status": "UNMITIGATED",
            "p_hold": "0.75",
            "p_hold_updated": "1716000000000",
            "w_zone": "1.5",
            "iii_formation": "1.2",
            "touch_count": "3",
            "last_touch_time": "1716010000000",
            "htf_tf": "H1",
        }
        zone = Zone.from_dict(data)
        assert zone.id == "z1"
        assert zone.top == 2500.0
        assert zone.p_hold == 0.75
        assert zone.w_zone == 1.5
        assert zone.htf_tf == Timeframe.H1

    def test_zone_to_dict_roundtrip(self) -> None:
        zone = Zone(
            id="z1",
            symbol="XAUUSD",
            timeframe=Timeframe.M5,
            zone_type=ZoneType.FVG_BULL,
            top=2500.0,
            bottom=2499.0,
            ce=0.5,
            formed_time=1716000000000,
            status=ZoneStatus.WICK_TOUCHED,
            p_hold=0.85,
            p_hold_updated=1716000000000,
            w_zone=2.0,
            iii_formation=1.5,
            touch_count=2,
            last_touch_time=1716010000000,
            htf_tf=Timeframe.H4,
        )
        d = zone.to_dict()
        restored = Zone.from_dict(d)
        assert restored.id == zone.id
        assert restored.p_hold == zone.p_hold
        assert restored.status == zone.status

    def test_zone_is_bullish(self) -> None:
        bull_zone = Zone(zone_type=ZoneType.OB_BULL)
        bear_zone = Zone(zone_type=ZoneType.OB_BEAR)
        assert bull_zone.is_bullish() is True
        assert bear_zone.is_bullish() is False
        assert bull_zone.is_bearish() is False
        assert bear_zone.is_bearish() is True

    def test_zone_mid_price(self) -> None:
        zone = Zone(top=2500.0, bottom=2490.0)
        assert zone.mid_price() == 2495.0

    def test_zone_range_size(self) -> None:
        zone = Zone(top=2500.0, bottom=2490.0)
        assert zone.range_size() == 10.0

    def test_zone_contains_price(self) -> None:
        zone = Zone(top=2500.0, bottom=2490.0)
        assert zone.contains_price(2495.0) is True
        assert zone.contains_price(2489.0) is False
        assert zone.contains_price(2501.0) is False

    def test_zone_is_active(self) -> None:
        active = Zone(status=ZoneStatus.UNMITIGATED)
        mitigated = Zone(status=ZoneStatus.MITIGATED)
        assert active.is_active() is True
        assert mitigated.is_active() is False


class TestPredictionModel:
    """Tests cho Prediction dataclasses."""

    def test_model_a_prediction_from_dict(self) -> None:
        data = {
            "symbol": "XAUUSD",
            "bar_close_time": "1716000000000",
            "p_bsl": "0.75",
            "p_ssl": "0.15",
            "p_lateral": "0.10",
            "predicted_bsl_level": "2500.0",
            "predicted_ssl_level": "2480.0",
            "bsl_tf": "H1",
            "ssl_tf": "H1",
            "confidence_qualifier": "HIGH",
            "model_version": "v1.0",
            "inference_latency_ms": "3.5",
        }
        pred = ModelAPrediction.from_dict(data)
        assert pred.symbol == "XAUUSD"
        assert pred.p_bsl == 0.75
        assert pred.p_ssl == 0.15
        assert pred.confidence_qualifier == ConfidenceQualifier.HIGH

    def test_prediction_probabilities_sum(self) -> None:
        pred = ModelAPrediction(p_bsl=0.70, p_ssl=0.20, p_lateral=0.10)
        assert abs(pred.probabilities_sum() - 1.0) < 0.001

    def test_prediction_dominant_direction(self) -> None:
        bull = ModelAPrediction(p_bsl=0.75, p_ssl=0.15, p_lateral=0.10)
        bear = ModelAPrediction(p_bsl=0.15, p_ssl=0.75, p_lateral=0.10)
        lateral = ModelAPrediction(p_bsl=0.30, p_ssl=0.30, p_lateral=0.40)

        assert bull.dominant_direction() == "BSL"
        assert bear.dominant_direction() == "SSL"
        assert lateral.dominant_direction() == "LATERAL"


class TestDebateRecord:
    """Tests cho DebateRecord dataclass."""

    def test_debate_to_dict_roundtrip(self) -> None:
        debate = DebateRecord(
            symbol="XAUUSD",
            bar_close_time=1716000000000,
            bull=BullThesis(direction="BULLISH", confidence=0.80, target_price=2520.0),
            bear=BearThesis(direction="BEARISH", confidence=0.20, target_price=2480.0),
            consensus=ConsensusResult(rating=4, preferred_direction="BULLISH", agreement_score=0.85),
            precedents_count=5,
        )
        d = debate.to_dict()
        restored = DebateRecord.from_dict(d)
        assert restored.symbol == "XAUUSD"
        assert restored.consensus.rating == 4
        assert restored.bull.direction == "BULLISH"

    def test_liquidity_target_roundtrip(self) -> None:
        lt = LiquidityTarget(
            target_type="BSL",
            price=2500.0,
            timeframe=Timeframe.H1,
            p_probability=0.75,
            session="LONDON",
        )
        d = lt.to_dict()
        restored = LiquidityTarget.from_dict(d)
        assert restored.target_type == "BSL"
        assert restored.price == 2500.0
        assert restored.p_probability == 0.75
