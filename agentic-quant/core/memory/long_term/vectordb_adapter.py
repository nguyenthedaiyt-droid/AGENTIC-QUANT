# =============================================================================
# AGENTIC-QUANT — VectorDB Adapter
# Abstraction layer supporting ChromaDB and Qdrant
# =============================================================================

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from core.memory.models import DebateRecord, Zone
from core.memory.models.enums import ZoneType

if TYPE_CHECKING:
    import chromadb
    import qdrant_client


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class DebateHit:
    """Mot ket qua tim kiem debate tu VectorDB."""

    debate_record: DebateRecord
    cosine_sim: float
    days_since: int
    re_rank_score: float
    point_id: str = ""


@dataclass
class ZoneHit:
    """Mot ket qua tim kiem zone tu VectorDB."""

    zone: Zone
    cosine_sim: float
    days_since: int
    re_rank_score: float
    point_id: str = ""


@dataclass
class VectorDBConfig:
    """Cau hinh VectorDB."""

    provider: str = "chromadb"  # "qdrant" | "chromadb"
    url: str = "http://127.0.0.1:6333"
    collection_debate: str = "debate_archive"
    collection_zones: str = "zone_embeddings"
    # ChromaDB specific
    persist_directory: str = "data/chromadb"
    distance_fn: str = "cosine"  # "cosine" | "euclidean" | "manhattan"


# =============================================================================
# Base VectorDB Interface
# =============================================================================
class BaseVectorDB(ABC):
    """Abstract interface for VectorDB providers."""

    @abstractmethod
    async def connect(self) -> None:
        """Ket noi / khoi dong VectorDB."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Dong ket noi."""
        pass

    @abstractmethod
    async def insert_debate(
        self,
        debate: DebateRecord,
        embedding: list[float],
    ) -> str:
        """Chen debate record voi embedding. Tra ve ID."""
        pass

    @abstractmethod
    async def search_similar_debates(
        self,
        embedding: list[float],
        symbol: str,
        macro_regime: str | None = None,
        k: int = 3,
    ) -> list[DebateHit]:
        """Tim kiem similar debates."""
        pass

    @abstractmethod
    async def insert_zone_embedding(
        self,
        zone: Zone,
        embedding: list[float],
    ) -> str:
        """Chen zone embedding. Tra ve ID."""
        pass

    @abstractmethod
    async def search_similar_zones(
        self,
        embedding: list[float],
        symbol: str,
        zone_type: str | None = None,
        k: int = 3,
    ) -> list[ZoneHit]:
        """Tim kiem similar zones."""
        pass

    @abstractmethod
    async def ensure_collections(self) -> None:
        """Dam bao collections da ton tai voi cau hinh dung."""
        pass


# =============================================================================
# ChromaDB Provider
# =============================================================================
class ChromaDBProvider(BaseVectorDB):
    """
    ChromaDB implementation.
    File-based, khong can server rieng.
    Default provider - duoc su dung khi khong cau hinh Qdrant.
    """

    def __init__(self, config: VectorDBConfig) -> None:
        self._config = config
        self._client: chromadb.PersistentClient | None = None
        self._debate_coll: Any | None = None
        self._zone_coll: Any | None = None

    async def connect(self) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(
            path=self._config.persist_directory,
        )
        await self.ensure_collections()
        logger.info(f"ChromaDB connected: {self._config.persist_directory}")

    async def close(self) -> None:
        self._client = None
        self._debate_coll = None
        self._zone_coll = None
        logger.info("ChromaDB disconnected")

    async def ensure_collections(self) -> None:
        if not self._client:
            return

        self._debate_coll = self._client.get_or_create_collection(
            name=self._config.collection_debate,
            metadata={"description": "Debate records archive"},
        )
        self._zone_coll = self._client.get_or_create_collection(
            name=self._config.collection_zones,
            metadata={"description": "Zone embeddings"},
        )

    async def insert_debate(
        self,
        debate: DebateRecord,
        embedding: list[float],
    ) -> str:
        if not self._debate_coll:
            raise RuntimeError("ChromaDB not connected")

        doc_id = f"{debate.symbol}_{debate.bar_close_time}"

        metadata = {
            "symbol": debate.symbol,
            "bar_close_time": debate.bar_close_time,
            "macro_regime": debate.macro_regime,
            "session": debate.session,
            "rating": debate.consensus.rating if debate.consensus else 0,
            "direction": debate.consensus.preferred_direction if debate.consensus else "NEUTRAL",
            "outcome": getattr(debate, "outcome", "") or "",
            "archived": debate.archived,
            "precedents_count": debate.precedents_count,
        }

        self._debate_coll.add(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[self._debate_to_text(debate)],
        )

        logger.debug(f"ChromaDB: inserted debate {doc_id}")
        return doc_id

    async def search_similar_debates(
        self,
        embedding: list[float],
        symbol: str,
        macro_regime: str | None = None,
        k: int = 3,
    ) -> list[DebateHit]:
        if not self._debate_coll:
            return []

        where_filter: dict[str, Any] = {"symbol": symbol}
        if macro_regime:
            where_filter["macro_regime"] = macro_regime

        try:
            results = self._debate_coll.query(
                query_embeddings=[embedding],
                n_results=k,
                where=where_filter,
                include=["metadatas", "distances"],
            )
        except Exception:
            # ChromaDB may not support where filter on all fields
            results = self._debate_coll.query(
                query_embeddings=[embedding],
                n_results=k,
                include=["metadatas", "distances"],
            )
            # Post-filter by symbol
            if results["ids"] and results["ids"][0]:
                filtered_ids = []
                filtered_embs = []
                filtered_metas = []
                filtered_dists = []
                for i, meta in enumerate(results["metadatas"][0]):
                    if isinstance(meta, dict) and meta.get("symbol") == symbol:
                        if macro_regime is None or meta.get("macro_regime") == macro_regime:
                            filtered_ids.append(results["ids"][0][i])
                            filtered_embs.append(embedding)
                            filtered_metas.append(meta)
                            filtered_dists.append(results["distances"][0][i])
                results = {
                    "ids": [filtered_ids],
                    "metadatas": [filtered_metas],
                    "distances": [filtered_dists],
                    "documents": [[]],
                    "embeddings": [],
                }

        hits: list[DebateHit] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return hits

        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i] if results.get("distances") else 1.0
            cosine_sim = 1.0 - distance  # ChromaDB uses distance, convert to similarity

            debate = self._meta_to_debate(meta)
            days_since = self._calc_days_since(meta.get("bar_close_time", 0))
            re_rank = self._rerank(cosine_sim, days_since)

            hits.append(DebateHit(
                debate_record=debate,
                cosine_sim=cosine_sim,
                days_since=days_since,
                re_rank_score=re_rank,
                point_id=str(doc_id),
            ))

        # Re-rank
        hits.sort(key=lambda h: h.re_rank_score, reverse=True)
        return hits[:k]

    async def insert_zone_embedding(
        self,
        zone: Zone,
        embedding: list[float],
    ) -> str:
        if not self._zone_coll:
            raise RuntimeError("ChromaDB not connected")

        doc_id = zone.id

        metadata = {
            "symbol": zone.symbol,
            "timeframe": zone.timeframe.value if hasattr(zone.timeframe, "value") else str(zone.timeframe),
            "zone_type": zone.zone_type.value if hasattr(zone.zone_type, "value") else str(zone.zone_type),
            "outcome": getattr(zone, "outcome", "") or "",
            "formed_time": zone.formed_time,
            "p_hold": zone.p_hold,
            "w_zone": zone.w_zone,
        }

        self._zone_coll.add(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[f"Zone {zone.zone_type} {zone.symbol} p_hold={zone.p_hold}"],
        )

        logger.debug(f"ChromaDB: inserted zone {doc_id}")
        return doc_id

    async def search_similar_zones(
        self,
        embedding: list[float],
        symbol: str,
        zone_type: str | None = None,
        k: int = 3,
    ) -> list[ZoneHit]:
        if not self._zone_coll:
            return []

        where_filter: dict[str, Any] = {"symbol": symbol}
        if zone_type:
            where_filter["zone_type"] = zone_type

        try:
            results = self._zone_coll.query(
                query_embeddings=[embedding],
                n_results=k,
                where=where_filter,
                include=["metadatas", "distances"],
            )
        except Exception:
            results = self._zone_coll.query(
                query_embeddings=[embedding],
                n_results=k,
                include=["metadatas", "distances"],
            )

        hits: list[ZoneHit] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return hits

        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i] if results.get("distances") else 1.0
            cosine_sim = 1.0 - distance

            zone = self._meta_to_zone(meta)
            days_since = self._calc_days_since(meta.get("formed_time", 0))
            re_rank = self._rerank(cosine_sim, days_since)

            hits.append(ZoneHit(
                zone=zone,
                cosine_sim=cosine_sim,
                days_since=days_since,
                re_rank_score=re_rank,
                point_id=str(doc_id),
            ))

        hits.sort(key=lambda h: h.re_rank_score, reverse=True)
        return hits[:k]

    # =========================================================================
    # Internal Helpers
    # =========================================================================
    def _debate_to_text(self, debate: DebateRecord) -> str:
        """Chuyen debate thanh text document."""
        parts = [
            f"Debate for {debate.symbol}",
            f"Macro regime: {debate.macro_regime}",
            f"Session: {debate.session}",
        ]
        if debate.bull:
            parts.append(f"Bull: {debate.bull.target_price}")
        if debate.bear:
            parts.append(f"Bear: {debate.bear.target_price}")
        if debate.consensus:
            parts.append(
                f"Consensus: rating={debate.consensus.rating}, "
                f"direction={debate.consensus.preferred_direction}"
            )
        return " | ".join(parts)

    def _meta_to_debate(self, meta: dict) -> DebateRecord:
        """Chuyen metadata thanh DebateRecord."""
        from core.memory.models import BullThesis, BearThesis, ConsensusResult

        return DebateRecord(
            symbol=meta.get("symbol", ""),
            bar_close_time=int(meta.get("bar_close_time", 0)),
            macro_regime=meta.get("macro_regime", "NORMAL"),
            session=meta.get("session", "ASIAN"),
            archived=bool(meta.get("archived", False)),
            precedents_count=int(meta.get("precedents_count", 0)),
            bull=BullThesis(
                direction="BULLISH",
                confidence=0.5,
                target_price=0.0,
                invalidation_price=0.0,
            ),
            bear=BearThesis(
                direction="BEARISH",
                confidence=0.5,
                target_price=0.0,
                invalidation_price=0.0,
            ),
            consensus=ConsensusResult(
                rating=int(meta.get("rating", 0)),
                preferred_direction=meta.get("direction", "NEUTRAL"),
            ),
        )

    def _meta_to_zone(self, meta: dict) -> Zone:
        """Chuyen metadata thanh Zone."""
        from core.memory.models.enums import Timeframe

        return Zone(
            id=meta.get("id", meta.get("zone_id", "")),
            symbol=meta.get("symbol", ""),
            timeframe=Timeframe(meta.get("timeframe", "M1")),
            zone_type=ZoneType(meta.get("zone_type", "OB_BULL")),
            formed_time=int(meta.get("formed_time", 0)),
            p_hold=float(meta.get("p_hold", 0.0)),
            w_zone=float(meta.get("w_zone", 1.0)),
        )

    @staticmethod
    def _calc_days_since(unix_ms: int) -> int:
        """Tinh so ngay tu unix ms."""
        if not unix_ms:
            return 999
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        then = datetime.fromtimestamp(unix_ms / 1000, tz=timezone.utc)
        return max(0, (now - then).days)

    @staticmethod
    def _rerank(cosine_sim: float, days_since: int) -> float:
        """
        Re-ranking: 0.7 * cosine_sim + 0.3 * recency_weight
        recency_weight = 1 / (1 + days_since)
        """
        recency = 1.0 / (1.0 + days_since)
        return 0.7 * cosine_sim + 0.3 * recency


# =============================================================================
# Qdrant Provider
# =============================================================================
class QdrantProvider(BaseVectorDB):
    """
    Qdrant implementation.
    Requires Qdrant server running (default: http://127.0.0.1:6333).
    """

    def __init__(self, config: VectorDBConfig) -> None:
        self._config = config
        self._client: qdrant_client.QdrantClient | None = None

    async def connect(self) -> None:
        from qdrant_client import QdrantClient

        self._client = QdrantClient(
            url=self._config.url,
            timeout=10.0,
        )
        await self.ensure_collections()
        logger.info(f"Qdrant connected: {self._config.url}")

    async def close(self) -> None:
        self._client = None
        logger.info("Qdrant disconnected")

    async def ensure_collections(self) -> None:
        if not self._client:
            return

        from qdrant_client.models import Distance, VectorParams
        from qdrant_client.http import models

        vector_size = 256  # e_USV projection dimension

        for coll_name in [self._config.collection_debate, self._config.collection_zones]:
            collections = self._client.get_collections().collections
            coll_names = [c.name for c in collections]
            if coll_name not in coll_names:
                self._client.create_collection(
                    collection_name=coll_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        m=16 if coll_name == self._config.collection_debate else 8,
                        ef_construct=200 if coll_name == self._config.collection_debate else 100,
                    ),
                )
                logger.info(f"Qdrant: created collection {coll_name}")

    async def insert_debate(
        self,
        debate: DebateRecord,
        embedding: list[float],
    ) -> str:
        if not self._client:
            raise RuntimeError("Qdrant not connected")

        from qdrant_client.models import PointStruct

        point_id = f"{debate.symbol}_{debate.bar_close_time}"

        payload = {
            "symbol": debate.symbol,
            "bar_close_time": debate.bar_close_time,
            "macro_regime": debate.macro_regime,
            "session": debate.session,
            "rating": debate.consensus.rating if debate.consensus else 0,
            "direction": debate.consensus.preferred_direction if debate.consensus else "NEUTRAL",
            "outcome": getattr(debate, "outcome", "") or "",
            "archived": debate.archived,
            "precedents_count": debate.precedents_count,
            "bull_json": json.dumps(debate.bull.to_dict() if hasattr(debate.bull, "to_dict") else {}),
            "bear_json": json.dumps(debate.bear.to_dict() if hasattr(debate.bear, "to_dict") else {}),
            "consensus_json": json.dumps(debate.consensus.to_dict() if hasattr(debate.consensus, "to_dict") else {}),
        }

        self._client.upsert(
            collection_name=self._config.collection_debate,
            points=[PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )],
        )

        logger.debug(f"Qdrant: upserted debate {point_id}")
        return point_id

    async def search_similar_debates(
        self,
        embedding: list[float],
        symbol: str,
        macro_regime: str | None = None,
        k: int = 3,
    ) -> list[DebateHit]:
        if not self._client:
            return []

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        search_filter = Filter(
            must=[
                FieldCondition(
                    key="symbol",
                    match=MatchValue(value=symbol),
                ),
            ]
        )
        if macro_regime:
            search_filter.must.append(
                FieldCondition(
                    key="macro_regime",
                    match=MatchValue(value=macro_regime),
                )
            )

        try:
            results = self._client.search(
                collection_name=self._config.collection_debate,
                query_vector=embedding,
                query_filter=search_filter,
                limit=k,
                with_payload=True,
                score_threshold=0.0,
            )
        except Exception as e:
            logger.warning(f"Qdrant search failed: {e}, trying without filter")
            results = self._client.search(
                collection_name=self._config.collection_debate,
                query_vector=embedding,
                limit=k * 3,
                with_payload=True,
                score_threshold=0.0,
            )

        hits: list[DebateHit] = []
        for result in results:
            meta = result.payload or {}
            cosine_sim = float(result.score)

            debate = self._qdrant_payload_to_debate(meta)
            days_since = ChromaDBProvider._calc_days_since(meta.get("bar_close_time", 0))
            re_rank = ChromaDBProvider._rerank(cosine_sim, days_since)

            hits.append(DebateHit(
                debate_record=debate,
                cosine_sim=cosine_sim,
                days_since=days_since,
                re_rank_score=re_rank,
                point_id=str(result.id),
            ))

        hits.sort(key=lambda h: h.re_rank_score, reverse=True)
        return hits[:k]

    async def insert_zone_embedding(
        self,
        zone: Zone,
        embedding: list[float],
    ) -> str:
        if not self._client:
            raise RuntimeError("Qdrant not connected")

        from qdrant_client.models import PointStruct

        payload = {
            "symbol": zone.symbol,
            "timeframe": zone.timeframe.value if hasattr(zone.timeframe, "value") else str(zone.timeframe),
            "zone_type": zone.zone_type.value if hasattr(zone.zone_type, "value") else str(zone.zone_type),
            "outcome": getattr(zone, "outcome", "") or "",
            "formed_time": zone.formed_time,
            "p_hold": zone.p_hold,
            "w_zone": zone.w_zone,
            "top": zone.top,
            "bottom": zone.bottom,
        }

        self._client.upsert(
            collection_name=self._config.collection_zones,
            points=[PointStruct(
                id=zone.id,
                vector=embedding,
                payload=payload,
            )],
        )

        logger.debug(f"Qdrant: upserted zone {zone.id}")
        return zone.id

    async def search_similar_zones(
        self,
        embedding: list[float],
        symbol: str,
        zone_type: str | None = None,
        k: int = 3,
    ) -> list[ZoneHit]:
        if not self._client:
            return []

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        search_filter = Filter(
            must=[
                FieldCondition(
                    key="symbol",
                    match=MatchValue(value=symbol),
                ),
            ]
        )
        if zone_type:
            search_filter.must.append(
                FieldCondition(
                    key="zone_type",
                    match=MatchValue(value=zone_type),
                )
            )

        try:
            results = self._client.search(
                collection_name=self._config.collection_zones,
                query_vector=embedding,
                query_filter=search_filter,
                limit=k,
                with_payload=True,
                score_threshold=0.0,
            )
        except Exception as e:
            logger.warning(f"Qdrant zone search failed: {e}")
            return []

        hits: list[ZoneHit] = []
        for result in results:
            meta = result.payload or {}
            cosine_sim = float(result.score)

            zone = self._qdrant_payload_to_zone(meta)
            days_since = ChromaDBProvider._calc_days_since(meta.get("formed_time", 0))
            re_rank = ChromaDBProvider._rerank(cosine_sim, days_since)

            hits.append(ZoneHit(
                zone=zone,
                cosine_sim=cosine_sim,
                days_since=days_since,
                re_rank_score=re_rank,
                point_id=str(result.id),
            ))

        hits.sort(key=lambda h: h.re_rank_score, reverse=True)
        return hits[:k]

    def _qdrant_payload_to_debate(self, payload: dict) -> DebateRecord:
        from core.memory.models import BullThesis, BearThesis, ConsensusResult

        bull_dict = json.loads(payload.get("bull_json", "{}"))
        bear_dict = json.loads(payload.get("bear_json", "{}"))
        consensus_dict = json.loads(payload.get("consensus_json", "{}"))

        return DebateRecord(
            symbol=payload.get("symbol", ""),
            bar_close_time=int(payload.get("bar_close_time", 0)),
            macro_regime=payload.get("macro_regime", "NORMAL"),
            session=payload.get("session", "ASIAN"),
            archived=bool(payload.get("archived", False)),
            precedents_count=int(payload.get("precedents_count", 0)),
            bull=BullThesis.from_dict(bull_dict) if bull_dict else BullThesis(),
            bear=BearThesis.from_dict(bear_dict) if bear_dict else BearThesis(),
            consensus=ConsensusResult.from_dict(consensus_dict) if consensus_dict else ConsensusResult(),
        )

    def _qdrant_payload_to_zone(self, payload: dict) -> Zone:
        from core.memory.models.enums import Timeframe

        return Zone(
            id=payload.get("zone_id", ""),
            symbol=payload.get("symbol", ""),
            timeframe=Timeframe(payload.get("timeframe", "M1")),
            zone_type=ZoneType(payload.get("zone_type", "OB_BULL")),
            formed_time=int(payload.get("formed_time", 0)),
            p_hold=float(payload.get("p_hold", 0.0)),
            w_zone=float(payload.get("w_zone", 1.0)),
            top=float(payload.get("top", 0.0)),
            bottom=float(payload.get("bottom", 0.0)),
        )


# =============================================================================
# VectorDB Factory
# =============================================================================
class VectorDBFactory:
    """Factory tao VectorDB provider dua tren config."""

    @staticmethod
    def create(config: VectorDBConfig | None = None) -> BaseVectorDB:
        """
        Tao VectorDB provider dua tren config.
        Default: ChromaDB (file-based, khong can server).
        """
        if config is None:
            config = VectorDBConfig()

        provider = config.provider.lower()
        if provider == "qdrant":
            return QdrantProvider(config)
        elif provider == "chromadb":
            return ChromaDBProvider(config)
        else:
            logger.warning(f"Unknown VectorDB provider '{provider}', defaulting to ChromaDB")
            return ChromaDBProvider(config)

    @staticmethod
    async def create_and_connect(config: VectorDBConfig | None = None) -> BaseVectorDB:
        """Tao va ket noi VectorDB provider."""
        db = VectorDBFactory.create(config)
        await db.connect()
        return db
