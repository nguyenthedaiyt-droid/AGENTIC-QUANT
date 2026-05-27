# =============================================================================
# AGENTIC-QUANT — Prediction Data Models
# Mirror tu TypeScript source of truth: ui/src/types/index.ts
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .enums import ConfidenceQualifier, PredictionOutcome, Timeframe

if TYPE_CHECKING:
    pass


# =============================================================================
# ZonePrediction: Ket qua predict zone tu Model B
# =============================================================================
@dataclass
class ZonePrediction:
    """
    Ket qua predict zone tu Model B (zone hold probability).
    Moi prediction co p_hold cho mot zone cu the.
    """

    zone_id: str = ""
    zone_type: str = ""
    p_hold: float = 0.0  # Xac suất zone duoc giu sau adjustment
    p_hold_pre_adj: float = 0.0  # Xac suất truoc khi adjustment
    zone_top: float = 0.0
    zone_bottom: float = 0.0
    zone_ce: float = 0.0
    threshold_used: float = 0.0

    def to_dict(self) -> dict:
        return {
            "zone_id": self.zone_id,
            "zone_type": self.zone_type,
            "p_hold": self.p_hold,
            "p_hold_pre_adj": self.p_hold_pre_adj,
            "zone_top": self.zone_top,
            "zone_bottom": self.zone_bottom,
            "zone_ce": self.zone_ce,
            "threshold_used": self.threshold_used,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ZonePrediction:
        return cls(
            zone_id=d["zone_id"],
            zone_type=d["zone_type"],
            p_hold=float(d["p_hold"]),
            p_hold_pre_adj=float(d["p_hold_pre_adj"]),
            zone_top=float(d["zone_top"]),
            zone_bottom=float(d["zone_bottom"]),
            zone_ce=float(d["zone_ce"]),
            threshold_used=float(d["threshold_used"]),
        )


# =============================================================================
# ModelAPrediction: Ket qua tu XGBoost Model A
# =============================================================================
@dataclass
class ModelAPrediction:
    """
    Ket qua prediction tu XGBoost Model A.
    Bao gom p_bsl, p_ssl, p_lateral va muc tieu gia.
    """

    symbol: str = ""
    bar_close_time: int = 0  # Unix ms

    p_bsl: float = 0.0  # Xac suất Buy-Side Liquidity se bi hit
    p_ssl: float = 0.0  # Xac suất Sell-Side Liquidity se bi hit
    p_lateral: float = 0.0  # Xac suất thi truong di ngang

    predicted_bsl_level: float = 0.0  # Muc gia BSL du doan
    predicted_ssl_level: float = 0.0  # Muc gia SSL du doan

    bsl_tf: Timeframe = Timeframe.M1
    ssl_tf: Timeframe = Timeframe.M1

    confidence_qualifier: ConfidenceQualifier = ConfidenceQualifier.MEDIUM
    model_version: str = ""
    inference_latency_ms: float = 0.0

    # Ket qua (duoc set sau boi OutcomeDeterminator)
    outcome: PredictionOutcome | None = None
    outcome_time: int | None = None  # Unix ms

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "bar_close_time": self.bar_close_time,
            "p_bsl": self.p_bsl,
            "p_ssl": self.p_ssl,
            "p_lateral": self.p_lateral,
            "predicted_bsl_level": self.predicted_bsl_level,
            "predicted_ssl_level": self.predicted_ssl_level,
            "bsl_tf": self.bsl_tf.value if isinstance(self.bsl_tf, Timeframe) else self.bsl_tf,
            "ssl_tf": self.ssl_tf.value if isinstance(self.ssl_tf, Timeframe) else self.ssl_tf,
            "confidence_qualifier": (
                self.confidence_qualifier.value
                if isinstance(self.confidence_qualifier, ConfidenceQualifier)
                else self.confidence_qualifier
            ),
            "model_version": self.model_version,
            "inference_latency_ms": self.inference_latency_ms,
            "outcome": self.outcome.value if self.outcome else None,
            "outcome_time": self.outcome_time,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ModelAPrediction:
        def _float(v):
            if isinstance(v, (int, float)):
                return float(v)
            try:
                return float(str(v))
            except (ValueError, TypeError):
                return 0.0

        def _int(v):
            if isinstance(v, int):
                return v
            try:
                return int(str(v))
            except (ValueError, TypeError):
                return 0

        return cls(
            symbol=str(d.get("symbol", "")),
            bar_close_time=_int(d.get("bar_close_time")),
            p_bsl=_float(d.get("p_bsl")),
            p_ssl=_float(d.get("p_ssl")),
            p_lateral=_float(d.get("p_lateral")),
            predicted_bsl_level=_float(d.get("predicted_bsl_level")),
            predicted_ssl_level=_float(d.get("predicted_ssl_level")),
            bsl_tf=Timeframe(str(d.get("bsl_tf", "M1"))),
            ssl_tf=Timeframe(str(d.get("ssl_tf", "M1"))),
            confidence_qualifier=ConfidenceQualifier(str(d.get("confidence_qualifier", "MEDIUM"))),
            model_version=str(d.get("model_version", "")),
            inference_latency_ms=_float(d.get("inference_latency_ms")),
            outcome=PredictionOutcome(d["outcome"]) if d.get("outcome") and str(d["outcome"]) not in ("None", "null", "") else None,
            outcome_time=_int(d["outcome_time"]) if d.get("outcome_time") and str(d["outcome_time"]) not in ("None", "null", "") else None,
        )

    def probabilities_sum(self) -> float:
        """Tong cua 3 xac suất (nen = 1.0)."""
        return self.p_bsl + self.p_ssl + self.p_lateral

    def dominant_direction(self) -> str:
        """Huong chinh cua prediction (BSL, SSL, hoac LATERAL)."""
        if self.p_bsl >= self.p_ssl and self.p_bsl >= self.p_lateral:
            return "BSL"
        elif self.p_ssl >= self.p_lateral:
            return "SSL"
        return "LATERAL"


# =============================================================================
# DebateEvidence: Bang chung trong mot thesis
# =============================================================================
@dataclass
class DebateEvidence:
    """Mot minh chung trong debate thesis."""

    text: str = ""
    source: str = ""
    weight: float = 0.0


# =============================================================================
# BullThesis / BearThesis: Hai phia cua debate
# =============================================================================
@dataclass
class BullThesis:
    """Thesis dau tu mua - evidence va muc tieu."""

    direction: str = "BULLISH"
    confidence: float = 0.0
    target_price: float = 0.0
    invalidation_price: float = 0.0
    evidence: list[DebateEvidence] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "confidence": self.confidence,
            "target_price": self.target_price,
            "invalidation_price": self.invalidation_price,
            "evidence": [
                {"text": e.text, "source": e.source, "weight": e.weight}
                for e in self.evidence
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> BullThesis:
        return cls(
            direction=d.get("direction", "BULLISH"),
            confidence=float(d.get("confidence", 0.0)),
            target_price=float(d.get("target_price", 0.0)),
            invalidation_price=float(d.get("invalidation_price", 0.0)),
            evidence=[
                DebateEvidence(text=e["text"], source=e["source"], weight=float(e["weight"]))
                for e in d.get("evidence", [])
            ],
        )


@dataclass
class BearThesis:
    """Thesis dau tu ban - evidence va muc tieu."""

    direction: str = "BEARISH"
    confidence: float = 0.0
    target_price: float = 0.0
    invalidation_price: float = 0.0
    evidence: list[DebateEvidence] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "confidence": self.confidence,
            "target_price": self.target_price,
            "invalidation_price": self.invalidation_price,
            "evidence": [
                {"text": e.text, "source": e.source, "weight": e.weight}
                for e in self.evidence
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> BearThesis:
        return cls(
            direction=d.get("direction", "BEARISH"),
            confidence=float(d.get("confidence", 0.0)),
            target_price=float(d.get("target_price", 0.0)),
            invalidation_price=float(d.get("invalidation_price", 0.0)),
            evidence=[
                DebateEvidence(text=e["text"], source=e["source"], weight=float(e["weight"]))
                for e in d.get("evidence", [])
            ],
        )


# =============================================================================
# ConsensusResult: Ket qua dong y cua debate
# =============================================================================
@dataclass
class ConsensusResult:
    """Ket qua consensus tu Multi-Agent Debate."""

    rating: int = 0  # [-4, +4]
    preferred_direction: str = "NEUTRAL"  # "BULLISH" | "BEARISH" | "NEUTRAL"
    conviction_zone_price: float | None = None
    reasoning: str = ""
    agreement_score: float = 0.0  # [0, 1]
    confidence_qualifier: ConfidenceQualifier = ConfidenceQualifier.MEDIUM
    is_fallback: bool = False

    def to_dict(self) -> dict:
        return {
            "rating": self.rating,
            "preferred_direction": self.preferred_direction,
            "conviction_zone_price": self.conviction_zone_price,
            "reasoning": self.reasoning,
            "agreement_score": self.agreement_score,
            "confidence_qualifier": (
                self.confidence_qualifier.value
                if isinstance(self.confidence_qualifier, ConfidenceQualifier)
                else self.confidence_qualifier
            ),
            "is_fallback": self.is_fallback,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ConsensusResult:
        return cls(
            rating=int(d.get("rating", 0)),
            preferred_direction=d.get("preferred_direction", "NEUTRAL"),
            conviction_zone_price=float(d["conviction_zone_price"]) if d.get("conviction_zone_price") else None,
            reasoning=d.get("reasoning", ""),
            agreement_score=float(d.get("agreement_score", 0.0)),
            confidence_qualifier=ConfidenceQualifier(d.get("confidence_qualifier", "MEDIUM")),
            is_fallback=bool(d.get("is_fallback", False)),
        )


# =============================================================================
# DebateRecord: Ban ghi debate day du
# =============================================================================
@dataclass
class DebateRecord:
    """
    Ban ghi debate day du, duoc luu vao VectorDB.
    Duoc tao moi tai moi BAR_CLOSE(M1) event.
    """

    symbol: str = ""
    bar_close_time: int = 0  # Unix ms
    bull: BullThesis = field(default_factory=BullThesis)
    bear: BearThesis = field(default_factory=BearThesis)
    consensus: ConsensusResult = field(default_factory=ConsensusResult)
    precedents_count: int = 0  # So precedent tu RAG retrieval
    latency_ms: float = 0.0  # Tong latency debate

    # e_USV vector cho VectorDB (tinh toan boi DebateArchiver)
    embedding: list[float] | None = None

    # Metadata cho retrieval
    archived: bool = False
    session: str = "ASIAN"
    macro_regime: str = "NORMAL"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "bar_close_time": self.bar_close_time,
            "bull": self.bull.to_dict(),
            "bear": self.bear.to_dict(),
            "consensus": self.consensus.to_dict(),
            "precedents_count": self.precedents_count,
            "latency_ms": self.latency_ms,
            "archived": self.archived,
            "session": self.session,
            "macro_regime": self.macro_regime,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DebateRecord:
        bull_d = d.get("bull", {})
        bear_d = d.get("bear", {})
        consensus_d = d.get("consensus", {})

        bull = BullThesis.from_dict(bull_d) if isinstance(bull_d, dict) else BullThesis()
        bear = BearThesis.from_dict(bear_d) if isinstance(bear_d, dict) else BearThesis()
        consensus = ConsensusResult.from_dict(consensus_d) if isinstance(consensus_d, dict) else ConsensusResult()

        return cls(
            symbol=d.get("symbol", ""),
            bar_close_time=int(d.get("bar_close_time", 0)),
            bull=bull,
            bear=bear,
            consensus=consensus,
            precedents_count=int(d.get("precedents_count", 0)),
            latency_ms=float(d.get("latency_ms", 0.0)),
            archived=bool(d.get("archived", False)),
            session=d.get("session", "ASIAN"),
            macro_regime=d.get("macro_regime", "NORMAL"),
        )


# =============================================================================
# FeatureVector: Feature vector cache cho Model A/B inference
# =============================================================================
@dataclass
class FeatureVector:
    """Feature vector duoc cache trong Redis."""

    symbol: str = ""
    bar_close_time: int = 0  # Unix ms
    model_name: str = ""  # "model_a" | "model_b"
    features: list[float] = field(default_factory=list)
    created_at: int = 0  # Unix ms

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "bar_close_time": self.bar_close_time,
            "model_name": self.model_name,
            "features": self.features,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FeatureVector:
        return cls(
            symbol=d["symbol"],
            bar_close_time=int(d["bar_close_time"]),
            model_name=d["model_name"],
            features=[float(x) for x in d["features"]],
            created_at=int(d["created_at"]),
        )
