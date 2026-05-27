# =============================================================================
# AGENTIC-QUANT — Rule-based Fallback cho Multi-Agent Debate
# Fallback khi MODEL_DEGRADED=True hoac LLM call that bai
# =============================================================================
"""
Rule-based Fallback su dung khi:
1. MODEL_DEGRADED=True (hieu suat model giam).
2. LLM API call that bai (timeout, error).
3. Budget exceeded (khong muon ton them chi phi).

Thay vi goi LLM, su dung cac rules deterministic:
- So sanh BSL/SSL probability tu Model A.
- Kiem tra premium/discount zone.
- Kiem tra FVG/OB zone signals.
- Kiem tra macro context (news, regime).

Output: ConsensusResult voi debate_used_fallback=True.

Usage::

    fallback = RuleFallback()
    consensus = await fallback.evaluate(macro_context={...})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from .bull_agent import BullResult
from .bear_agent import BearResult
from .critic_agent import ConsensusResult


# =============================================================================
# Constants
# =============================================================================
# Weights cho tung tin hieu
W_BSL_PROB = 0.30  # Model A BSL probability
W_SSL_PROB = 0.30  # Model A SSL probability
W_DISCOUNT = 0.15  # Discount zone = bullish
W_PREMIUM = 0.15   # Premium zone = bearish
W_FVG_BULL = 0.10  # FVG Bullish chua mitigate
W_FVG_BEAR = 0.10  # FVG Bearish chua mitigate
W_MACRO = 0.05     # Macro regime

# Thresholds
THRESHOLD_BULL = 0.40  # > 0.4 -> BULLISH
THRESHOLD_BEAR = -0.40  # < -0.4 -> BEARISH


# =============================================================================
# Rule-based Score
# =============================================================================
@dataclass
class RuleScore:
    """
    Score tu rule-based analysis.

    Attributes:
        total_score: Score tong (-1.0 to +1.0, + = bullish).
        components: Chi tiet tung component score.
        rating: Rating cuoi cung (-4 to +4).
        direction: Huong giao dich.
        confidence_qualifier: Muc do tin cay.
    """

    total_score: float = 0.0  # -1 to +1
    components: dict[str, float] = field(default_factory=dict)
    rating: int = 0
    direction: str = "NEUTRAL"
    confidence_qualifier: str = "LOW"

    def to_dict(self) -> dict:
        return {
            "total_score": self.total_score,
            "components": self.components,
            "rating": self.rating,
            "direction": self.direction,
            "confidence_qualifier": self.confidence_qualifier,
        }


# =============================================================================
# Rule Fallback
# =============================================================================
class RuleFallback:
    """
    Rule-based fallback khi LLM unavailable.

    Phan tich deterministic dua tren:
    - Model A probabilities (BSL/SSL)
    - Price position (premium/discount)
    - Active zones (FVG/OB)
    - Macro context (news, regime)

    Usage::

        fallback = RuleFallback()
        consensus = await fallback.evaluate(macro_context={...})
    """

    def __init__(self) -> None:
        logger.debug("RuleFallback: initialized")

    # =========================================================================
    # Main Evaluation
    # =========================================================================
    async def evaluate(
        self,
        prediction_features: dict[str, Any] | None = None,
        macro_context: dict[str, Any] | None = None,
    ) -> ConsensusResult:
        """
        Evaluate rules va tra ve ConsensusResult.

        Args:
            prediction_features: Dict chua model outputs, features.
            macro_context: Dict chua macro regime, news info.

        Returns:
            ConsensusResult with debate_used_fallback=True.
        """
        logger.debug("RuleFallback: evaluating")

        features = prediction_features or {}
        macro = macro_context or {}

        # === Compute component scores ===
        components: dict[str, float] = {}

        # 1. BSL probability
        p_bsl = features.get("p_bsl", 0.5)
        p_ssl = features.get("p_ssl", 0.3)
        p_lat = features.get("p_lateral", 0.2)
        bsl_edge = min(1.0, max(-1.0, (p_bsl - p_ssl) * 2))
        components["model_a_bsl_edge"] = round(bsl_edge * W_BSL_PROB, 4)

        # 2. Premium / Discount
        current_price = features.get("current_price", 0.0)
        equilibrium = features.get("equilibrium", 0.0)
        if current_price > 0 and equilibrium > 0:
            pd_ratio = (current_price - equilibrium) / equilibrium * 100
            if pd_ratio > 0:
                # Premium zone -> bearish
                pd_score = min(1.0, pd_ratio / 5.0) * W_PREMIUM * -1
            else:
                # Discount zone -> bullish
                pd_score = min(1.0, abs(pd_ratio) / 5.0) * W_DISCOUNT
            components["premium_discount"] = round(pd_score, 4)
        else:
            components["premium_discount"] = 0.0

        # 3. Active zones
        zones = features.get("zones_created", []) or []
        bull_zones = len([z for z in zones if "BULL" in z.get("zone_type", "") and z.get("status") != "MITIGATED"])
        bear_zones = len([z for z in zones if "BEAR" in z.get("zone_type", "") and z.get("status") != "MITIGATED"])

        zone_score = min(1.0, bull_zones * 0.3) * W_FVG_BULL - min(1.0, bear_zones * 0.3) * W_FVG_BEAR
        components["active_zones"] = round(zone_score, 4)

        # 4. Macro regime
        regime = macro.get("regime", macro.get("macro_regime", "NEUTRAL"))
        regime_map = {
            "BULLISH": 0.5,
            "BULL": 0.5,
            "BEARISH": -0.5,
            "BEAR": -0.5,
            "NEUTRAL": 0.0,
            "TRENDING_UP": 0.3,
            "TRENDING_DOWN": -0.3,
            "RANGING": 0.0,
            "VOLATILE": 0.0,
        }
        regime_score = regime_map.get(regime.upper(), 0.0) * W_MACRO
        components["macro_regime"] = round(regime_score, 4)

        # 5. News guardrail / surprise
        if macro.get("MAJOR_SURPRISE_FLAG", False):
            surprise_dir = macro.get("surprise_direction", "NEUTRAL")
            if surprise_dir == "POSITIVE":
                components["news_surprise"] = round(0.1 * W_MACRO, 4)
            elif surprise_dir == "NEGATIVE":
                components["news_surprise"] = round(-0.1 * W_MACRO, 4)
            else:
                components["news_surprise"] = 0.0
        else:
            components["news_surprise"] = 0.0

        # === Total score ===
        total = sum(components.values())
        total = max(-1.0, min(1.0, total))

        # === Convert to rating ===
        rating = self._score_to_rating(total)
        direction = self._rating_to_direction(rating)
        confidence = self._get_confidence(total)

        logger.debug(
            f"RuleFallback: score={total:.4f}, rating={rating}, "
            f"direction={direction}, components={components}"
        )

        return ConsensusResult(
            rating=rating,
            direction=direction,
            confidence_qualifier=confidence,
            agreement_score=0.5,  # Default middle agreement
            bull_thesis=self._build_thesis(total > 0, features, macro, components),
            bear_thesis=self._build_thesis(total <= 0, features, macro, components),
            reasoning=self._build_reasoning(total, rating, components),
            bull_evidence=[],  # Rule-based khong co evidence citations
            bear_evidence=[],
            debate_used_fallback=True,
            success=True,
        )

    # =========================================================================
    # Score Conversion Helpers
    # =========================================================================
    @staticmethod
    def _score_to_rating(score: float) -> int:
        """
        Convert score [-1, 1] sang rating [-4, 4].

        Mapping:
            [-1.0, -0.8] -> -4 (max bear)
            [-0.8, -0.6] -> -3
            [-0.6, -0.4] -> -2
            [-0.4, -0.15] -> -1
            [-0.15, 0.15] -> 0 (neutral)
            [0.15, 0.4] -> +1
            [0.4, 0.6] -> +2
            [0.6, 0.8] -> +3
            [0.8, 1.0] -> +4 (max bull)
        """
        if score >= 0.8:
            return 4
        if score >= 0.6:
            return 3
        if score >= 0.4:
            return 2
        if score >= 0.15:
            return 1
        if score > -0.15:
            return 0
        if score >= -0.4:
            return -1
        if score >= -0.6:
            return -2
        if score >= -0.8:
            return -3
        return -4

    @staticmethod
    def _rating_to_direction(rating: int) -> str:
        """Convert rating to direction."""
        if rating >= 2:
            return "BULL"
        if rating <= -2:
            return "BEAR"
        return "NEUTRAL"

    @staticmethod
    def _get_confidence(score: float) -> str:
        """Xac dinh confidence qualifier tu score."""
        abs_score = abs(score)
        if abs_score >= 0.5:
            return "MEDIUM"
        if abs_score >= 0.2:
            return "LOW"
        return "LOW"

    # =========================================================================
    # Thesis / Reasoning Builders
    # =========================================================================
    def _build_thesis(
        self,
        is_bull: bool,
        features: dict[str, Any],
        macro: dict[str, Any],
        components: dict[str, float],
    ) -> str:
        """Build brief thesis statement."""
        parts = []
        side = "BULLISH" if is_bull else "BEARISH"

        if is_bull:
            pd = components.get("premium_discount", 0)
            if pd > 0:
                parts.append("Price in DISCOUNT zone")
            if components.get("model_a_bsl_edge", 0) > 0:
                parts.append("Model A favors BSL")
            if components.get("active_zones", 0) > 0:
                parts.append("Bullish zones active")
        else:
            pd = components.get("premium_discount", 0)
            if pd < 0:
                parts.append("Price in PREMIUM zone")
            if components.get("model_a_bsl_edge", 0) < 0:
                parts.append("Model A favors SSL")
            if components.get("active_zones", 0) < 0:
                parts.append("Bearish zones active")

        if not parts:
            parts.append("No strong signals detected")

        return f"Rule-based {side}: {' + '.join(parts)}"

    def _build_reasoning(
        self,
        total: float,
        rating: int,
        components: dict[str, float],
    ) -> str:
        """Build detailed reasoning string."""
        lines = [
            f"Rule-based fallback consensus (DEBUG mode).",
            f"Total Score: {total:.4f} | Rating: {rating:+d}",
            "",
            "Component Scores:",
        ]
        for name, score in sorted(components.items(), key=lambda x: abs(x[1]), reverse=True):
            lines.append(f"  {name}: {score:+.4f}")

        lines.append("")
        lines.append(
            "Note: This is a rule-based estimate. "
            "No AI LLM was used for this decision."
        )

        return "\n".join(lines)
