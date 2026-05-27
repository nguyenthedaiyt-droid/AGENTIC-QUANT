# =============================================================================
# AGENTIC-QUANT — Technical Brief Builder
# Xay dung context brief (max 8000 tokens) cho Multi-Agent Debate
# =============================================================================
"""
Technical Brief chua tat ca thong tin can thiet de cac agent tranh bien:
- HTF Structure: Cau truc thi truong Higher Timeframe (H1/H4/D1)
- Liquidity Target: Cac muc thanh khoan BSL/SSL
- Active Zones: FVG, OB, Order Blocks dang active
- Model Outputs: Ket qua tu XGBoost Model A & B
- Macro Context: Su kien kinh te, regime
- Precedents (RAG): Cac debate tuong tu trong qua khu

Luong token toi da: 8000 tokens (~32000 ky tu UTF-8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

# Tich hop RAG Retriever thuc te
try:
    from core.memory.long_term.rag_retriever import RAGRetriever, Precedent
except ImportError:
    RAGRetriever = None  # type: ignore
    Precedent = None  # type: ignore

if TYPE_CHECKING:
    from core.ai_engine.feature_engineering.pipeline import FeatureOutput


# =============================================================================
# Constants
# =============================================================================
MAX_TOKENS = 8000
MAX_CHARS = 32000  # ~4 ky tu / token
TOKEN_SAFETY_MARGIN = 0.85  # Chi dung 85% de tranh overflow
SAFE_CHARS = int(MAX_CHARS * TOKEN_SAFETY_MARGIN)  # ~27200


# =============================================================================
# Technical Brief Data
# =============================================================================
@dataclass
class TechnicalBriefData:
    """
    Du lieu Technical Brief sau khi build.
    """

    symbol: str = ""
    timeframe: str = ""
    bar_close_time: int = 0
    current_price: float = 0.0
    brief_text: str = ""  # Text final (max SAFE_CHARS ky tu)
    token_estimate: int = 0  # Uoc luong token
    truncated: bool = False  # Co bi cat bot khong
    sections_built: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "bar_close_time": self.bar_close_time,
            "current_price": self.current_price,
            "brief_text": self.brief_text,
            "token_estimate": self.token_estimate,
            "truncated": self.truncated,
            "sections_built": self.sections_built,
        }


# =============================================================================
# Technical Brief Builder
# =============================================================================
class TechnicalBrief:
    """
    Xay dung Technical Brief tu FeatureOutput + macro context + precedents.

    Flow:
    1. Nhan FeatureOutput, macro context, precedents tu RAG.
    2. Ghep tung section, kiem tra token limit.
    3. Cat bot section Precedents truoc neu vuot qua.
    4. Tra ve TechnicalBriefData.

    Usage::

        brief = TechnicalBrief()
        result = await brief.build(
            features=features,
            symbol="XAUUSD",
            timeframe="H1",
            bar_close_time=1234567890,
            current_price=2500.0,
            macro_context={"regime": "BULLISH", "news_impact": "LOW"},
            precedents=[...],
        )
    """

    def __init__(self, max_safe_chars: int = SAFE_CHARS) -> None:
        self._max_safe_chars = max_safe_chars

    # =========================================================================
    # Main Build
    # =========================================================================
    async def build(
        self,
        features: dict[str, Any] | None,
        symbol: str,
        timeframe: str,
        bar_close_time: int,
        current_price: float,
        macro_context: dict[str, Any] | None = None,
        precedents: list[Any] | None = None,
    ) -> TechnicalBriefData:
        """
        Build Technical Brief tu tat ca du lieu dau vao.

        Args:
            features: FeatureOutput dictionary (hoac dict tuong thich).
            symbol: Ma symbol (VD: "XAUUSD").
            timeframe: Khung thoi gian (VD: "H1", "H4").
            bar_close_time: Unix timestamp (giay).
            current_price: Gia hien tai.
            macro_context: Dict chua macro regime, news impact, etc.
            precedents: List Precedent tu RAGRetriever.

        Returns:
            TechnicalBriefData da build.
        """
        logger.debug(
            f"TechnicalBrief: building for {symbol} {timeframe} @ {current_price}"
        )

        sections: list[str] = []
        sections_built: list[str] = []

        # === Section 1: HTF Structure ===
        sec1 = self._build_htf_structure(features, timeframe)
        sections.append(sec1)
        sections_built.append("HTF_STRUCTURE")

        # === Section 2: Liquidity Targets ===
        sec2 = self._build_liquidity_targets(features)
        sections.append(sec2)
        sections_built.append("LIQUIDITY_TARGETS")

        # === Section 3: Active Zones ===
        sec3 = self._build_active_zones(features)
        sections.append(sec3)
        sections_built.append("ACTIVE_ZONES")

        # === Section 4: Model Outputs ===
        sec4 = self._build_model_outputs(features)
        sections.append(sec4)
        sections_built.append("MODEL_OUTPUTS")

        # === Section 5: Macro Context ===
        sec5 = self._build_macro_context(macro_context)
        sections.append(sec5)
        sections_built.append("MACRO_CONTEXT")

        # === Section 6: Precedents (RAG) ===
        sec6 = self._build_precedents(precedents)
        sections.append(sec6)
        sections_built.append("PRECEDENTS")

        # === Ghep sections, dam bao token limit ===
        full_text, truncated = self._assemble(sections)

        # Uoc luong token
        token_est = len(full_text) // 4

        logger.debug(
            f"TechnicalBrief: done ({len(full_text)} chars, ~{token_est} tokens, "
            f"truncated={truncated})"
        )

        return TechnicalBriefData(
            symbol=symbol,
            timeframe=timeframe,
            bar_close_time=bar_close_time,
            current_price=current_price,
            brief_text=full_text,
            token_estimate=token_est,
            truncated=truncated,
            sections_built=sections_built,
        )

    # =========================================================================
    # Section Builders
    # =========================================================================
    def _build_htf_structure(
        self,
        features: dict[str, Any] | None,
        timeframe: str,
    ) -> str:
        """Xay dung section: HTF Structure.

        Bao gom: equilibrium, swing points IT/LT, cau truuc premium/discount.
        """
        lines: list[str] = [
            "=== HTF STRUCTURE ===",
            f"Timeframe: {timeframe}",
        ]

        if not features:
            lines.append("No feature data available.")
            return "\n".join(lines)

        # Equilibrium
        eq = features.get("equilibrium", None)
        if eq is not None:
            lines.append(f"Equilibrium: {eq:.2f}")
        else:
            lines.append("Equilibrium: N/A")

        # Premium / Discount
        current = features.get("current_price", 0.0)
        if current and eq:
            zone = "PREMIUM" if current > eq else "DISCOUNT" if current < eq else "AT_EQ"
            lines.append(f"Price Zone: {zone} (price={current:.2f})")
            pd_ratio = abs(current - eq) / (eq + 1e-9) * 100
            lines.append(f"Premium/Discount Ratio: {pd_ratio:.2f}%")

        # Structure map
        sm = features.get("structure_map")
        if sm and isinstance(sm, dict):
            lines.append(f"Structure: {sm.get('structure_type', 'N/A')}")
            lines.append(
                f"Swing Highs: {sm.get('swing_highs_count', 0)} | "
                f"Swing Lows: {sm.get('swing_lows_count', 0)}"
            )

        return "\n".join(lines)

    def _build_liquidity_targets(
        self,
        features: dict[str, Any] | None,
    ) -> str:
        """Xay dung section: Liquidity Targets.

        Bao gom: BSL (Buy-Side Liquidity), SSL (Sell-Side Liquidity),
        khoang cach den cac muc thanh khoan.
        """
        lines: list[str] = [
            "=== LIQUIDITY TARGETS ===",
        ]

        if not features:
            lines.append("No feature data available.")
            return "\n".join(lines)

        # BSL / SSL targets tu feature output
        bsl_targets = features.get("bsl_targets", [])
        ssl_targets = features.get("ssl_targets", [])
        current = features.get("current_price", 0.0)

        lines.append(
            f"BSL Targets ({len(bsl_targets)}): "
            + (
                ", ".join(f"{t:.2f}" for t in bsl_targets[:5])
                if bsl_targets
                else "None"
            )
        )
        lines.append(
            f"SSL Targets ({len(ssl_targets)}): "
            + (
                ", ".join(f"{t:.2f}" for t in ssl_targets[:5])
                if ssl_targets
                else "None"
            )
        )

        # Khoang cach
        if current > 0:
            nearest_bsl = min(bsl_targets, default=None)
            nearest_ssl = max(ssl_targets, default=None)
            if nearest_bsl:
                dist_bsl = abs(nearest_bsl - current) / current * 100
                lines.append(f"Nearest BSL: {nearest_bsl:.2f} ({dist_bsl:.2f}% away)")
            if nearest_ssl:
                dist_ssl = abs(nearest_ssl - current) / current * 100
                lines.append(f"Nearest SSL: {nearest_ssl:.2f} ({dist_ssl:.2f}% away)")

        return "\n".join(lines)

    def _build_active_zones(
        self,
        features: dict[str, Any] | None,
    ) -> str:
        """Xay dung section: Active Zones.

        Bao gom: FVG, OB, Order Blocks dang active va trang thai cua chung.
        """
        lines: list[str] = [
            "=== ACTIVE ZONES ===",
        ]

        if not features:
            lines.append("No feature data available.")
            return "\n".join(lines)

        zones = features.get("zones_created", []) or []
        if not zones:
            lines.append("No active zones.")
            return "\n".join(lines)

        # Loc zones chua bi mitigate
        active = [z for z in zones if z.get("status") != "MITIGATED"]

        bull_zones = [z for z in active if z.get("zone_type", "").endswith("BULL")]
        bear_zones = [z for z in active if z.get("zone_type", "").endswith("BEAR")]

        lines.append(f"Total Active Zones: {len(active)}")
        lines.append(f"  Bullish Zones (FVG/OB): {len(bull_zones)}")
        lines.append(f"  Bearish Zones (FVG/OB): {len(bear_zones)}")

        if bull_zones:
            lines.append("  Top Bullish Zones:")
            for z in bull_zones[:3]:
                lines.append(
                    f"    - {z.get('zone_id', '?')} top={z.get('top', 0):.2f} "
                    f"bot={z.get('bottom', 0):.2f} p_hold={z.get('p_hold', 0):.2f}"
                )

        if bear_zones:
            lines.append("  Top Bearish Zones:")
            for z in bear_zones[:3]:
                lines.append(
                    f"    - {z.get('zone_id', '?')} top={z.get('top', 0):.2f} "
                    f"bot={z.get('bottom', 0):.2f} p_hold={z.get('p_hold', 0):.2f}"
                )

        return "\n".join(lines)

    def _build_model_outputs(
        self,
        features: dict[str, Any] | None,
    ) -> str:
        """Xay dung section: Model Outputs.

        Bao gom: XGBoost Model A (BSL/SSL/lateral), Model B (zones),
        confidence.
        """
        lines: list[str] = [
            "=== MODEL OUTPUTS ===",
        ]

        if not features:
            lines.append("No model data available.")
            return "\n".join(lines)

        # Model A
        pa_bsl = features.get("p_bsl", None)
        pa_ssl = features.get("p_ssl", None)
        pa_lat = features.get("p_lateral", None)
        if pa_bsl is not None:
            lines.append(
                f"Model A: BSL={pa_bsl:.3f} SSL={pa_ssl:.3f} "
                f"Lateral={pa_lat:.3f}"
            )
            direction = "BULLISH" if pa_bsl > max(pa_ssl or 0, pa_lat or 0) else \
                        "BEARISH" if pa_ssl > max(pa_bsl or 0, pa_lat or 0) else \
                        "NEUTRAL"
            lines.append(f"Model A Direction: {direction}")

        # Model B
        zones_pred = features.get("zones_predicted", None)
        if zones_pred is not None and isinstance(zones_pred, list):
            lines.append(f"Model B Zones Predicted: {len(zones_pred)}")
            for zp in zones_pred[:3]:
                lines.append(f"  - {zp}")

        # Confidence
        conf = features.get("confidence_qualifier", "")
        if conf:
            lines.append(f"Confidence Qualifier: {conf}")

        # Guardrail
        guardrail = features.get("active_guardrail", False)
        lines.append(f"Active Guardrail: {guardrail}")

        return "\n".join(lines)

    def _build_macro_context(
        self,
        macro_context: dict[str, Any] | None,
    ) -> str:
        """Xay dung section: Macro Context.

        Bao gom: regime, news impact, su kien sap toi.
        """
        lines: list[str] = [
            "=== MACRO CONTEXT ===",
        ]

        if not macro_context:
            lines.append("No macro data available.")
            return "\n".join(lines)

        regime = macro_context.get("regime", macro_context.get("macro_regime", "UNKNOWN"))
        lines.append(f"Regime: {regime}")

        news_impact = macro_context.get("news_impact", "NONE")
        lines.append(f"News Impact: {news_impact}")

        # News events sap toi
        upcoming = macro_context.get("upcoming_events", [])
        if upcoming:
            lines.append(f"Upcoming Events ({len(upcoming)}):")
            for ev in upcoming[:5]:
                lines.append(f"  - {ev.get('title', '?')} @ {ev.get('time', '?')}")

        # Surprise flags
        surprise = macro_context.get("MAJOR_SURPRISE_FLAG", False)
        if surprise:
            lines.append("⚠ MAJOR SURPRISE FLAG ACTIVE")

        return "\n".join(lines)

    def _build_precedents(
        self,
        precedents: list[Any] | None,
    ) -> str:
        """Xay dung section: Precedents (RAG).

        Bao gom: cac debate tuong tu trong qua khu, outcome cua chung.
        Tich hop voi RAGRetriever.retrieve_precedents() thuc te.
        Neu RAG fail (exception), skip precedents section, khong crash.
        """
        lines: list[str] = [
            "=== PRECEDENTS (RAG) ===",
        ]

        try:
            if not precedents:
                lines.append("No precedents found.")
                return "\n".join(lines)

            lines.append(f"Similar Precedents Found: {len(precedents)}")
            for i, p in enumerate(precedents[:5], 1):
                # Precedent co the la object (Precedent dataclass) hoac dict
                try:
                    if hasattr(p, "to_dict"):
                        pd = p.to_dict()
                    elif isinstance(p, dict):
                        pd = p
                    else:
                        pd = {}

                    direction = pd.get("direction", "?")
                    rating = pd.get("rating", 0)
                    outcome = pd.get("outcome", pd.get("actual_outcome", "?"))
                    sim = pd.get("cosine_sim", pd.get("re_rank_score", 0))

                    lines.append(
                        f"  #{i}: {pd.get('symbol', '?')} | "
                        f"Direction: {direction} | "
                        f"Rating: {rating:+d} | "
                        f"Outcome: {outcome} | "
                        f"Similarity: {sim:.2f}"
                    )
                except Exception as item_err:
                    logger.warning(f"TechnicalBrief: error processing precedent #{i}: {item_err}")
                    lines.append(f"  #{i}: <error processing precedent>")

        except Exception as e:
            # Neu RAG fail (exception), skip precedents section, khong crash
            logger.warning(f"TechnicalBrief: RAG precedents error, skipping section: {e}")
            lines.append("Precedents unavailable due to RAG error.")
            return "\n".join(lines)

        return "\n".join(lines)

    # =========================================================================
    # Assembly
    # =========================================================================
    def _assemble(
        self,
        sections: list[str],
    ) -> tuple[str, bool]:
        """
        Ghep cac sections, dam bao khong vuot qua token limit.

        Cat bot tu section cuoi cung (Precedents) neu can.
        """
        separator = "\n\n" + "-" * 40 + "\n\n"
        full_text = separator.join(sections)

        if len(full_text) <= self._max_safe_chars:
            return full_text, False

        # Can cat bot: cat Precedents truoc
        truncated = True
        logger.warning(
            f"TechnicalBrief: {len(full_text)} chars exceeds limit "
            f"{self._max_safe_chars}, truncating..."
        )

        # Thu cat Precedents
        sections_copy = list(sections)  # sections[-1] = Precedents
        while len(sections_copy) > 1:
            # Cat bot section cuoi cung
            last_section = sections_copy.pop()
            new_full = separator.join(sections_copy)
            if len(new_full) <= self._max_safe_chars:
                return new_full, True

        # Neu van qua: cat tung section
        result = ""
        for sec in sections_copy:
            candidate = result + ("\n\n" + "-" * 40 + "\n\n" if result else "") + sec
            if len(candidate) <= self._max_safe_chars:
                result = candidate
            else:
                # Cat section nay
                remaining = self._max_safe_chars - len(result) - 50  # separator overhead
                if remaining > 100:
                    result += ("\n\n" + "-" * 40 + "\n\n" if result else "")
                    result += sec[:remaining] + "\n... [TRUNCATED]"
                break

        return result, True
