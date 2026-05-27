# =============================================================================
# AGENTIC-QUANT — Critic Agent
# Tong hop Bull + Bear evidence thanh ConsensusResult
# =============================================================================
"""
Critic Agent dong vai tro "trong tai" tong hop lap luan Bull + Bear.

Quy trinh:
1. Nhan BullResult + BearResult tu 2 agent.
2. Phan tich bang chung tu ca 2 phia.
3. Tinh agreement_score, rating (-4 to +4), direction.
4. Tra ve ConsensusResult.

ConsensusResult:
- rating: -4 (max bear) den +4 (max bull)
- direction: "BULL" | "BEAR" | "NEUTRAL"
- agreement_score: [0, 1] — muc do dong thuan giua 2 ben
- bull_thesis / bear_thesis: Tom tat lap luan
- reasoning: Giai thich ly do consensus
- bull_evidence / bear_evidence: Danh sach bang chung
- debate_used_fallback: Co su dung fallback khong

Usage::

    critic = CriticAgent()
    consensus = await critic.evaluate(bull_result, bear_result, active_guardrail=False)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from .bull_agent import BullResult
from .bear_agent import BearResult


# =============================================================================
# Constants
# =============================================================================
DEFAULT_TIMEOUT_S = 5.0  # 5 seconds timeout (critic can thoi gian hon)


# =============================================================================
# Consensus Result
# =============================================================================
@dataclass
class ConsensusResult:
    """
    Ket cuoi cung sau khi Critic tong hop Bull + Bear.

    Attributes:
        rating: Rating tong hop tu -4 (max bear) den +4 (max bull).
        direction: Huong giao dich cuoi cung.
        confidence_qualifier: Muc do tin cay ("HIGH" | "MEDIUM" | "LOW").
        agreement_score: Muc dong thuan [0, 1].
        bull_thesis: Tom tat lap luan Bull.
        bear_thesis: Tom tat lap luan Bear.
        reasoning: Giai thich ly do consensus.
        bull_evidence: Danh sach bang chung Bull.
        bear_evidence: Danh sach bang chung Bear.
        debate_used_fallback: Co su dung fallback khong.
        success: Critic chay thanh cong khong.
        error: Thong bao loi (neu co).
        token_count: So tokens tieu thu.
    """

    rating: int = 0  # -4 to +4
    direction: str = "NEUTRAL"  # "BULL" | "BEAR" | "NEUTRAL"
    confidence_qualifier: str = "MEDIUM"  # "HIGH" | "MEDIUM" | "LOW"
    agreement_score: float = 0.0  # [0, 1]
    bull_thesis: str = ""
    bear_thesis: str = ""
    reasoning: str = ""
    bull_evidence: list[str] = field(default_factory=list)
    bear_evidence: list[str] = field(default_factory=list)
    debate_used_fallback: bool = False
    success: bool = False
    error: str = ""
    token_count: int = 0

    def to_dict(self) -> dict:
        return {
            "rating": self.rating,
            "direction": self.direction,
            "confidence_qualifier": self.confidence_qualifier,
            "agreement_score": self.agreement_score,
            "bull_thesis": self.bull_thesis,
            "bear_thesis": self.bear_thesis,
            "reasoning": self.reasoning,
            "bull_evidence": self.bull_evidence,
            "bear_evidence": self.bear_evidence,
            "debate_used_fallback": self.debate_used_fallback,
            "success": self.success,
            "error": self.error,
            "token_count": self.token_count,
        }


# =============================================================================
# Critic Agent
# =============================================================================
class CriticAgent:
    """
    Critic Agent: Trong tai tong hop Bull + Bear thanh consensus.

    Quy trinh:
    1. Nhan BullResult va BearResult.
    2. Neu ca 2 deu failed -> raise exception -> orchestrator fallback.
    3. Neu 1 failed -> su dung ben kia + giam confidence.
    4. Neu ca 2 OK -> LLM call de tong hop.
    5. Tinh rating, direction, agreement_score.
    """

    # =========================================================================
    # System Prompt (template)
    # =========================================================================
    SYSTEM_PROMPT_TEMPLATE = """Bạn là Critic Agent - chuyên gia phản biện và tổng hợp thị trường tài chính.

NHIỆM VỤ:
Bạn nhận được lập luận từ Bull Agent (TĂNG GIÁ) và Bear Agent (GIẢM GIÁ).
Nhiệm vụ của bạn là:
1. Phân tích các bằng chứng từ CẢ HAI phía.
2. Đánh giá mức độ thuyết phục của mỗi bên.
3. Đưa ra quyết định CUỐI CÙNG (consensus).
4. Giải thích lý do tại sao bạn chọn hướng đó.

HƯỚNG DẪN:
- rating: từ -4 (cực kỳ bear) đến +4 (cực kỳ bull). 0 = trung lập.
- agreement_score: từ 0.0 (hoàn toàn bất đồng) đến 1.0 (hoàn toàn đồng thuận).
  Nếu Bull và Bear đều đưa ra bằng chứng mạnh, agreement_score thấp.
  Nếu một bên áp đảo, agreement_score cao.
- confidence_qualifier: "HIGH" nếu rating có |rating| >= 3 và agreement > 0.6.
  "MEDIUM" nếu |rating| >= 1. "LOW" nếu |rating| = 0 hoặc agreement < 0.3.
- bull_thesis: Tóm tắt luận điểm Bull (2-3 câu).
- bear_thesis: Tóm tắt luận điểm Bear (2-3 câu).
- reasoning: Giải thích tại sao bạn chọn hướng này (3-5 câu).

{news_caveat_section}

KẾT QUẢ ĐẦU RA (JSON):
{{
    "rating": <-4 to 4>,
    "agreement_score": <0.0-1.0>,
    "confidence_qualifier": "HIGH" | "MEDIUM" | "LOW",
    "bull_thesis": "<tóm tắt luận Bull>",
    "bear_thesis": "<tóm tắt luận Bear>",
    "reasoning": "<giải thích lý do consensus>"
}}
"""

    NEWS_CAVEAT = (
        "⚠ LƯU Ý ĐẶC BIỆT (NEWS GUARDRAIL ACTIVE):\n"
        "Guardrail tin tức đang hoạt động. Hãy thận trọng khi đưa ra consensus:\n"
        "- Giảm |rating| xuống 1 nếu không có xác nhận rõ ràng từ cả 2 phía.\n"
        "- Giảm confidence_qualifier xuống 1 bậc.\n"
        "- Thêm caveat vào reasoning về ảnh hưởng của tin tức.\n"
    )

    def __init__(self, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self._timeout_s = timeout_s

    # =========================================================================
    # Main Evaluation
    # =========================================================================
    async def evaluate(
        self,
        bull_result: BullResult,
        bear_result: BearResult,
        active_guardrail: bool = False,
    ) -> ConsensusResult:
        """
        Tong hop Bull + Bear thanh ConsensusResult.

        Args:
            bull_result: Ket qua tu Bull Agent.
            bear_result: Ket qua tu Bear Agent.
            active_guardrail: Co guardrail dang active khong.

        Returns:
            ConsensusResult da tong hop.
        """
        start_time = time.monotonic()
        logger.debug("CriticAgent: starting evaluation")

        # === Neu ca 2 agent deu failed ===
        if not bull_result.success and not bear_result.success:
            logger.error("CriticAgent: both Bull and Bear agents failed")
            return ConsensusResult(
                rating=0,
                direction="NEUTRAL",
                confidence_qualifier="LOW",
                agreement_score=0.0,
                reasoning="Cả Bull và Bear agent đều không thể phân tích.",
                debate_used_fallback=True,
                success=False,
                error="Both agents failed",
            )

        # === Neu 1 agent failed, su dung ben con lai ===
        if not bull_result.success:
            logger.warning("CriticAgent: Bull agent failed, using Bear only (reduced confidence)")
            return self._single_sided_result(bear_result, is_bull=False, guardrail=active_guardrail)

        if not bear_result.success:
            logger.warning("CriticAgent: Bear agent failed, using Bull only (reduced confidence)")
            return self._single_sided_result(bull_result, is_bull=True, guardrail=active_guardrail)

        # === Ca 2 OK -> LLM call ===
        news_caveat_section = self.NEWS_CAVEAT if active_guardrail else ""
        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            news_caveat_section=news_caveat_section,
        )

        user_message = self._build_user_message(bull_result, bear_result)

        try:
            raw_output = await self._call_llm(system_prompt, user_message)
        except Exception as e:
            elapsed = time.monotonic() - start_time
            logger.error(f"CriticAgent: LLM call failed after {elapsed:.2f}s: {e}")
            return self._rule_based_fallback(bull_result, bear_result, active_guardrail)

        # === Parse JSON ===
        try:
            parsed = self._parse_json_output(raw_output)
        except (json.JSONDecodeError, ValueError) as e:
            elapsed = time.monotonic() - start_time
            logger.error(f"CriticAgent: JSON parse failed after {elapsed:.2f}s: {e}")
            return self._rule_based_fallback(bull_result, bear_result, active_guardrail)

        # === Validate ===
        rating = parsed.get("rating", 0)
        rating = max(-4, min(4, rating))  # Clamp to [-4, 4]

        # Determine direction
        if rating > 1:
            direction = "BULL"
        elif rating < -1:
            direction = "BEAR"
        else:
            direction = "NEUTRAL"

        agreement = parsed.get("agreement_score", 0.5)
        agreement = max(0.0, min(1.0, agreement))

        conf = parsed.get("confidence_qualifier", "MEDIUM")
        if conf not in ("HIGH", "MEDIUM", "LOW"):
            conf = "MEDIUM"

        # Guardrail adjustment
        news_caveat_text = ""
        if active_guardrail:
            if abs(rating) >= 2:
                rating = rating - (1 if rating > 0 else -1)  # Giam 1 bac
            if conf == "HIGH":
                conf = "MEDIUM"
            elif conf == "MEDIUM":
                conf = "LOW"
            news_caveat_text = "Guardrail active - consensus da duoc dieu chinh giam."

        elapsed = time.monotonic() - start_time
        token_count = self._estimate_tokens(system_prompt, user_message, raw_output)

        logger.debug(
            f"CriticAgent: done in {elapsed:.2f}s, "
            f"rating={rating}, direction={direction}, "
            f"agreement={agreement:.2f}, tokens={token_count}"
        )

        return ConsensusResult(
            rating=rating,
            direction=direction,
            confidence_qualifier=conf,
            agreement_score=agreement,
            bull_thesis=parsed.get("bull_thesis", bull_result.reasoning[:200]),
            bear_thesis=parsed.get("bear_thesis", bear_result.reasoning[:200]),
            reasoning=parsed.get("reasoning", ""),
            bull_evidence=bull_result.evidence,
            bear_evidence=bear_result.evidence,
            debate_used_fallback=False,
            success=True,
            token_count=token_count,
        )

    # =========================================================================
    # Single-sided Fallback
    # =========================================================================
    def _single_sided_result(
        self,
        agent_result: BullResult | BearResult,
        is_bull: bool,
        guardrail: bool,
    ) -> ConsensusResult:
        """Tao ConsensusResult khi chi co 1 agent chay duoc."""
        direction = "BULL" if is_bull else "BEAR"
        rating = agent_result.rating if is_bull else -agent_result.rating

        # Reduce confidence
        conf = "LOW"
        agreement = 0.0

        if guardrail and abs(rating) >= 2:
            rating = max(1, rating - 1) if is_bull else min(-1, rating + 1)

        return ConsensusResult(
            rating=rating,
            direction=direction,
            confidence_qualifier=conf,
            agreement_score=agreement,
            bull_thesis=agent_result.reasoning[:200] if is_bull else "",
            bear_thesis=agent_result.reasoning[:200] if not is_bull else "",
            reasoning=(
                f"Chi có {'Bull' if is_bull else 'Bear'} agent hoạt động. "
                f"Độ tin cậy thấp do thiếu đối trọng."
            ),
            bull_evidence=agent_result.evidence if is_bull else [],
            bear_evidence=agent_result.evidence if not is_bull else [],
            debate_used_fallback=True,
            success=True,
            token_count=agent_result.token_count,
        )

    # =========================================================================
    # Rule-based Fallback
    # =========================================================================
    def _rule_based_fallback(
        self,
        bull_result: BullResult,
        bear_result: BearResult,
        active_guardrail: bool,
    ) -> ConsensusResult:
        """
        Fallback dua tren rules khi LLM call that bai.

        So sanh rating va confidence cua 2 ben, chon ben cao hon.
        """
        logger.warning("CriticAgent: using rule-based fallback")

        bull_weight = bull_result.rating * bull_result.confidence
        bear_weight = bear_result.rating * bear_result.confidence

        if bull_weight > bear_weight:
            rating = min(4, bull_result.rating)
            direction = "BULL"
            bull_thesis = bull_result.reasoning[:200]
            bear_thesis = bear_result.reasoning[:200]
            evidence = bull_result.evidence
        elif bear_weight > bull_weight:
            rating = max(-4, -bear_result.rating)
            direction = "BEAR"
            bull_thesis = bull_result.reasoning[:200]
            bear_thesis = bear_result.reasoning[:200]
            evidence = bear_result.evidence
        else:
            rating = 0
            direction = "NEUTRAL"
            bull_thesis = bull_result.reasoning[:200]
            bear_thesis = bear_result.reasoning[:200]

        agreement = 1.0 - abs(bull_weight - bear_weight) / (max(bull_weight, bear_weight, 0.01))
        agreement = max(0.0, min(1.0, agreement))

        conf = "MEDIUM" if abs(rating) >= 2 else "LOW"

        if active_guardrail:
            conf = "LOW"

        total_tokens = bull_result.token_count + bear_result.token_count

        return ConsensusResult(
            rating=rating,
            direction=direction,
            confidence_qualifier=conf,
            agreement_score=agreement,
            bull_thesis=bull_thesis,
            bear_thesis=bear_thesis,
            reasoning=f"Rule-based fallback: Bull weight={bull_weight:.2f}, Bear weight={bear_weight:.2f}",
            bull_evidence=bull_result.evidence,
            bear_evidence=bear_result.evidence,
            debate_used_fallback=True,
            success=True,
            token_count=total_tokens,
        )

    # =========================================================================
    # Build User Message
    # =========================================================================
    def _build_user_message(
        self,
        bull_result: BullResult,
        bear_result: BearResult,
    ) -> str:
        """Build user message cho LLM call."""
        lines = [
            "=== BULL AGENT ===",
            f"Rating: {bull_result.rating}/4",
            f"Confidence: {bull_result.confidence:.2f}",
            f"Target Price: {bull_result.target_price}",
            "",
            "Evidence:",
        ]
        for ev in bull_result.evidence:
            lines.append(f"  - {ev}")

        lines.extend([
            "",
            f"Reasoning: {bull_result.reasoning}",
            "",
            "=== BEAR AGENT ===",
            f"Rating: {bear_result.rating}/4",
            f"Confidence: {bear_result.confidence:.2f}",
            f"Target Price: {bear_result.target_price}",
            "",
            "Evidence:",
        ])
        for ev in bear_result.evidence:
            lines.append(f"  - {ev}")

        lines.extend([
            "",
            f"Reasoning: {bear_result.reasoning}",
            "",
            "---",
            "Hãy đưa ra consensus dựa trên thông tin trên.",
        ])

        return "\n".join(lines)

    # =========================================================================
    # LLM Call
    # =========================================================================
    async def _call_llm(
        self,
        system_prompt: str,
        user_message: str,
    ) -> str:
        """
        Call LLM API de tong hop Bull + Bear.

        Simulated: tra ve mock consensus.
        Trong production, thay bang API call thuc.

        Raises:
            Exception: Neu API call that bai.
        """
        import asyncio

        await asyncio.sleep(0.5)

        mock_response = {
            "rating": 2,
            "agreement_score": 0.65,
            "confidence_qualifier": "MEDIUM",
            "bull_thesis": (
                "Thị trường đang ở vùng DISCOUNT, BSL chưa quét, FVG Bullish chưa mitigate. "
                "CVD tăng cho thấy dòng tiền vào."
            ),
            "bear_thesis": (
                "Có SSL phía dưới chưa quét, FVG Bearish chưa mitigate. "
                "Tuy nhiên các bằng chứng Bear yếu hơn Bull."
            ),
            "reasoning": (
                "Bull có 3 bằng chứng mạnh (DISCOUNT zone, BSL phía trên, FVG Bullish), "
                "trong khi Bear chỉ có 2 bằng chứng trung bình. "
                "Bull confidence (0.75) cao hơn Bear (0.72). "
                "Kết luận: nghiêng về BULL với rating +2."
            ),
        }

        return json.dumps(mock_response, ensure_ascii=False, indent=2)

    # =========================================================================
    # JSON Output Parser
    # =========================================================================
    def _parse_json_output(self, raw_output: str) -> dict[str, Any]:
        """Parse JSON tu raw LLM output."""
        text = raw_output.strip()
        if text.startswith("```"):
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()
            else:
                text = re.sub(r"```(?:json)?\s*", "", text)

        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*\]", "]", text)

        parsed = json.loads(text)

        if "rating" not in parsed:
            raise ValueError("Missing 'rating' field")
        if "agreement_score" not in parsed:
            raise ValueError("Missing 'agreement_score' field")

        return parsed

    # =========================================================================
    # Token Estimation
    # =========================================================================
    @staticmethod
    def _estimate_tokens(
        system_prompt: str,
        user_message: str,
        raw_output: str,
    ) -> int:
        """Uoc luong tokens tieu thu."""
        total_chars = len(system_prompt) + len(user_message) + len(raw_output)
        return total_chars // 4
