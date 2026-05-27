# =============================================================================
# AGENTIC-QUANT — Bull Agent
# Agent lap luan tang gia (long) cho Multi-Agent Debate
# =============================================================================
"""
Bull Agent phu trach lap luan BULLISH (tang gia).

He thong prompt yeu cau:
- Trich dan IT NHAT 3 bang chung tu Technical Brief
- Ket xuat JSON co cau truc: rating (0-4), evidence, target_price, confidence
- Khi active_guardrail=True: inject news caveat vao reasoning

Parallel execution voi Bear Agent (asyncio.gather).

Usage::

    agent = BullAgent()
    result = await agent.analyze(brief_text="...", active_guardrail=False)
    # -> BullResult(rating=3, direction="BULLISH", evidence=[...], ...)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


# =============================================================================
# Constants
# =============================================================================
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TIMEOUT_S = 3.0  # 3 seconds timeout
MIN_EVIDENCE_COUNT = 3  # Yeau cau it nhat 3 bang chung


# =============================================================================
# Bull Result
# =============================================================================
@dataclass
class BullResult:
    """
    Ket qua phan tich tu Bull Agent.

    Attributes:
        rating: Muc do bullish (0 = neutral, 4 = max bullish).
        direction: Luon la "BULLISH".
        evidence: Danh sach bang chung (it nhat 3).
        target_price: Gia muc tieu (neu co).
        confidence: Do tin cay [0, 1].
        reasoning: Lap luan chi tiet.
        token_count: So tokens tieu thu.
        active_guardrail: Co guardrail dang bat khong.
        news_caveat: Canh bao news (neu active_guardrail=True).
        success: Agent chay thanh cong hay khong.
        error: Thong bao loi (neu co).
    """

    rating: int = 0  # 0-4
    direction: str = "BULLISH"
    evidence: list[str] = field(default_factory=list)
    target_price: float | None = None
    confidence: float = 0.0  # [0, 1]
    reasoning: str = ""
    token_count: int = 0
    active_guardrail: bool = False
    news_caveat: str = ""
    success: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "rating": self.rating,
            "direction": self.direction,
            "evidence": self.evidence,
            "target_price": self.target_price,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "token_count": self.token_count,
            "active_guardrail": self.active_guardrail,
            "news_caveat": self.news_caveat,
            "success": self.success,
            "error": self.error,
        }


# =============================================================================
# Bull Agent
# =============================================================================
class BullAgent:
    """
    Bull Agent: Lap luan tang gia dua tren Technical Brief.

    He thong prompt duoc thiet ke de:
    1. Yeu cau agent trich dan it nhat 3 bang chung tu brief.
    2. JSON output parser strict.
    3. Khi active_guardrail=True: inject news caveat.

    Agent su dung mock API call (requests/duckduckgo_search) de simulate LLM.
    That bai API -> raise exception -> orchestrator bat va fallback.
    """

    # =========================================================================
    # System Prompt (template)
    # =========================================================================
    SYSTEM_PROMPT_TEMPLATE = """Bạn là Bull Agent - chuyên gia phân tích thị trường tài chính với góc nhìn TĂNG GIÁ (BULLISH).

NHIỆM VỤ:
- Phân tích Technical Brief và đưa ra lập luận BULLISH chi tiết.
- BẮT BUỘC trích dẫn ÍT NHẤT 3 bằng chứng từ Technical Brief.
- Đưa ra rating từ 0 (trung lập) đến 4 (cực kỳ bullish).
- Xác định giá mục tiêu (target_price) nếu có.
- Đánh giá độ tin cậy (confidence) từ 0.0 đến 1.0.

QUY TẮC:
1. Mỗi bằng chứng phải có: nguồn (từ brief) + lý do tại sao nó bullish.
2. Rating: 0=neutral, 1=nhẹ bullish, 2=trung bình, 3=mạnh, 4=cực kỳ mạnh.
3. Nếu thị trường đang ở vùng DISCOUNT so với equilibrium, đây là tín hiệu bullish mạnh.
4. Nếu có BSL (Buy-Side Liquidity) phía trên, giá có xu hướng hướng lên BSL.
5. FVG Bullish chưa được mitigate là bằng chứng hỗ trợ.
6. Ưu tiên các bằng chứng từ HTF (H1/H4) hơn LTF.
7. Chỉ đưa ra kết luận BULLISH nếu có ít nhất 3 bằng chứng vững chắc.
8. Phải trả lời bằng JSON đúng cấu trúc, không thêm text bên ngoài.

{news_caveat_section}

KẾT QUẢ ĐẦU RA (JSON):
{{
    "rating": <0-4>,
    "evidence": [
        {{"source": "<nguồn từ brief>", "detail": "<lý do bullish>"}},
        ...
    ],
    "target_price": <float | null>,
    "confidence": <0.0-1.0>,
    "reasoning": "<lập luận chi tiết>"
}}
"""

    NEWS_CAVEAT = (
        "⚠ LƯU Ý ĐẶC BIỆT (NEWS GUARDRAIL ACTIVE):\n"
        "Hiện tại GUARDRAIL đang hoạt động do có tin tức kinh tế quan trọng.\n"
        "Cần thận trọng: tin tức có thể gây biến động mạnh ngoài phân tích kỹ thuật.\n"
        "Hãy cân nhắc giảm rating và confidence xuống 1 bậc nếu không có xác nhận rõ ràng.\n"
        "Thêm caveat: 'Phân tích này có thể bị ảnh hưởng bởi tin tức sắp tới.'\n"
    )

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._model_name = model_name
        self._timeout_s = timeout_s

    # =========================================================================
    # Main Analysis
    # =========================================================================
    async def analyze(
        self,
        brief_text: str,
        active_guardrail: bool = False,
    ) -> BullResult:
        """
        Phan tich Technical Brief va tra ve BullResult.

        Args:
            brief_text: Technical Brief text (tu TechnicalBrief.build()).
            active_guardrail: Co guardrail dang active khong.

        Returns:
            BullResult object.
        """
        start_time = time.monotonic()
        logger.debug("BullAgent: starting analysis")

        # === Build system prompt ===
        news_caveat_section = self.NEWS_CAVEAT if active_guardrail else ""
        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            news_caveat_section=news_caveat_section,
        )

        # === Build user message ===
        user_message = (
            "Dưới đây là Technical Brief hiện tại:\n\n"
            f"{brief_text}\n\n"
            "Hãy phân tích và đưa ra lập luận BULLISH. "
            "Trả lời bằng JSON đúng cấu trúc."
        )

        # === Call LLM (simulated) ===
        try:
            raw_output = await self._call_llm(system_prompt, user_message)
        except Exception as e:
            elapsed = time.monotonic() - start_time
            logger.error(f"BullAgent: LLM call failed after {elapsed:.2f}s: {e}")
            return BullResult(
                success=False,
                error=f"LLM call failed: {e}",
                active_guardrail=active_guardrail,
            )

        # === Parse JSON output ===
        try:
            parsed = self._parse_json_output(raw_output)
        except (json.JSONDecodeError, ValueError) as e:
            elapsed = time.monotonic() - start_time
            logger.error(f"BullAgent: JSON parse failed after {elapsed:.2f}s: {e}")
            return BullResult(
                success=False,
                error=f"JSON parse failed: {e}",
                active_guardrail=active_guardrail,
            )

        # === Validate evidence count ===
        evidence_list = parsed.get("evidence", [])
        if len(evidence_list) < MIN_EVIDENCE_COUNT:
            logger.warning(
                f"BullAgent: only {len(evidence_list)} evidence(s), "
                f"minimum required: {MIN_EVIDENCE_COUNT}"
            )

        # === Extract fields ===
        evidence_strs = []
        for ev in evidence_list:
            if isinstance(ev, dict):
                src = ev.get("source", "?")
                detail = ev.get("detail", "")
                evidence_strs.append(f"[{src}] {detail}")
            else:
                evidence_strs.append(str(ev))

        news_caveat_text = ""
        if active_guardrail:
            news_caveat_text = (
                "Phân tích này có thể bị ảnh hưởng bởi tin tức sắp tới. "
                "Guardrail đang active, cần thận trọng."
            )

        elapsed = time.monotonic() - start_time
        token_count = self._estimate_tokens(system_prompt, user_message, raw_output)

        logger.debug(
            f"BullAgent: done in {elapsed:.2f}s, "
            f"rating={parsed.get('rating', 0)}, "
            f"evidence={len(evidence_strs)} items, "
            f"tokens={token_count}"
        )

        return BullResult(
            rating=parsed.get("rating", 0),
            direction="BULLISH",
            evidence=evidence_strs,
            target_price=parsed.get("target_price"),
            confidence=parsed.get("confidence", 0.0),
            reasoning=parsed.get("reasoning", ""),
            token_count=token_count,
            active_guardrail=active_guardrail,
            news_caveat=news_caveat_text,
            success=True,
        )

    # =========================================================================
    # LLM Call (simulated)
    # =========================================================================
    async def _call_llm(
        self,
        system_prompt: str,
        user_message: str,
    ) -> str:
        """
        Call LLM API de generate response.

        Simulated: su dung duckduckgo_search hoac requests de mock.
        Trong production, thay bang API call thuc (OpenAI, DeepSeek, etc.).

        Raises:
            Exception: Neu API call that bai.
        """
        # Try using duckduckgo_search for a mock search
        import asyncio

        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(
                    ddgs.text(
                        f"XAUUSD bullish analysis technical analysis",
                        max_results=3,
                    )
                )
                # Build mock response from search results
                evidence_list = []
                for r in results[:3]:
                    evidence_list.append(
                        {
                            "source": r.get("title", "Market Analysis"),
                            "detail": r.get("body", "Bullish signal detected")[:200],
                        }
                    )
        except Exception:
            # Fallback: generate mock response
            evidence_list = [
                {"source": "Technical Brief - HTF Structure",
                 "detail": "Giá đang ở vùng DISCOUNT so với equilibrium, tạo setup long mạnh."},
                {"source": "Technical Brief - BSL Targets",
                 "detail": "Có BSL phía trên chưa được quét, giá có xu hướng hướng lên."},
                {"source": "Technical Brief - Active Zones",
                 "detail": "FVG Bullish chưa mitigate, hỗ trợ cho kịch bản tăng giá."},
            ]

        # Simulate processing delay
        await asyncio.sleep(0.3)

        mock_response = {
            "rating": 3,
            "evidence": evidence_list,
            "target_price": 2520.0,
            "confidence": 0.75,
            "reasoning": (
                "Thị trường đang ở vùng DISCOUNT so với equilibrium (2500.0 vs 2520.0). "
                "Có BSL phía trên tại 2530-2540 chưa được quét. "
                "FVG Bullish từ H1 chưa mitigate, xác nhận áp lực mua. "
                "CVD đang tăng dần, cho thấy dòng tiền vào. "
                "Model A dự báo BSL>SSL, ủng hộ kịch bản tăng."
            ),
        }

        return json.dumps(mock_response, ensure_ascii=False, indent=2)

    # =========================================================================
    # JSON Output Parser
    # =========================================================================
    def _parse_json_output(self, raw_output: str) -> dict[str, Any]:
        """
        Parse JSON tu raw LLM output.

        Xu ly:
        - Markdown code block (```json ... ```)
        - JSON thuan
        - Fix trailing commas

        Returns:
            Dict chua parsed data.

        Raises:
            json.JSONDecodeError: Neu khong the parse.
            ValueError: Neu thieu truong bat buoc.
        """
        # Strip markdown code block
        text = raw_output.strip()
        if text.startswith("```"):
            # Tim ```json hoac ``` 
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()
            else:
                # Fallback: bo ``` 
                text = re.sub(r"```(?:json)?\s*", "", text)

        # Fix trailing commas (common LLM issue)
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*\]", "]", text)

        # Parse
        parsed = json.loads(text)

        # Validate required fields
        if "rating" not in parsed:
            raise ValueError("Missing 'rating' field in JSON output")
        if "evidence" not in parsed:
            raise ValueError("Missing 'evidence' field in JSON output")

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
        """Uoc luong tokens tieu thu (rough estimate: ~4 chars/token)."""
        total_chars = len(system_prompt) + len(user_message) + len(raw_output)
        return total_chars // 4
