# =============================================================================
# AGENTIC-QUANT — Multi-Agent Debate System
# Phase 7: Bull / Bear / Critic tranh bien de hinh thanh consensus
# =============================================================================
"""
Multi-Agent Debate System cho AGENTIC-QUANT.

He thong gom:
- TechnicalBrief: Xay dung context cho debate tu features, macro, RAG precedents
- BullAgent: Lap luan tang gia (long) voi bang chung cu the
- BearAgent: Lap luan giam gia (short) song song voi Bull
- CriticAgent: Tong hop Bull + Bear thanh ConsensusResult
- DebateOrchestrator: Dieu phoi toan bo luong debate + timeout + Redis
- RuleFallback: Fallback dua tren rules khi model bi degraded
"""

from .bull_agent import BullAgent, BullResult
from .bear_agent import BearAgent, BearResult
from .critic_agent import CriticAgent, ConsensusResult
from .debate_orchestrator import DebateOrchestrator
from .rule_fallback import RuleFallback
from .technical_brief import TechnicalBrief, TechnicalBriefData

__all__ = [
    "BullAgent",
    "BullResult",
    "BearAgent",
    "BearResult",
    "CriticAgent",
    "ConsensusResult",
    "DebateOrchestrator",
    "RuleFallback",
    "TechnicalBrief",
    "TechnicalBriefData",
]
