# =============================================================================
# AGENTIC-QUANT — XGBoost Models (Phase 6)
# Model A (Multiclass: BSL/SSL/Lateral) + Model B (Binary: P_hold)
# Stacked Ensemble + Session-aware Inference
# =============================================================================

from __future__ import annotations

from core.ai_engine.xgboost.feature_builder import XGBoostFeatureBuilder
from core.ai_engine.xgboost.model_a import XGBoostModelA, overconfidence_penalty_objective
from core.ai_engine.xgboost.model_b import XGBoostModelB, theta_star_threshold
from core.ai_engine.xgboost.ensemble import XGBoostEnsemble
from core.ai_engine.xgboost.inference import (
    InferenceEngineA,
    InferenceEngineB,
    RollbackController,
)

__all__ = [
    "XGBoostFeatureBuilder",
    "XGBoostModelA",
    "overconfidence_penalty_objective",
    "XGBoostModelB",
    "theta_star_threshold",
    "XGBoostEnsemble",
    "InferenceEngineA",
    "InferenceEngineB",
    "RollbackController",
]
