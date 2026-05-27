# =============================================================================
# AGENTIC-QUANT — XGBoost Inference Engines (Phase 6.4)
# Inference Engine A (direction) + Inference Engine B (zone hold)
# + Rollback Controller (3 versions: standard, aggressive, conservative)
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from core.ai_engine.xgboost.feature_builder import XGBoostFeatures
    from core.ai_engine.xgboost.model_a import XGBoostModelA
    from core.ai_engine.xgboost.model_b import XGBoostModelB
    from core.ai_engine.xgboost.ensemble import XGBoostEnsemble
else:
    # Lazy imports to avoid circular deps at import time
    from core.ai_engine.xgboost.feature_builder import XGBoostFeatures
    from core.ai_engine.xgboost.model_a import XGBoostModelA
    from core.ai_engine.xgboost.model_b import XGBoostModelB
    from core.ai_engine.xgboost.ensemble import XGBoostEnsemble


# =============================================================================
# Constants
# =============================================================================

# Confidence thresholds
CONFIDENCE_HIGH = 0.85
CONFIDENCE_MEDIUM = 0.65

# Rollback versions
ROLLBACK_STANDARD = "standard"
ROLLBACK_AGGRESSIVE = "aggressive"
ROLLBACK_CONSERVATIVE = "conservative"

# Default session weights (matching Phase 4 event_arbitrage logic)
SESSION_WEIGHT_ACCUMULATION: float = 0.2  # P_lateral += (P_BSL+P_SSL) * 0.2
SESSION_WEIGHT_EXPANSION: float = 0.0     # confidence_qualifier -> HIGH if >= MEDIUM
SESSION_WEIGHT_REVERSAL: float = 0.10     # Bear Evidence weight += 10%

# Regime codes (from core.memory.models.enums.RegimeType)
REGIME_NORMAL = 0
REGIME_TRENDING_LV = 1
REGIME_TRENDING_HV = 2
REGIME_CHOPPY_HV = 3


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ModelAOutput:
    """Output from Inference Engine A.

    Attributes:
        p_bsl: Probability of BSL (Buy-side Liquidity) hit
        p_ssl: Probability of SSL (Sell-side Liquidity) hit
        p_lateral: Probability of lateral/no clear direction
        predicted_class: Predicted class (0=BSL, 1=SSL, 2=LATERAL)
        predicted_label: Human-readable label ("BSL", "SSL", "LATERAL")
        confidence_qualifier: "HIGH", "MEDIUM", or "LOW"
        max_prob: Maximum probability value
        adjusted: Whether adjusted by session weights
        regime_code: Market regime at prediction time
    """
    p_bsl: float = 0.0
    p_ssl: float = 0.0
    p_lateral: float = 0.0
    predicted_class: int = 2  # Default: LATERAL
    predicted_label: str = "LATERAL"
    confidence_qualifier: str = "LOW"
    max_prob: float = 0.0
    adjusted: bool = False
    regime_code: int = 0


@dataclass
class ModelBOutput:
    """Output from Inference Engine B.

    Attributes:
        p_hold: Probability that zone holds (unmitigated)
        p_not_hold: Probability that zone is mitigated
        predicted_hold: Binary: 1=hold, 0=not_hold (using theta*)
        confidence: Prediction confidence (margin from threshold)
        theta_star: Decision threshold used
    """
    p_hold: float = 0.0
    p_not_hold: float = 0.0
    predicted_hold: int = 0
    confidence: float = 0.0
    theta_star: float = 0.71


@dataclass
class EnsembleOutput:
    """Output from stacked ensemble.

    Attributes:
        p_bsl: Ensemble P_BSL
        p_ssl: Ensemble P_SSL
        p_lateral: Ensemble P_lateral
        predicted_class: Predicted class
        predicted_label: Human-readable label
        xgb_weight: How much XGBoost contributed
        lstm_weight: How much LSTM contributed
        regime_bias: Regime-specific bias direction
    """
    p_bsl: float = 0.0
    p_ssl: float = 0.0
    p_lateral: float = 0.0
    predicted_class: int = 2
    predicted_label: str = "LATERAL"
    xgb_weight: float = 0.5
    lstm_weight: float = 0.5
    regime_bias: str = "NEUTRAL"


@dataclass
class InferenceResult:
    """Complete inference result from both engines + ensemble.

    Attributes:
        model_a: Model A output (direction)
        model_b: Model B output (zone hold)
        ensemble: Ensemble output (optional)
        feature_snapshot: Feature vector snapshot for debugging
        session_code: Session code at inference time
        regime_code: Regime code at inference time
        timestamp_ms: Inference timestamp
        rollback_version: Which rollback was applied
    """
    model_a: ModelAOutput = field(default_factory=ModelAOutput)
    model_b: ModelBOutput | None = None
    ensemble: EnsembleOutput | None = None
    feature_snapshot: dict[str, Any] = field(default_factory=dict)
    session_code: int = 0
    regime_code: int = 0
    timestamp_ms: int = 0
    rollback_version: str = ROLLBACK_STANDARD


# =============================================================================
# Inference Engine A: Direction
# =============================================================================

class InferenceEngineA:
    """Inference Engine for Model A — Direction Prediction.

    Uses XGBoost Model A to predict BSL/SSL/LATERAL direction,
    then applies session weight adjustments from Phan IV.1.

    Session weight logic:
      - ACCUMULATION (NORMAL, TRENDING_LV):
          P_lateral += (P_BSL + P_SSL) * 0.2
      - EXPANSION: confidence_qualifier -> HIGH if >= MEDIUM
      - REVERSAL_RISK: Bear Evidence weight += 10%
    """

    def __init__(
        self,
        model_a: XGBoostModelA | None = None,
    ) -> None:
        self._model_a = model_a

    @property
    def model_a(self) -> XGBoostModelA | None:
        return self._model_a

    @model_a.setter
    def model_a(self, model: XGBoostModelA) -> None:
        self._model_a = model

    # ------------------------------------------------------------------
    # Main inference
    # ------------------------------------------------------------------

    def predict(
        self,
        X_A: np.ndarray,
        regime_code: int = 0,
        bear_evidence: float = 0.0,
        apply_session_weights: bool = True,
    ) -> ModelAOutput:
        """Run inference with Model A.

        Args:
            X_A: Feature vector (648,) or (n_samples, 648)
            regime_code: Market regime code
            bear_evidence: Bear evidence weight (0.0-1.0)
            apply_session_weights: Apply session weight adjustments

        Returns:
            ModelAOutput with probabilities and metadata
        """
        if self._model_a is None or not self._model_a.is_trained:
            logger.warning("Model A not trained. Returning default output.")
            return ModelAOutput()

        # Ensure 2D
        if X_A.ndim == 1:
            X_A = X_A.reshape(1, -1)

        # Predict probabilities
        probas = self._model_a.predict_proba(X_A)

        # Apply session weights if requested
        if apply_session_weights:
            probas = self._model_a.apply_session_weights(
                probas, regime_code, bear_evidence
            )

        # Extract single sample result
        proba = probas[0]
        p_bsl = float(proba[0])
        p_ssl = float(proba[1])
        p_lateral = float(proba[2])

        predicted_class = int(np.argmax(proba))
        max_prob = float(np.max(proba))

        # Confidence qualifier (Phan IV.1: EXPANSION -> HIGH if >= MEDIUM)
        if max_prob >= CONFIDENCE_HIGH:
            confidence_qualifier = "HIGH"
        elif max_prob >= CONFIDENCE_MEDIUM:
            confidence_qualifier = "MEDIUM"
        else:
            confidence_qualifier = "LOW"

        # EXPANSION: if regime is trending and confidence is at least MEDIUM,
        # upgrade to HIGH
        if regime_code in (REGIME_TRENDING_LV, REGIME_TRENDING_HV):
            if confidence_qualifier in ("MEDIUM", "HIGH"):
                confidence_qualifier = "HIGH"

        predicted_label = ["BSL", "SSL", "LATERAL"][predicted_class]

        return ModelAOutput(
            p_bsl=p_bsl,
            p_ssl=p_ssl,
            p_lateral=p_lateral,
            predicted_class=predicted_class,
            predicted_label=predicted_label,
            confidence_qualifier=confidence_qualifier,
            max_prob=max_prob,
            adjusted=apply_session_weights,
            regime_code=regime_code,
        )

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        X_A: np.ndarray,
        regime_codes: np.ndarray | None = None,
        bear_evidences: np.ndarray | None = None,
        apply_session_weights: bool = True,
    ) -> list[ModelAOutput]:
        """Run batch inference with Model A.

        Args:
            X_A: Feature matrix (n_samples, 648)
            regime_codes: Regime codes per sample (n_samples,)
            bear_evidences: Bear evidence per sample (n_samples,)
            apply_session_weights: Apply session weight adjustments

        Returns:
            List of ModelAOutput per sample
        """
        if X_A.ndim == 1:
            X_A = X_A.reshape(1, -1)

        n_samples = X_A.shape[0]

        if regime_codes is None:
            regime_codes = np.zeros(n_samples, dtype=np.int32)
        if bear_evidences is None:
            bear_evidences = np.zeros(n_samples, dtype=np.float64)

        results: list[ModelAOutput] = []
        for i in range(n_samples):
            result = self.predict(
                X_A[i],
                regime_code=int(regime_codes[i]),
                bear_evidence=float(bear_evidences[i]),
                apply_session_weights=apply_session_weights,
            )
            results.append(result)

        return results


# =============================================================================
# Inference Engine B: Zone Hold
# =============================================================================

class InferenceEngineB:
    """Inference Engine for Model B — Zone Hold Prediction.

    Uses XGBoost Model B to predict whether a zone will hold (survive)
    or be mitigated, using theta* = 0.71 calibrated threshold.
    """

    def __init__(
        self,
        model_b: XGBoostModelB | None = None,
    ) -> None:
        self._model_b = model_b

    @property
    def model_b(self) -> XGBoostModelB | None:
        return self._model_b

    @model_b.setter
    def model_b(self, model: XGBoostModelB) -> None:
        self._model_b = model

    # ------------------------------------------------------------------
    # Main inference
    # ------------------------------------------------------------------

    def predict(
        self,
        X_B: np.ndarray,
    ) -> ModelBOutput:
        """Run inference with Model B.

        Args:
            X_B: Feature vector (560,) or (n_samples, 560)

        Returns:
            ModelBOutput with P_hold, prediction, and confidence
        """
        if self._model_b is None or not self._model_b.is_trained:
            logger.warning("Model B not trained. Returning default output.")
            return ModelBOutput()

        if X_B.ndim == 1:
            X_B = X_B.reshape(1, -1)

        probas = self._model_b.predict_proba(X_B)
        p_not_hold = float(probas[0, 0])
        p_hold = float(probas[0, 1])

        predicted_hold = int(self._model_b.predict(X_B)[0])
        confidence = float(self._model_b.predict_with_confidence(X_B)[0])

        return ModelBOutput(
            p_hold=p_hold,
            p_not_hold=p_not_hold,
            predicted_hold=predicted_hold,
            confidence=confidence,
            theta_star=self._model_b.theta_star,
        )

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        X_B: np.ndarray,
    ) -> list[ModelBOutput]:
        """Run batch inference with Model B."""
        if X_B.ndim == 1:
            X_B = X_B.reshape(1, -1)

        probas = self._model_b.predict_proba(X_B)
        preds = self._model_b.predict(X_B)
        confs = self._model_b.predict_with_confidence(X_B)
        theta = self._model_b.theta_star

        results: list[ModelBOutput] = []
        for i in range(probas.shape[0]):
            results.append(ModelBOutput(
                p_hold=float(probas[i, 1]),
                p_not_hold=float(probas[i, 0]),
                predicted_hold=int(preds[i]),
                confidence=float(confs[i]),
                theta_star=theta,
            ))

        return results


# =============================================================================
# Rollback Controller
# =============================================================================

class RollbackController:
    """Rollback Controller — 3 versions for different market conditions.

    Phan IV.1 — Session Weight Logic:
      Rollback decides whether to override model predictions based on
      market conditions, confidence, and regime.

    Rollback versions:
      - STANDARD: Default — no override if confidence >= MEDIUM
      - AGGRESSIVE: More likely to rollback to lateral -> override if moving
                    fast with low confidence
      - CONSERVATIVE: Less likely to rollback -> only override on HIGH
                      confidence or extreme conditions
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Rollback logic
    # ------------------------------------------------------------------

    def should_rollback(
        self,
        model_a_output: ModelAOutput,
        model_b_output: ModelBOutput | None = None,
        version: str = ROLLBACK_STANDARD,
        price_momentum: float = 0.0,
        volatility: float = 0.0,
        regime_code: int = 0,
    ) -> tuple[bool, str]:
        """Determine if prediction should be rolled back.

        Args:
            model_a_output: Output from Inference Engine A
            model_b_output: Output from Inference Engine B (optional)
            version: Rollback version ("standard", "aggressive", "conservative")
            price_momentum: Current price momentum
            volatility: Current market volatility
            regime_code: Market regime code

        Returns:
            (should_rollback, reason) tuple
        """
        confidence = model_a_output.confidence_qualifier
        max_prob = model_a_output.max_prob
        predicted_label = model_a_output.predicted_label

        # ---------------------------------------------------------------
        # Version 1: STANDARD
        # ---------------------------------------------------------------
        if version == ROLLBACK_STANDARD:
            # No rollback if confidence is at least MEDIUM
            if confidence in ("HIGH", "MEDIUM"):
                return False, ""

            # Rollback to lateral if low confidence
            if confidence == "LOW":
                return True, f"LOW confidence ({max_prob:.2f}) -> rollback to LATERAL"

            return False, ""

        # ---------------------------------------------------------------
        # Version 2: AGGRESSIVE
        # More likely to rollback — especially in choppy/sideways markets
        # ---------------------------------------------------------------
        if version == ROLLBACK_AGGRESSIVE:
            # Always rollback if low confidence
            if confidence == "LOW":
                return True, f"AGGRESSIVE: LOW confidence ({max_prob:.2f}) -> rollback"

            # Rollback in choppy regimes with medium confidence
            if regime_code == REGIME_CHOPPY_HV and confidence == "MEDIUM":
                return True, "AGGRESSIVE: CHOPPY_HV + MEDIUM confidence -> rollback"

            # Rollback if momentum contradicts prediction
            if predicted_label == "BSL" and price_momentum < -0.5:
                return True, "AGGRESSIVE: BSL vs bearish momentum -> rollback"
            if predicted_label == "SSL" and price_momentum > 0.5:
                return True, "AGGRESSIVE: SSL vs bullish momentum -> rollback"

            return False, ""

        # ---------------------------------------------------------------
        # Version 3: CONSERVATIVE
        # Less likely to rollback — only on extreme conditions
        # ---------------------------------------------------------------
        if version == ROLLBACK_CONSERVATIVE:
            # Only rollback on very low confidence
            if confidence == "LOW" and max_prob < 0.50:
                return True, f"CONSERVATIVE: Very LOW confidence ({max_prob:.2f}) -> rollback"

            # Rollback only if Model B also says zone won't hold
            if model_b_output is not None and confidence == "LOW":
                if model_b_output.predicted_hold == 0:
                    return True, (
                        f"CONSERVATIVE: LOW confidence + zone not hold -> rollback"
                    )

            # Rollback on extreme volatility (chop + low confidence)
            if regime_code == REGIME_CHOPPY_HV and volatility > 2.0 and confidence == "LOW":
                return True, "CONSERVATIVE: Extreme volatility + LOW confidence -> rollback"

            return False, ""

        # Default: no rollback
        return False, ""

    def apply_rollback(
        self,
        model_a_output: ModelAOutput,
        version: str = ROLLBACK_STANDARD,
    ) -> ModelAOutput:
        """Apply rollback by overriding prediction to LATERAL.

        Args:
            model_a_output: Original Model A output
            version: Rollback version

        Returns:
            Rolled-back Model A output (P_lateral = 1.0)
        """
        rolled_back = ModelAOutput(
            p_bsl=0.0,
            p_ssl=0.0,
            p_lateral=1.0,
            predicted_class=2,
            predicted_label="LATERAL",
            confidence_qualifier="LOW",
            max_prob=1.0,
            adjusted=model_a_output.adjusted,
            regime_code=model_a_output.regime_code,
        )

        logger.debug(
            f"Rollback ({version}) applied: "
            f"{model_a_output.predicted_label} -> LATERAL"
        )

        return rolled_back

    # ------------------------------------------------------------------
    # Full inference pipeline with rollback
    # ------------------------------------------------------------------

    def run_inference_with_rollback(
        self,
        engine_a: InferenceEngineA,
        engine_b: InferenceEngineB | None,
        X_A: np.ndarray,
        X_B: np.ndarray | None = None,
        regime_code: int = 0,
        bear_evidence: float = 0.0,
        price_momentum: float = 0.0,
        volatility: float = 0.0,
        version: str = ROLLBACK_STANDARD,
        apply_session_weights: bool = True,
    ) -> InferenceResult:
        """Run full inference pipeline with rollback.

        Flow:
          1. Inference Engine A -> direction probabilities
          2. Inference Engine B -> zone hold probability
          3. Check rollback conditions
          4. Apply rollback if needed
          5. Return combined result

        Args:
            engine_a: Inference Engine A instance
            engine_b: Inference Engine B instance (optional)
            X_A: Feature vector for Model A (648,)
            X_B: Feature vector for Model B (560,)
            regime_code: Market regime code
            bear_evidence: Bear evidence weight
            price_momentum: Price momentum for rollback check
            volatility: Market volatility for rollback check
            version: Rollback version
            apply_session_weights: Apply session weight adjustments

        Returns:
            InferenceResult with all outputs
        """
        # Step 1: Model A inference
        model_a_out = engine_a.predict(
            X_A,
            regime_code=regime_code,
            bear_evidence=bear_evidence,
            apply_session_weights=apply_session_weights,
        )

        # Step 2: Model B inference (optional)
        model_b_out = None
        if engine_b is not None and X_B is not None:
            model_b_out = engine_b.predict(X_B)

        # Step 3: Check rollback
        should_roll, reason = self.should_rollback(
            model_a_out,
            model_b_out,
            version=version,
            price_momentum=price_momentum,
            volatility=volatility,
            regime_code=regime_code,
        )

        # Step 4: Apply rollback if needed
        if should_roll:
            model_a_out = self.apply_rollback(model_a_out, version=version)
            logger.info(f"Rollback triggered: {reason}")

        # Step 5: Build feature snapshot
        snapshot: dict[str, Any] = {}
        if isinstance(X_A, np.ndarray):
            snapshot["X_A_mean"] = float(np.mean(X_A))
            snapshot["X_A_std"] = float(np.std(X_A))
            snapshot["X_A_norm"] = float(np.linalg.norm(X_A))
        if isinstance(X_B, np.ndarray):
            snapshot["X_B_mean"] = float(np.mean(X_B))
            snapshot["X_B_std"] = float(np.std(X_B))

        return InferenceResult(
            model_a=model_a_out,
            model_b=model_b_out,
            feature_snapshot=snapshot,
            session_code=0,
            regime_code=regime_code,
            timestamp_ms=0,
            rollback_version=version,
        )

    def run_inference_with_ensemble(
        self,
        engine_a: InferenceEngineA,
        engine_b: InferenceEngineB | None,
        ensemble: XGBoostEnsemble | None,
        X_A: np.ndarray,
        X_B: np.ndarray | None = None,
        lstm_consensus: float = 0.0,
        regime_code: int = 0,
        bear_evidence: float = 0.0,
        price_momentum: float = 0.0,
        volatility: float = 0.0,
        version: str = ROLLBACK_STANDARD,
        apply_session_weights: bool = True,
    ) -> InferenceResult:
        """Run full inference with ensemble meta-learner.

        Flow:
          1. Model A inference
          2. Model B inference
          3. Build meta-features: [p_bsl_xgb, p_ssl_xgb, p_lat_xgb, lstm_consensus, regime]
          4. Ensemble meta-prediction
          5. Check rollback on ensemble output
          6. Return result

        Args:
            engine_a: Inference Engine A
            engine_b: Inference Engine B
            ensemble: XGBoostEnsemble meta-learner
            X_A: Feature vector for Model A
            X_B: Feature vector for Model B
            lstm_consensus: LSTM consensus score
            regime_code: Market regime code
            bear_evidence: Bear evidence weight
            price_momentum: Price momentum
            volatility: Market volatility
            version: Rollback version
            apply_session_weights: Apply session weights

        Returns:
            InferenceResult with ensemble output
        """
        # Step 1: Model A inference
        model_a_out = engine_a.predict(
            X_A,
            regime_code=regime_code,
            bear_evidence=bear_evidence,
            apply_session_weights=apply_session_weights,
        )

        # Step 2: Model B inference
        model_b_out = None
        if engine_b is not None and X_B is not None:
            model_b_out = engine_b.predict(X_B)

        # Step 3: Ensemble meta-prediction
        ensemble_out: EnsembleOutput | None = None
        if ensemble is not None and ensemble.is_trained:
            meta_X = ensemble.build_meta_features(
                np.array([model_a_out.p_bsl]),
                np.array([model_a_out.p_ssl]),
                np.array([model_a_out.p_lateral]),
                np.array([lstm_consensus]),
                np.array([regime_code]),
            )
            ensemble_probas = ensemble.predict_proba(meta_X)[0]
            ensemble_pred = int(np.argmax(ensemble_probas))

            # Get weights
            weights = ensemble.get_feature_weights()

            ensemble_out = EnsembleOutput(
                p_bsl=float(ensemble_probas[0]),
                p_ssl=float(ensemble_probas[1]),
                p_lateral=float(ensemble_probas[2]),
                predicted_class=ensemble_pred,
                predicted_label=["BSL", "SSL", "LATERAL"][ensemble_pred],
                xgb_weight=weights.get("p_bsl_xgb", 0.5),
                lstm_weight=weights.get("lstm_consensus", 0.5),
                regime_bias="BULLISH" if ensemble_pred == 0 else (
                    "BEARISH" if ensemble_pred == 1 else "NEUTRAL"
                ),
            )

            # For rollback check, use ensemble output instead of model_a
            rollback_check_target = ModelAOutput(
                p_bsl=ensemble_out.p_bsl,
                p_ssl=ensemble_out.p_ssl,
                p_lateral=ensemble_out.p_lateral,
                predicted_class=ensemble_out.predicted_class,
                predicted_label=ensemble_out.predicted_label,
                confidence_qualifier=(
                    "HIGH" if max(ensemble_probas) >= CONFIDENCE_HIGH
                    else "MEDIUM" if max(ensemble_probas) >= CONFIDENCE_MEDIUM
                    else "LOW"
                ),
                max_prob=float(max(ensemble_probas)),
                adjusted=apply_session_weights,
                regime_code=regime_code,
            )
        else:
            rollback_check_target = model_a_out

        # Step 4: Rollback check
        should_roll, reason = self.should_rollback(
            rollback_check_target,
            model_b_out,
            version=version,
            price_momentum=price_momentum,
            volatility=volatility,
            regime_code=regime_code,
        )

        if should_roll:
            if ensemble_out is not None:
                ensemble_out.p_bsl = 0.0
                ensemble_out.p_ssl = 0.0
                ensemble_out.p_lateral = 1.0
                ensemble_out.predicted_class = 2
                ensemble_out.predicted_label = "LATERAL"

            rollback_check_target = self.apply_rollback(
                rollback_check_target, version=version
            )
            logger.info(f"Ensemble rollback triggered: {reason}")

        return InferenceResult(
            model_a=model_a_out,
            model_b=model_b_out,
            ensemble=ensemble_out,
            session_code=0,
            regime_code=regime_code,
            timestamp_ms=0,
            rollback_version=version,
        )
