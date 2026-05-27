# =============================================================================
# AGENTIC-QUANT — XGBoost Stacked Ensemble (Phase 6.3)
# Meta-learner: LogisticRegression
# Input: [p_bsl_xgb, p_ssl_xgb, p_lat_xgb, lstm_consensus, regime_code]
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from sklearn.linear_model import LogisticRegression
else:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        LogisticRegression = None  # type: ignore[assignment]


# =============================================================================
# Constants
# =============================================================================

# Meta-features dimension
N_META_FEATURES = 5  # [P_BSL_xgb, P_SSL_xgb, P_lat_xgb, lstm_consensus, regime_code]

# Default meta-learner params (scikit-learn 1.8+)
DEFAULT_META_PARAMS: dict[str, Any] = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 42,
}


@dataclass
class EnsembleConfig:
    """Configuration for XGBoost Stacked Ensemble.

    Attributes:
        meta_params: LogisticRegression parameters for meta-learner
        use_cv: Use cross-validation for meta-training (default: False)
        cv_folds: Number of CV folds (default: 5)
        model_path: Path to save/load ensemble
    """
    meta_params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_META_PARAMS))
    use_cv: bool = False
    cv_folds: int = 5
    model_path: str | Path = "models/xgboost/ensemble_meta.json"


class XGBoostEnsemble:
    """Stacked ensemble meta-learner combining XGBoost models with LSTM.

    Phan III.8 — Ensemble Architecture:
      Input: [p_bsl_xgb, p_ssl_xgb, p_lat_xgb, lstm_consensus, regime_code]
      Meta-learner: LogisticRegression (multinomial)

    The ensemble learns how to weight XGBoost predictions vs. LSTM consensus
    based on the current market regime. This allows the system to adapt
    which model to trust in different conditions.

    Session weight logic (Phan IV.1) is applied AFTER meta-prediction:
      - ACCUMULATION: P_lateral += (P_BSL + P_SSL) * 0.2
      - EXPANSION: confidence_qualifier -> HIGH if >= MEDIUM
      - REVERSAL_RISK: Bear Evidence weight += 10%
    """

    def __init__(self, config: EnsembleConfig | None = None) -> None:
        if LogisticRegression is None:
            raise ImportError(
                "scikit-learn is required for ensemble. "
                "Install with: pip install scikit-learn"
            )

        self._config = config or EnsembleConfig()
        self._meta_learner: LogisticRegression | None = None
        self._is_trained: bool = False
        self._training_accuracy: float = 0.0
        self._feature_names: list[str] = [
            "p_bsl_xgb",
            "p_ssl_xgb",
            "p_lat_xgb",
            "lstm_consensus",
            "regime_code",
        ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> EnsembleConfig:
        return self._config

    @property
    def meta_learner(self) -> LogisticRegression | None:
        return self._meta_learner

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def coefficients(self) -> dict[str, float] | None:
        """Get meta-learner coefficients per class.

        Returns:
            Dict mapping feature names to coefficient values,
            or None if not trained.
        """
        if self._meta_learner is None or not self._is_trained:
            return None

        coefs = self._meta_learner.coef_
        # coefs shape: (n_classes, n_features)
        # For multinomial, each class has its own coefficient vector
        class_names = ["BSL", "SSL", "LATERAL"]
        result: dict[str, float] = {}
        for cls_idx, cls_name in enumerate(class_names):
            for feat_idx, feat_name in enumerate(self._feature_names):
                result[f"{cls_name}/{feat_name}"] = float(coefs[cls_idx, feat_idx])
        return result

    # ------------------------------------------------------------------
    # Meta-feature builder
    # ------------------------------------------------------------------

    def build_meta_features(
        self,
        p_bsl_xgb: np.ndarray,
        p_ssl_xgb: np.ndarray,
        p_lat_xgb: np.ndarray,
        lstm_consensus: np.ndarray,
        regime_code: np.ndarray | int,
    ) -> np.ndarray:
        """Build meta-feature matrix for ensemble.

        Args:
            p_bsl_xgb: P_BSL from XGBoost Model A (n_samples,) or scalar
            p_ssl_xgb: P_SSL from XGBoost Model A (n_samples,) or scalar
            p_lat_xgb: P_lateral from XGBoost Model A (n_samples,) or scalar
            lstm_consensus: LSTM consensus score (n_samples,) or scalar
                           — typically softmax probability from LSTM
            regime_code: Market regime code (0-3) or array

        Returns:
            Meta-feature matrix (n_samples, 5)
        """
        # Ensure all are arrays
        p_bsl = np.asarray(p_bsl_xgb, dtype=np.float64).ravel()
        p_ssl = np.asarray(p_ssl_xgb, dtype=np.float64).ravel()
        p_lat = np.asarray(p_lat_xgb, dtype=np.float64).ravel()
        lstm = np.asarray(lstm_consensus, dtype=np.float64).ravel()
        regime = np.asarray(regime_code, dtype=np.float64).ravel()

        n_samples = max(
            p_bsl.shape[0], p_ssl.shape[0], p_lat.shape[0],
            lstm.shape[0], regime.shape[0],
        )

        # Broadcast scalars
        def _broadcast(arr: np.ndarray, n: int) -> np.ndarray:
            if arr.shape[0] == 1 and n > 1:
                return np.full(n, arr[0])
            return arr

        p_bsl = _broadcast(p_bsl, n_samples)
        p_ssl = _broadcast(p_ssl, n_samples)
        p_lat = _broadcast(p_lat, n_samples)
        lstm = _broadcast(lstm, n_samples)
        regime = _broadcast(regime, n_samples)

        meta_X = np.column_stack([p_bsl, p_ssl, p_lat, lstm, regime])
        return meta_X

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        meta_X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Train the meta-learner (LogisticRegression).

        Args:
            meta_X: Meta-feature matrix (n_samples, 5)
                   Columns: [p_bsl_xgb, p_ssl_xgb, p_lat_xgb, lstm_consensus, regime_code]
            y: True labels (n_samples,) — 0=BSL, 1=SSL, 2=LATERAL

        Returns:
            Training accuracy
        """
        if meta_X.shape[1] != N_META_FEATURES:
            raise ValueError(
                f"Meta-features must have {N_META_FEATURES} columns, "
                f"got {meta_X.shape[1]}"
            )

        self._meta_learner = LogisticRegression(**self._config.meta_params)

        logger.info(
            f"Training ensemble meta-learner: meta_X={meta_X.shape}, y={y.shape}, "
            f"classes={np.unique(y)}"
        )

        self._meta_learner.fit(meta_X, y)
        self._is_trained = True

        # Training accuracy
        y_pred = self._meta_learner.predict(meta_X)
        self._training_accuracy = float(np.mean(y_pred == y))

        logger.info(
            f"Ensemble trained — accuracy: {self._training_accuracy:.4f}, "
            f"coefs: {self.coefficients}"
        )

        return self._training_accuracy

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, meta_X: np.ndarray) -> np.ndarray:
        """Predict ensemble probabilities.

        Args:
            meta_X: Meta-feature matrix (n_samples, 5)

        Returns:
            Probability array (n_samples, 3) — [P_BSL, P_SSL, P_LATERAL]
        """
        if not self._is_trained or self._meta_learner is None:
            raise RuntimeError("Ensemble is not trained yet. Call fit() first.")

        if meta_X.ndim == 1:
            meta_X = meta_X.reshape(1, -1)

        return self._meta_learner.predict_proba(meta_X)

    def predict(self, meta_X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            meta_X: Meta-feature matrix (n_samples, 5)

        Returns:
            Predicted class indices (0=BSL, 1=SSL, 2=LATERAL)
        """
        if not self._is_trained or self._meta_learner is None:
            raise RuntimeError("Ensemble is not trained yet. Call fit() first.")

        if meta_X.ndim == 1:
            meta_X = meta_X.reshape(1, -1)

        return self._meta_learner.predict(meta_X)

    def predict_single(
        self,
        p_bsl_xgb: float,
        p_ssl_xgb: float,
        p_lat_xgb: float,
        lstm_consensus: float,
        regime_code: int,
    ) -> np.ndarray:
        """Predict ensemble for a single sample.

        Args:
            p_bsl_xgb: P_BSL from Model A
            p_ssl_xgb: P_SSL from Model A
            p_lat_xgb: P_lateral from Model A
            lstm_consensus: LSTM consensus score
            regime_code: Market regime code

        Returns:
            Probability array (3,) — [P_BSL, P_SSL, P_LATERAL]
        """
        meta_X = self.build_meta_features(
            np.array([p_bsl_xgb]),
            np.array([p_ssl_xgb]),
            np.array([p_lat_xgb]),
            np.array([lstm_consensus]),
            np.array([regime_code]),
        )
        return self.predict_proba(meta_X)[0]

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> None:
        """Save ensemble meta-learner."""
        if self._meta_learner is None:
            raise RuntimeError("No ensemble to save.")

        save_path = Path(path or self._config.model_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        import joblib
        joblib.dump(self._meta_learner, str(save_path))

        logger.info(f"Ensemble saved to {save_path}")

    def load(self, path: str | Path | None = None) -> None:
        """Load ensemble meta-learner."""
        load_path = Path(path or self._config.model_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Ensemble file not found: {load_path}")

        import joblib
        self._meta_learner = joblib.load(str(load_path))
        self._is_trained = True

        logger.info(f"Ensemble loaded from {load_path}")

    # ------------------------------------------------------------------
    # Ensemble interpretation
    # ------------------------------------------------------------------

    def get_feature_weights(self) -> dict[str, float]:
        """Get the learned weights of each meta-feature.

        Returns the mean absolute coefficient across all classes.

        Returns:
            Dict mapping feature name to importance weight
        """
        if self._meta_learner is None or not self._is_trained:
            return {name: 0.0 for name in self._feature_names}

        coefs = self._meta_learner.coef_
        # Mean absolute coefficient across classes
        mean_abs_coefs = np.mean(np.abs(coefs), axis=0)
        return {
            name: float(mean_abs_coefs[i])
            for i, name in enumerate(self._feature_names)
        }

    def get_regime_bias(self) -> dict[int, dict[str, float]]:
        """Get how the ensemble biases predictions per regime.

        Returns dict mapping regime_code to bias direction.
        """
        if self._meta_learner is None or not self._is_trained:
            return {}

        coefs = self._meta_learner.coef_
        class_names = ["BSL", "SSL", "LATERAL"]
        regime_idx = 4  # regime_code is the 5th feature

        result: dict[int, dict[str, float]] = {}
        for regime_code in range(4):
            regime_bias = {
                cls_name: float(coefs[cls_idx, regime_idx])
                for cls_idx, cls_name in enumerate(class_names)
            }
            result[regime_code] = regime_bias

        return result
