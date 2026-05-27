# =============================================================================
# AGENTIC-QUANT — XGBoost Model B (Phase 6.2)
# Binary Classifier: P_hold (probability a zone will hold)
# Cost-sensitive: scale_pos_weight for class imbalance
# theta* = 0.71 decision threshold
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    import xgboost as xgb
    from sklearn.metrics import classification_report
else:
    try:
        import xgboost as xgb
    except ImportError:
        xgb = None  # type: ignore[assignment]
    try:
        from sklearn.metrics import classification_report
    except ImportError:
        classification_report = None  # type: ignore[assignment]


# =============================================================================
# Constants
# =============================================================================

# theta* = optimal decision threshold (from backtesting calibration)
THETA_STAR = 0.71

# Default training params for binary classification
DEFAULT_MODEL_B_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "random_state": 42,
    "verbosity": 0,
    "n_jobs": -1,
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "auc"],
    "scale_pos_weight": 2.0,  # Default: weight positive class (hold=1)
    "use_label_encoder": False,
}


def theta_star_threshold(probs: np.ndarray, theta: float = THETA_STAR) -> np.ndarray:
    """Apply theta* threshold for binary decision.

    Phan III.8 — Model B threshold:
      theta* = 0.71 (optimized from backtesting)
      P_hold >= theta* -> zone giu (hold=1)
      P_hold < theta* -> zone khong giu (hold=0)

    Args:
        probs: Predicted probabilities (n_samples,)
        theta: Decision threshold (default: 0.71)

    Returns:
        Binary predictions (0 or 1)
    """
    return (probs >= theta).astype(np.int32)


# =============================================================================
# Metrics
# =============================================================================

def compute_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_thresholds: int = 100,
    beta: float = 1.0,
) -> tuple[float, float]:
    """Compute optimal decision threshold using F-beta score.

    Searches over [0.0, 1.0] to find threshold maximizing F-beta.
    Default beta=1.0 => F1 score.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        n_thresholds: Number of threshold candidates
        beta: F-beta parameter (default: 1.0 => F1)

    Returns:
        (optimal_threshold, max_fbeta)
    """
    thresholds = np.linspace(0.05, 0.95, n_thresholds)
    best_fbeta = -1.0
    best_threshold = THETA_STAR

    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(np.int32)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)

        beta_sq = beta ** 2
        fbeta = (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall + 1e-10)

        if fbeta > best_fbeta:
            best_fbeta = fbeta
            best_threshold = thresh

    return best_threshold, best_fbeta


def compute_positive_rate(y_pred: np.ndarray) -> float:
    """Compute rate of positive predictions (hold=1)."""
    return float(np.mean(y_pred))


# =============================================================================
# Model B: XGBoost Binary Classifier
# =============================================================================

@dataclass
class ModelBConfig:
    """Configuration for XGBoost Model B.

    Attributes:
        params: XGBoost training parameters (includes scale_pos_weight)
        theta_star: Decision threshold (default: 0.71)
        model_path: Path to save/load model
    """
    params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_MODEL_B_PARAMS))
    theta_star: float = THETA_STAR
    model_path: str | Path = "models/xgboost/model_b.json"


class XGBoostModelB:
    """XGBoost Model B — Binary Classifier for P_hold.

    Phan III.8 — Cost-sensitive classification:
      - scale_pos_weight balances hold vs. not-hold classes
      - theta* = 0.71 threshold (calibrated from backtesting)

    Predicts whether a zone will hold (survive) or be mitigated.
    """

    def __init__(self, config: ModelBConfig | None = None) -> None:
        if xgb is None:
            raise ImportError(
                "xgboost is required. Install with: pip install xgboost"
            )

        self._config = config or ModelBConfig()
        self._model: xgb.XGBClassifier | None = None
        self._is_trained: bool = False
        self._feature_importance: dict[str, float] = {}
        self._training_history: dict[str, list[float]] = {}
        self._optimal_threshold: float = self._config.theta_star
        self._class_balance: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> ModelBConfig:
        return self._config

    @property
    def model(self) -> xgb.XGBClassifier | None:
        return self._model

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def theta_star(self) -> float:
        return self._optimal_threshold

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        auto_weight: bool = True,
    ) -> dict[str, list[float]]:
        """Train the XGBoost Model B.

        Cost-sensitive: if auto_weight=True, compute scale_pos_weight
        from class distribution.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,) — 0=not_hold, 1=hold
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            auto_weight: Auto-compute scale_pos_weight from data

        Returns:
            Training history dict
        """
        params = dict(self._config.params)

        # Auto-compute scale_pos_weight from class distribution
        if auto_weight:
            n_neg = np.sum(y == 0)
            n_pos = np.sum(y == 1)
            if n_pos > 0 and n_neg > 0:
                scale_weight = n_neg / n_pos
                params["scale_pos_weight"] = scale_weight
                self._class_balance = {
                    "n_negative": int(n_neg),
                    "n_positive": int(n_pos),
                    "ratio": float(n_neg / n_pos) if n_pos > 0 else 0.0,
                    "scale_pos_weight": scale_weight,
                }
                logger.info(
                    f"Model B auto-weight: neg={n_neg}, pos={n_pos}, "
                    f"scale_pos_weight={scale_weight:.2f}"
                )

        self._model = xgb.XGBClassifier(**params)
        eval_set = None
        callbacks = None

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            if False:  # We use eval_metric from params
                pass

        logger.info(
            f"Training Model B: X={X.shape}, y={y.shape}, "
            f"classes={np.unique(y)}, scale_pos_weight={params.get('scale_pos_weight', 1.0)}"
        )

        self._model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False,
        )

        self._is_trained = True

        # Compute optimal threshold on training data
        y_pred_proba = self._model.predict_proba(X)[:, 1]
        optimal_thresh, optimal_fbeta = compute_optimal_threshold(y, y_pred_proba)
        self._optimal_threshold = optimal_thresh

        logger.info(
            f"Model B trained — optimal theta*={optimal_thresh:.4f} (F-beta={optimal_fbeta:.4f}), "
            f"default={self._config.theta_star}"
        )

        # Feature importance
        if hasattr(self._model, "feature_importances_"):
            self._feature_importance = {
                f"f{i}": float(v)
                for i, v in enumerate(self._model.feature_importances_)
            }

        # Store training history
        try:
            if self._model.evals_result():
                self._training_history = {
                    k: [float(vv) for vv in v]
                    for k, v in self._model.evals_result().items()
                }
        except Exception:
            self._training_history = {}

        return self._training_history

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict P_hold probability.

        Args:
            X: Feature vector (n_samples, n_features) or single (n_features,)

        Returns:
            Probability array (n_samples, 2) — [P_not_hold, P_hold]
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model B is not trained yet. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        return self._model.predict_proba(X)

    def predict_p_hold(self, X: np.ndarray) -> np.ndarray:
        """Predict P_hold probabilities directly.

        Args:
            X: Feature vector (n_samples, n_features)

        Returns:
            P_hold array (n_samples,) — probability that zone holds
        """
        probas = self.predict_proba(X)
        return probas[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class using theta* threshold.

        Args:
            X: Feature vector (n_samples, n_features)

        Returns:
            Binary predictions (0=not_hold, 1=hold)
        """
        p_hold = self.predict_p_hold(X)
        return theta_star_threshold(p_hold, self._optimal_threshold)

    def predict_with_confidence(self, X: np.ndarray) -> np.ndarray:
        """Predict with margin-of-victory confidence.

        Returns:
            Confidence scores: |P_hold - theta*| normalized
        """
        p_hold = self.predict_p_hold(X)
        confidence = np.abs(p_hold - self._optimal_threshold)
        # Normalize: 0.0 to 1.0 (max confidence = 1 - theta* = 0.29)
        confidence = confidence / max(self._optimal_threshold, 1.0 - self._optimal_threshold)
        return np.clip(confidence, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> None:
        """Save model to JSON file."""
        if self._model is None:
            raise RuntimeError("No model to save.")

        save_path = Path(path or self._config.model_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(save_path))

        # Save threshold alongside model
        meta_path = save_path.with_suffix(".meta.json")
        import json
        meta = {
            "theta_star": self._optimal_threshold,
            "class_balance": self._class_balance,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Model B saved to {save_path} (theta*={self._optimal_threshold:.4f})")

    def load(self, path: str | Path | None = None) -> None:
        """Load model from JSON file."""
        if xgb is None:
            raise ImportError("xgboost is required to load Model B")

        load_path = Path(path or self._config.model_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        self._model = xgb.XGBClassifier()
        self._model.load_model(str(load_path))
        self._is_trained = True

        # Load threshold from meta file
        meta_path = load_path.with_suffix(".meta.json")
        if meta_path.exists():
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            self._optimal_threshold = meta.get("theta_star", self._config.theta_star)
            self._class_balance = meta.get("class_balance", {})

        logger.info(f"Model B loaded from {load_path} (theta*={self._optimal_threshold:.4f})")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, Any]:
        """Evaluate model B performance.

        Returns:
            Dict with accuracy, precision, recall, F1, AUC, etc.
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model B is not trained yet.")

        y_pred_proba = self._model.predict_proba(X)[:, 1]
        y_pred = theta_star_threshold(y_pred_proba, self._optimal_threshold)

        # Confusion matrix
        tp = np.sum((y_pred == 1) & (y == 1))
        tn = np.sum((y_pred == 0) & (y == 0))
        fp = np.sum((y_pred == 1) & (y == 0))
        fn = np.sum((y_pred == 0) & (y == 1))

        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        positive_rate = float(np.mean(y_pred))
        true_positive_rate = recall

        result: dict[str, Any] = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "theta_star": self._optimal_threshold,
            "confusion_matrix": {
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
            },
            "positive_rate": float(positive_rate),
            "true_positive_rate": float(true_positive_rate),
            "n_samples": len(y),
            "class_distribution": {
                "n_hold": int(np.sum(y == 1)),
                "n_not_hold": int(np.sum(y == 0)),
            },
        }

        if classification_report is not None:
            try:
                result["classification_report"] = classification_report(
                    y, y_pred,
                    target_names=["not_hold", "hold"],
                    output_dict=True,
                )
            except Exception:
                pass

        return result
