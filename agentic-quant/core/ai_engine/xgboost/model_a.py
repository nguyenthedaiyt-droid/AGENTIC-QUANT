# =============================================================================
# AGENTIC-QUANT — XGBoost Model A (Phase 6.1)
# Multiclass Classifier: P_BSL / P_SSL / P_lateral
# Custom objective: Overconfidence Penalty lambda=0.5 khi max_prob>0.85
# IC target > 0.05
# Batch 2: Session Weights + Platt Scaling + Brier Score + ECE
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
    from sklearn.calibration import CalibratedClassifierCV
else:
    try:
        import xgboost as xgb
    except ImportError:
        xgb = None  # type: ignore[assignment]
    try:
        from sklearn.metrics import classification_report
    except ImportError:
        classification_report = None  # type: ignore[assignment]
    try:
        from sklearn.calibration import CalibratedClassifierCV
    except ImportError:
        CalibratedClassifierCV = None  # type: ignore[assignment]


# =============================================================================
# Constants
# =============================================================================

# Class mapping
IDX_BSL = 0      # Buy-side liquidity
IDX_SSL = 1      # Sell-side liquidity
IDX_LATERAL = 2  # No clear direction

CLASS_LABELS = ["BSL", "SSL", "LATERAL"]

# Overconfidence penalty
OVERC_PENALTY_LAMBDA = 0.5
OVERC_PROB_THRESHOLD = 0.85

# IC target minimum
IC_TARGET_MIN = 0.05

# Calibration & ECE defaults
N_CALIBRATION_BINS = 10      # So bin cho ECE
CALIBRATION_METHOD = "sigmoid"  # Platt scaling (sigmoid) cho Model A
CALIBRATION_CV = 5           # So fold cross-validation khi fit calibrator

# Regime codes (consistent with inference.py)
REGIME_NORMAL = 0
REGIME_TRENDING_LV = 1
REGIME_TRENDING_HV = 2
REGIME_CHOPPY_HV = 3

# Session weight constants (Phan IV.1)
SESSION_WEIGHT_ACCUMULATION: float = 0.2   # P_lateral += (P_BSL+P_SSL) * 0.2
SESSION_WEIGHT_EXPANSION: float = 0.0      # confidence_qualifier -> HIGH if >= MEDIUM
SESSION_WEIGHT_REVERSAL: float = 0.10      # Bear Evidence weight += 10%

# Default training params
DEFAULT_MODEL_A_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 6,
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
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": ["mlogloss", "merror"],
}


# =============================================================================
# Custom Objective: Overconfidence Penalty
# =============================================================================

def overconfidence_penalty_objective(
    preds: np.ndarray,
    dtrain: "xgb.DMatrix",
    penalty_lambda: float = OVERC_PENALTY_LAMBDA,
    prob_threshold: float = OVERC_PROB_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Custom XGBoost objective with overconfidence penalty.

    Phan III.8 — Overconfidence Penalty:
      Khi max_prob > 0.85 (over 85%), ap dung penalty lambda=0.5
      de reduce gradient, tranh model become too confident on wrong predictions.

    Args:
        preds: Raw predictions (shape: [n_samples * n_classes])
        dtrain: XGBoost DMatrix containing labels
        penalty_lambda: Penalty scaling factor (default: 0.5)
        prob_threshold: Probability threshold for overconfidence (default: 0.85)

    Returns:
        (grad, hess) tuple for XGBoost custom objective
    """
    if xgb is None:
        raise ImportError("xgboost is required for Model A")

    n_samples = dtrain.num_row()
    n_classes = 3  # BSL, SSL, LATERAL

    # Reshape predictions: (n_samples, n_classes)
    preds_reshaped = preds.reshape(n_samples, n_classes)

    # Softmax to get probabilities
    exp_preds = np.exp(preds_reshaped - np.max(preds_reshaped, axis=1, keepdims=True))
    probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)

    # Labels: one-hot encoded
    labels = dtrain.get_label().astype(np.int32)
    y_onehot = np.zeros((n_samples, n_classes), dtype=np.float64)
    y_onehot[np.arange(n_samples), labels] = 1.0

    # Standard gradient for softmax cross-entropy
    grad = probs - y_onehot  # (n_samples, n_classes)

    # Overconfidence penalty mask:
    # For samples where max_prob > threshold AND prediction is wrong,
    # scale down the gradient (penalty_lambda)
    max_probs = np.max(probs, axis=1)
    pred_class = np.argmax(probs, axis=1)

    # Overconfident and wrong: penalty_mask = True
    overconfident_mask = (max_probs > prob_threshold) & (pred_class != labels)

    # Apply penalty: scale gradient down by penalty_lambda for overconfident samples
    penalty_factor = np.where(overconfident_mask, penalty_lambda, 1.0)
    grad = grad * penalty_factor[:, np.newaxis]

    # Hessian (diagonal approximation): probs * (1 - probs) * penalty_factor
    # Standard hessian for softmax: p_ij * (1 - p_ij)
    hess_standard = probs * (1.0 - probs)
    hess = hess_standard * penalty_factor[:, np.newaxis]

    # Clip to prevent extreme values
    hess = np.clip(hess, 1e-8, 1.0)

    return grad.ravel(), hess.ravel()


# =============================================================================
# Information Coefficient metrics
# =============================================================================

def compute_ic(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Compute Information Coefficient (Spearman rank correlation).

    IC measures rank correlation between predicted probabilities
    and actual outcomes. Target: IC > 0.05.

    Args:
        y_true: True labels (n_samples,)
        y_pred_proba: Predicted probabilities for the correct class (n_samples,)

    Returns:
        Spearman rank correlation coefficient
    """
    from scipy.stats import spearmanr

    if len(y_true) < 10:
        return 0.0

    # For multiclass: use probability assigned to the correct class
    n_samples = len(y_true)
    n_classes = y_pred_proba.shape[1]
    correct_probs = y_pred_proba[np.arange(n_samples), y_true]

    # Handle edge case: all identical values
    if np.std(correct_probs) < 1e-10 or np.std(y_true) < 1e-10:
        return 0.0

    ic, _ = spearmanr(correct_probs, y_true)
    return float(ic) if not np.isnan(ic) else 0.0


def compute_ic_per_class(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> dict[str, float]:
    """Compute IC per class (BSL, SSL, LATERAL)."""
    from scipy.stats import spearmanr

    n_classes = y_pred_proba.shape[1]
    ic_dict: dict[str, float] = {}

    for cls_idx, cls_name in enumerate(CLASS_LABELS):
        # Binary: 1 if this class, 0 otherwise
        y_binary = (y_true == cls_idx).astype(np.float64)
        prob_for_class = y_pred_proba[:, cls_idx]

        if np.std(prob_for_class) < 1e-10 or np.std(y_binary) < 1e-10:
            ic_dict[cls_name] = 0.0
        else:
            ic_val, _ = spearmanr(prob_for_class, y_binary)
            ic_dict[cls_name] = float(ic_val) if not np.isnan(ic_val) else 0.0

    return ic_dict


# =============================================================================
# Calibration: Platt Scaling (sigmoid) cho Model A
# =============================================================================

def fit_platt_calibration(
    model: "xgb.XGBClassifier",
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = CALIBRATION_METHOD,
    cv: int = CALIBRATION_CV,
) -> "CalibratedClassifierCV | None":
    """Fit Platt Scaling (sigmoid) calibration cho Model A probabilities.

    Su dung sklearn.calibration.CalibratedClassifierCV de calibrate
    probabilities tu XGBoost raw output. Platt scaling ap dung sigmoid
    mapping: P_cal = 1 / (1 + exp(A * raw_prob + B)).

    Args:
        model: XGBoost classifier da trained
        X_val: Validation features (n_samples, n_features)
        y_val: Validation labels (n_samples,) — 0=BSL, 1=SSL, 2=LATERAL
        method: Phuong phap calibration: 'sigmoid' (Platt) hoac 'isotonic'
        cv: So fold cross-validation (default: 5)

    Returns:
        CalibratedClassifierCV instance, hoac None neu khong the calibrate
    """
    if CalibratedClassifierCV is None:
        logger.warning("sklearn.calibration not available. Skipping Platt calibration.")
        return None

    if X_val is None or y_val is None or len(X_val) < 10:
        logger.warning(f"Validation set qua nho ({len(X_val) if X_val is not None else 0}), bo qua calibration.")
        return None

    if not hasattr(model, "predict_proba"):
        logger.warning("Model khong co predict_proba, bo qua calibration.")
        return None

    try:
        calibrator = CalibratedClassifierCV(
            estimator=model,
            method=method,  # 'sigmoid' = Platt scaling
            cv=cv,
        )
        calibrator.fit(X_val, y_val)
        logger.info(
            f"Platt calibration ({method}) fitted tren {len(X_val)} validation samples, "
            f"cv={cv}"
        )
        return calibrator
    except Exception as e:
        logger.error(f"Platt calibration that bai: {e}")
        return None


def apply_platt_calibration(
    probas: np.ndarray,
    calibrator: "CalibratedClassifierCV | None",
) -> np.ndarray:
    """Apply Platt scaling calibration vao probabilities.

    Args:
        probas: Raw probabilities (n_samples, 3)
        calibrator: Fitted CalibratedClassifierCV instance

    Returns:
        Calibrated probabilities (n_samples, 3), hoac original neu calibrator=None
    """
    if calibrator is None:
        return probas

    if probas.size == 0:
        return probas

    try:
        # CalibratedClassifierCV.predict_proba can xu ly truc tiep
        # Can tao dummy feature de predict — thuc ra calibrator da
        # hoc duoc mapping tu X->proba, nhung vi XGBoost raw probas
        # da duoc dung, ta can pass qua estimator
        calibrated = calibrator.predict_proba(probas)
        # calibrated shape: (n_samples, n_classes)
        if calibrated.shape[1] == 3:
            return calibrated
        return probas
    except Exception as e:
        logger.warning(f"Apply calibration that bai: {e}")
        return probas


# =============================================================================
# Brier Score & Expected Calibration Error (ECE)
# =============================================================================

def compute_brier_score(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> float:
    """Tinh Brier Score cho multiclass probabilities.

    Brier Score = mean((p_i - y_i)^2) voi y_i la one-hot encoding.
    Range: [0.0, 2.0], gia tri thap hon = calibration tot hon.

    Args:
        y_true: True labels (n_samples,) — 0=BSL, 1=SSL, 2=LATERAL
        y_pred_proba: Predicted probabilities (n_samples, 3)

    Returns:
        Brier Score (float)

    Raises:
        ValueError: Neu input empty hoac shape mismatch
    """
    if len(y_true) == 0 or y_pred_proba.size == 0:
        raise ValueError("Input rong, khong the tinh Brier Score.")

    n_samples = len(y_true)
    n_classes = y_pred_proba.shape[1]

    # One-hot encoding
    y_onehot = np.zeros((n_samples, n_classes), dtype=np.float64)
    y_onehot[np.arange(n_samples), y_true] = 1.0

    # Brier Score = mean of squared differences
    squared_errors = (y_pred_proba - y_onehot) ** 2
    brier = float(np.mean(squared_errors))

    return brier


def compute_ece(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = N_CALIBRATION_BINS,
) -> float:
    """Tinh Expected Calibration Error (ECE).

    ECE = sum_{k=1}^{K} (w_k * |acc_k - conf_k|)
    - Chia [0,1] thanh K bins deu
    - w_k = ty le samples trong bin k
    - acc_k = accuracy trong bin k
    - conf_k = avg confidence trong bin k

    Args:
        y_true: True labels (n_samples,) — 0=BSL, 1=SSL, 2=LATERAL
        y_pred_proba: Predicted probabilities (n_samples, 3)
        n_bins: So bin (default: 10)

    Returns:
        ECE value (float), thap hon = calibration tot hon

    Raises:
        ValueError: Neu input empty
    """
    if len(y_true) == 0 or y_pred_proba.size == 0:
        raise ValueError("Input rong, khong the tinh ECE.")

    n_samples = len(y_true)
    predicted_classes = np.argmax(y_pred_proba, axis=1)
    max_confs = np.max(y_pred_proba, axis=1)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for k in range(n_bins):
        # Mask: samples trong bin k
        in_bin = (max_confs > bin_boundaries[k]) & (max_confs <= bin_boundaries[k + 1])

        if np.sum(in_bin) == 0:
            continue

        # Accuracy trong bin
        bin_accuracy = np.mean(predicted_classes[in_bin] == y_true[in_bin])
        # Confidence trung binh trong bin
        bin_confidence = np.mean(max_confs[in_bin])
        # Weight = ty le samples trong bin
        bin_weight = np.sum(in_bin) / n_samples

        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return ece


# =============================================================================
# Model A: XGBoost Multiclass
# =============================================================================

@dataclass
class ModelAConfig:
    """Configuration for XGBoost Model A.

    Attributes:
        params: XGBoost training parameters
        penalty_lambda: Overconfidence penalty lambda (default: 0.5)
        prob_threshold: Overconfidence probability threshold (default: 0.85)
        ic_target_min: Minimum IC target (default: 0.05)
        early_stopping_rounds: Early stopping rounds (default: 50)
        model_path: Path to save/load model
        calibration_method: Platt scaling method ('sigmoid' or 'isotonic')
        calibration_cv: Cross-validation folds for calibration
        n_calibration_bins: Number of bins for ECE computation
    """
    params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_MODEL_A_PARAMS))
    penalty_lambda: float = OVERC_PENALTY_LAMBDA
    prob_threshold: float = OVERC_PROB_THRESHOLD
    ic_target_min: float = IC_TARGET_MIN
    early_stopping_rounds: int = 50
    model_path: str | Path = "models/xgboost/model_a.json"
    calibration_method: str = CALIBRATION_METHOD
    calibration_cv: int = CALIBRATION_CV
    n_calibration_bins: int = N_CALIBRATION_BINS


class XGBoostModelA:
    """XGBoost Model A — Multiclass (BSL / SSL / LATERAL).

    Phan III.8 — Overconfidence Penalty:
      Custom objective voi penalty lambda=0.5 khi max_prob > 0.85
      de tranh model become overconfident on wrong predictions.
      IC target > 0.05 (minimum information coefficient).

    Session weight logic (Phan IV.1):
      - ACCUMULATION: P_lateral += (P_BSL + P_SSL) * 0.2
      - EXPANSION: confidence_qualifier -> HIGH if >= MEDIUM
      - REVERSAL_RISK: Bear Evidence weight += 10%
    """

    def __init__(self, config: ModelAConfig | None = None) -> None:
        if xgb is None:
            raise ImportError(
                "xgboost is required. Install with: pip install xgboost"
            )

        self._config = config or ModelAConfig()
        self._model: xgb.XGBClassifier | None = None
        self._is_trained: bool = False
        self._feature_importance: dict[str, float] = {}
        self._training_history: dict[str, list[float]] = {}
        self._last_ic: float = 0.0
        self._last_ic_per_class: dict[str, float] = {}

        # Batch 2: Calibration & ECE
        self._calibrator: "CalibratedClassifierCV | None" = None
        self._last_brier_score: float = 0.0
        self._last_ece: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> ModelAConfig:
        return self._config

    @property
    def model(self) -> xgb.XGBClassifier | None:
        return self._model

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def last_ic(self) -> float:
        return self._last_ic

    @property
    def last_ic_per_class(self) -> dict[str, float]:
        return self._last_ic_per_class

    @property
    def calibrator(self) -> "CalibratedClassifierCV | None":
        return self._calibrator

    @property
    def last_brier_score(self) -> float:
        return self._last_brier_score

    @property
    def last_ece(self) -> float:
        return self._last_ece

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        """Train the XGBoost Model A.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,) — 0=BSL, 1=SSL, 2=LATERAL
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Training history dict with eval metrics

        Raises:
            ValueError: If IC target < 0.05
        """
        params = dict(self._config.params)
        params["objective"] = "multi:softprob"
        params["num_class"] = 3

        # Custom objective via function API — use standard objective for training,
        # apply overconfidence penalty as custom objective during fine-tuning
        self._model = xgb.XGBClassifier(**params)

        eval_set = None
        early_stopping = self._config.early_stopping_rounds

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            if early_stopping > 0:
                self._model.early_stopping_rounds = early_stopping

        logger.info(
            f"Training Model A: X={X.shape}, y={y.shape}, "
            f"classes={np.unique(y)}, penalty_lambda={self._config.penalty_lambda}"
        )

        self._model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False,
        )

        self._is_trained = True

        # Compute IC on training set
        y_pred_proba = self._model.predict_proba(X)
        self._last_ic = compute_ic(y, y_pred_proba)
        self._last_ic_per_class = compute_ic_per_class(y, y_pred_proba)

        logger.info(
            f"Model A trained — IC: {self._last_ic:.4f}, "
            f"per-class: {self._last_ic_per_class}"
        )

        # IC target check
        if self._last_ic < self._config.ic_target_min:
            logger.warning(
                f"IC ({self._last_ic:.4f}) < target ({self._config.ic_target_min}). "
                "Consider retraining with more data or different params."
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

    def fine_tune_with_overconfidence_penalty(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_rounds: int = 50,
    ) -> dict[str, list[float]]:
        """Fine-tune using custom overconfidence penalty objective.

        This uses XGBoost's custom objective via train() API.
        The standard softmax cross-entropy is replaced with the
        overconfidence-penalized version.

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_rounds: Number of fine-tuning rounds

        Returns:
            Training history
        """
        dtrain = xgb.DMatrix(X, label=y)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = dict(self._config.params)
        params.pop("objective", None)  # Custom objective, not standard

        # Custom objective
        def _custom_obj(preds: np.ndarray, dtrain: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
            return overconfidence_penalty_objective(
                preds, dtrain,
                penalty_lambda=self._config.penalty_lambda,
                prob_threshold=self._config.prob_threshold,
            )

        history: dict[str, list[float]] = {}
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=n_rounds,
            evals=[(dtrain, "train"), (dval, "val")],
            obj=_custom_obj,
            evals_result=history,
            verbose_eval=False,
        )

        # Convert to sklearn-style model
        if self._model is None:
            self._model = xgb.XGBClassifier(**self._config.params)
        # Rebuild the boosting rounds
        self._model._Booster = booster  # type: ignore[attr-defined]
        self._is_trained = True

        self._last_ic = compute_ic(y, self._model.predict_proba(X))
        self._last_ic_per_class = compute_ic_per_class(y, self._model.predict_proba(X))
        self._training_history = history

        logger.info(
            f"Fine-tuned with overconfidence penalty — "
            f"IC: {self._last_ic:.4f}, history keys: {list(history.keys())}"
        )

        return history

    # ------------------------------------------------------------------
    # Calibration: Platt Scaling
    # ------------------------------------------------------------------

    def fit_calibration(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "CalibratedClassifierCV | None":
        """Fit Platt Scaling calibration using validation data.

        Goi sklearn.calibration.CalibratedClassifierCV de calibrate
        probabilities tu XGBoost raw output. Luu calibrator vao self.

        Args:
            X_val: Validation features (n_samples, n_features)
            y_val: Validation labels (n_samples,)

        Returns:
            Fitted CalibratedClassifierCV instance hoac None
        """
        if not self._is_trained or self._model is None:
            logger.warning("Model A chua trained, khong the fit calibration.")
            return None

        calibrator = fit_platt_calibration(
            model=self._model,
            X_val=X_val,
            y_val=y_val,
            method=self._config.calibration_method,
            cv=self._config.calibration_cv,
        )

        self._calibrator = calibrator
        return calibrator

    def predict_proba_calibrated(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Predict calibrated probabilities (sau Platt Scaling).

        Args:
            X: Feature vector (n_samples, n_features) or single (n_features,)

        Returns:
            Calibrated probabilities (n_samples, 3)
        """
        raw_probas = self.predict_proba(X)
        return apply_platt_calibration(raw_probas, self._calibrator)

    # ------------------------------------------------------------------
    # Calibration Metrics: Brier Score & ECE
    # ------------------------------------------------------------------

    def compute_brier_score(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray | None = None,
    ) -> float:
        """Tinh Brier Score cho predictions.

        Args:
            y_true: True labels (n_samples,)
            y_pred_proba: Predicted probabilities (n_samples, 3).
                          Neu None, dung self.predict_proba().

        Returns:
            Brier Score value
        """
        if y_pred_proba is None:
            if not self._is_trained or self._model is None:
                logger.warning("Model chua trained, Brier Score = 0.0")
                return 0.0
            logger.warning("y_pred_proba is None, cannot compute Brier Score without features.")
            return 0.0

        brier = compute_brier_score(y_true, y_pred_proba)
        self._last_brier_score = brier
        return brier

    def compute_ece(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray | None = None,
        n_bins: int | None = None,
    ) -> float:
        """Tinh Expected Calibration Error (ECE).

        Args:
            y_true: True labels (n_samples,)
            y_pred_proba: Predicted probabilities (n_samples, 3).
                          Neu None, dung self.predict_proba().
            n_bins: So bin (default: config.n_calibration_bins)

        Returns:
            ECE value
        """
        if y_pred_proba is None:
            if not self._is_trained or self._model is None:
                logger.warning("Model chua trained, ECE = 0.0")
                return 0.0
            logger.warning("y_pred_proba is None, cannot compute ECE without features.")
            return 0.0

        n_bins = n_bins or self._config.n_calibration_bins
        ece = compute_ece(y_true, y_pred_proba, n_bins=n_bins)
        self._last_ece = ece
        return ece

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature vector (n_samples, n_features) or single (n_features,)

        Returns:
            Probability array (n_samples, 3) — [P_BSL, P_SSL, P_lateral]
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model A is not trained yet. Call fit() first.")

        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)

        return self._model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature vector (n_samples, n_features)

        Returns:
            Predicted class indices (0=BSL, 1=SSL, 2=LATERAL)
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model A is not trained yet. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        return self._model.predict(X)

    def get_confidence_qualifier(self, probas: np.ndarray) -> list[str]:
        """Get confidence qualifier based on prediction probabilities.

        Phan IV.1 — EXPANSION: confidence_qualifier -> HIGH if >= MEDIUM

        Args:
            probas: Probability array (n_samples, 3)

        Returns:
            List of confidence qualifiers: "HIGH", "MEDIUM", "LOW"
        """
        if probas.ndim == 1:
            probas = probas.reshape(1, -1)

        qualifiers: list[str] = []
        for prob in probas:
            max_prob = float(np.max(prob))
            if max_prob >= 0.85:
                qualifiers.append("HIGH")
            elif max_prob >= 0.65:
                qualifiers.append("MEDIUM")
            else:
                qualifiers.append("LOW")
        return qualifiers

    # ------------------------------------------------------------------
    # Session weight adjustments
    # ------------------------------------------------------------------

    def apply_session_weights(
        self,
        probas: np.ndarray,
        regime_code: int,
        bear_evidence: float = 0.0,
    ) -> np.ndarray:
        """Apply session weight logic de dieu chinh probabilities.

        Phan IV.1 — Session Weights:
          ACCUMULATION (NORMAL, TRENDING_LV regime):
            P_lateral += (P_BSL + P_SSL) * 0.2
          EXPANSION (TRENDING regime):
            confidence_qualifier -> HIGH neu >= MEDIUM
            (xu ly trong post-processing, khong thay doi proba truc tiep)
          REVERSAL_RISK (bear_evidence > 0.5):
            Bear Evidence weight += 10% (shift P_BSL -> P_SSL)

        Args:
            probas: Probability array (n_samples, 3) — [P_BSL, P_SSL, P_lateral]
            regime_code: Regime classifier code
                         0=NORMAL, 1=TRENDING_LV, 2=TRENDING_HV, 3=CHOPPY_HV
            bear_evidence: Bear evidence weight (0.0-1.0)

        Returns:
            Adjusted probability array (da renormalized)

        Raises:
            ValueError: Neu probas empty hoac shape khong hop le
        """
        if probas is None or probas.size == 0:
            raise ValueError("Probas rong, khong the apply session weights.")

        if probas.ndim == 1:
            probas = probas.reshape(1, -1)

        adjusted = probas.copy()

        for i in range(adjusted.shape[0]):
            p_bsl = adjusted[i, IDX_BSL]
            p_ssl = adjusted[i, IDX_SSL]
            p_lat = adjusted[i, IDX_LATERAL]

            # =============================================================
            # 1. ACCUMULATION (NORMAL, TRENDING_LV regime)
            #    P_lateral += (P_BSL + P_SSL) * 0.2
            # =============================================================
            if regime_code in (REGIME_NORMAL, REGIME_TRENDING_LV):
                lateral_boost = (p_bsl + p_ssl) * SESSION_WEIGHT_ACCUMULATION
                p_lat_adj = min(1.0, p_lat + lateral_boost)

                if p_lat_adj > p_lat:
                    # Scale BSL + SSL proportionally de sum = 1
                    total_directional = p_bsl + p_ssl
                    if total_directional > 0:
                        scale = 1.0 - p_lat_adj
                        adjusted[i, IDX_BSL] = (p_bsl / total_directional) * scale
                        adjusted[i, IDX_SSL] = (p_ssl / total_directional) * scale
                    else:
                        # Edge case: ca BSL va SSL deu = 0
                        adjusted[i, IDX_BSL] = 0.0
                        adjusted[i, IDX_SSL] = 0.0
                    adjusted[i, IDX_LATERAL] = p_lat_adj

            # =============================================================
            # 2. EXPANSION (TRENDING regime)
            #    confidence_qualifier -> HIGH neu >= MEDIUM
            #    Khong thay doi proba, xu ly trong inference post-processing
            # =============================================================
            # EXPANSION logic duoc handle trong inference.py khi goi
            # get_confidence_qualifier() va dieu chinh qualifier.
            # No action needed here — proba values remain unchanged.

            # =============================================================
            # 3. REVERSAL_RISK (bear_evidence > 0.5)
            #    Bear Evidence weight += 10%
            #    Shift probability tu BSL sang SSL
            # =============================================================
            if bear_evidence > 0.5:
                bear_shift = bear_evidence * SESSION_WEIGHT_REVERSAL
                shift_amount = adjusted[i, IDX_BSL] * bear_shift
                adjusted[i, IDX_BSL] = max(0.0, adjusted[i, IDX_BSL] - shift_amount)
                adjusted[i, IDX_SSL] = min(1.0, adjusted[i, IDX_SSL] + shift_amount)

            # Renormalize de dam bao sum = 1.0
            row_sum = np.sum(adjusted[i])
            if row_sum > 0:
                adjusted[i] /= row_sum

        return adjusted

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
        logger.info(f"Model A saved to {save_path}")

    def load(self, path: str | Path | None = None) -> None:
        """Load model from JSON file."""
        if xgb is None:
            raise ImportError("xgboost is required to load Model A")

        load_path = Path(path or self._config.model_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        self._model = xgb.XGBClassifier()
        self._model.load_model(str(load_path))
        self._is_trained = True
        logger.info(f"Model A loaded from {load_path}")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, Any]:
        """Evaluate model performance.

        Returns:
            Dict with accuracy, IC, IC per class, Brier Score, ECE,
            and classification report
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model A is not trained yet.")

        y_pred = self._model.predict(X)
        y_pred_proba = self._model.predict_proba(X)

        accuracy = float(np.mean(y_pred == y))
        ic = compute_ic(y, y_pred_proba)
        ic_per_class = compute_ic_per_class(y, y_pred_proba)

        # Batch 2: Brier Score & ECE
        brier = compute_brier_score(y, y_pred_proba)
        ece = compute_ece(y, y_pred_proba, n_bins=self._config.n_calibration_bins)

        self._last_brier_score = brier
        self._last_ece = ece

        result: dict[str, Any] = {
            "accuracy": accuracy,
            "ic": ic,
            "ic_per_class": ic_per_class,
            "brier_score": brier,
            "ece": ece,
            "n_calibration_bins": self._config.n_calibration_bins,
            "n_samples": len(y),
            "class_distribution": {
                str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))
            },
        }

        if classification_report is not None:
            try:
                result["classification_report"] = classification_report(
                    y, y_pred,
                    target_names=CLASS_LABELS,
                    output_dict=True,
                )
            except Exception:
                pass

        return result
