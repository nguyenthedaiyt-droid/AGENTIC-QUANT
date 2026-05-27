#!/usr/bin/env python3
"""
train_model_b.py — Train XGBoost Model B (Phase 6.2)

Chức năng:
  - Load pre-built X_B.parquet + y_B.parquet từ data/xgboost/
  - Cost-sensitive training với scale_pos_weight auto-tuning
  - theta* tuning (optimal decision threshold)
  - Isotonic Regression calibration fit
  - ECE per regime check
  - Save: models/xgboost/model_b.json + model_b_calibrated.pkl + model_b_metrics.json

Usage:
    python scripts/train_model_b.py --data-dir data/xgboost --output-dir models/xgboost

Author: AGENTIC-QUANT Team
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

try:
    import xgboost as xgb
except ImportError:
    xgb = None  # type: ignore[assignment]

try:
    import joblib
except ImportError:
    joblib = None  # type: ignore[assignment]

from core.ai_engine.xgboost.model_b import (
    XGBoostModelB,
    ModelBConfig,
    compute_optimal_threshold,
    compute_ece_binary,
    compute_ece_per_regime,
    fit_isotonic_calibration,
    apply_isotonic_calibration,
    theta_star_threshold,
    DEFAULT_MODEL_B_PARAMS,
)


# =============================================================================
# Constants
# =============================================================================

X_B_FILENAME: str = "X_B.parquet"
Y_B_FILENAME: str = "y_B.parquet"
REGIME_FILENAME: str = "regime_codes.parquet"

MODEL_JSON: str = "model_b.json"
CALIBRATED_PKL: str = "model_b_calibrated.pkl"
METRICS_JSON: str = "model_b_metrics.json"

# Default training params
DEFAULT_PARAMS: dict[str, Any] = dict(DEFAULT_MODEL_B_PARAMS)

# Regime names
REGIME_NAMES: dict[int, str] = {0: "NORMAL", 1: "TRENDING_LV", 2: "TRENDING_HV", 3: "CHOPPY_HV"}


# =============================================================================
# Data Loading
# =============================================================================

def load_data(
    data_dir: Path,
    split: str = "train",
    load_regime: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load X_B, y_B, và optional regime codes từ parquet files.

    Args:
        data_dir: Thư mục chứa X_B.parquet và y_B.parquet
        split: Split name (train, val, test)
        load_regime: Nếu True, cũng load regime_codes.parquet

    Returns:
        (X, y) hoặc (X, y, regime_codes) tuple

    Raises:
        FileNotFoundError: Nếu không tìm thấy file
    """
    # Try split-specific directory first
    x_path = data_dir / split / X_B_FILENAME
    y_path = data_dir / split / Y_B_FILENAME
    r_path = data_dir / split / REGIME_FILENAME if load_regime else None

    if not x_path.exists() or not y_path.exists():
        x_path = data_dir / X_B_FILENAME
        y_path = data_dir / Y_B_FILENAME
        if load_regime:
            r_path = data_dir / REGIME_FILENAME

    if not x_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy X_B: {x_path}. "
            f"Các files trong {data_dir.resolve()}: {list(data_dir.iterdir())}"
        )
    if not y_path.exists():
        raise FileNotFoundError(f"Không tìm thấy y_B: {y_path}")

    logger.info(f"Đang load X_B: {x_path}")
    X_df = pd.read_parquet(x_path)
    logger.info(f"Đang load y_B: {y_path}")
    y_df = pd.read_parquet(y_path)

    X = X_df.values.astype(np.float64)
    y = y_df.values.ravel().astype(np.int32)

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    logger.info(f"  X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"  Class distribution: hold={n_pos}, not_hold={n_neg} "
                f"(ratio={n_neg / max(n_pos, 1):.2f})")

    if load_regime and r_path is not None and r_path.exists():
        regime_df = pd.read_parquet(r_path)
        regime_codes = regime_df.values.ravel().astype(np.int32)
        logger.info(f"  Regime codes shape: {regime_codes.shape}")
        logger.info(f"  Regime distribution: {dict(zip(*np.unique(regime_codes, return_counts=True)))}")
        return X, y, regime_codes

    return X, y


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_all_metrics(
    model: XGBoostModelB,
    X: np.ndarray,
    y: np.ndarray,
    regime_codes: np.ndarray | None = None,
    label: str = "train",
) -> dict[str, Any]:
    """Tính toán tất cả metrics cho Model B evaluation.

    Args:
        model: Trained XGBoostModelB instance
        X: Feature matrix
        y: True binary labels
        regime_codes: Regime codes (optional, for ECE per regime)
        label: Label cho logging

    Returns:
        Dict chứa các metrics
    """
    y_pred_proba = model.predict_p_hold(X)
    y_pred = model.predict(X)

    # Confusion matrix
    tp = int(np.sum((y_pred == 1) & (y == 1)))
    tn = int(np.sum((y_pred == 0) & (y == 0)))
    fp = int(np.sum((y_pred == 1) & (y == 0)))
    fn = int(np.sum((y_pred == 0) & (y == 1)))

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    positive_rate = float(np.mean(y_pred))
    brier = float(np.mean((y_pred_proba - y.astype(np.float64)) ** 2))
    ece = compute_ece_binary(y, y_pred_proba)

    metrics: dict[str, Any] = {
        f"accuracy_{label}": float(accuracy),
        f"precision_{label}": float(precision),
        f"recall_{label}": float(recall),
        f"f1_{label}": float(f1),
        f"brier_score_{label}": brier,
        f"ece_{label}": ece,
        f"positive_rate_{label}": positive_rate,
        f"n_samples_{label}": len(y),
        f"confusion_matrix_{label}": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }

    logger.info(f"  [{label}] accuracy={accuracy:.4f}, precision={precision:.4f}, "
                f"recall={recall:.4f}, F1={f1:.4f}")
    logger.info(f"  [{label}] Brier={brier:.4f}, ECE={ece:.4f}, "
                f"pos_rate={positive_rate:.4f}")

    # ECE per regime
    if regime_codes is not None:
        ece_regime = compute_ece_per_regime(
            y_true=y,
            y_pred_proba=y_pred_proba,
            regime_codes=regime_codes,
        )
        metrics[f"ece_per_regime_{label}"] = ece_regime
        logger.info(f"  [{label}] ECE per regime: {ece_regime.get('ece_per_regime', {})}")
        logger.info(f"  [{label}] All regimes below target: {ece_regime.get('all_below_target', 'N/A')}")

    return metrics


# =============================================================================
# Main Training Pipeline
# =============================================================================

def run_training(
    data_dir: Path,
    output_dir: Path,
    params_override: dict[str, Any] | None = None,
    theta_star: float = 0.71,
    auto_weight: bool = True,
    calibration_out_of_bounds: str = "clip",
    ece_target: float = 0.08,
) -> None:
    """Run the full Model B training pipeline.

    Args:
        data_dir: Thư mục chứa dataset parquet files
        output_dir: Thư mục lưu model outputs
        params_override: Override các XGBoost params
        theta_star: Initial decision threshold
        auto_weight: Auto-compute scale_pos_weight
        calibration_out_of_bounds: Out-of-bounds handling cho Isotonic Regression
        ece_target: Target ECE per regime
    """
    logger.info("=" * 60)
    logger.info("MODEL B TRAINING PIPELINE")
    logger.info("=" * 60)

    # --- Load data ---
    logger.info("[1/6] Đang load dataset...")
    X_train, y_train = load_data(data_dir, split="train")
    X_val, y_val = load_data(data_dir, split="val")
    X_test, y_test, regime_test = load_data(data_dir, split="test", load_regime=True)

    # --- Build config ---
    logger.info("[2/6] Đang cấu hình model...")
    params = dict(DEFAULT_PARAMS)
    if params_override:
        params.update(params_override)

    config = ModelBConfig(
        params=params,
        theta_star=theta_star,
        model_path=str(output_dir / MODEL_JSON),
        calibration_out_of_bounds=calibration_out_of_bounds,
        ece_target=ece_target,
    )

    model = XGBoostModelB(config=config)
    logger.info(f"  Params: {params}")
    logger.info(f"  Theta* initial: {theta_star}")

    # --- Train model (cost-sensitive) ---
    logger.info("[3/6] Đang train XGBoost Model B (cost-sensitive)...")
    model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        auto_weight=auto_weight,
    )
    logger.info(f"  Optimal theta*: {model.theta_star:.4f}")
    logger.info(f"  Class balance: {model._class_balance}")

    # --- Evaluate ---
    logger.info("[4/6] Đang evaluate model...")
    metrics: dict[str, Any] = {}

    train_metrics = compute_all_metrics(model, X_train, y_train, label="train")
    val_metrics = compute_all_metrics(model, X_val, y_val, label="val")
    test_metrics = compute_all_metrics(
        model, X_test, y_test, regime_codes=regime_test, label="test"
    )

    metrics.update(train_metrics)
    metrics.update(val_metrics)
    metrics.update(test_metrics)
    metrics["theta_star_optimal"] = model.theta_star
    metrics["theta_star_initial"] = theta_star
    metrics["params"] = params

    # --- Calibration: Isotonic Regression ---
    logger.info("[5/6] Đang fit Isotonic Regression calibration...")
    isotonic = model.fit_isotonic_calibration(X_val, y_val)
    isotonic_path: Path | None = None
    if isotonic is not None:
        isotonic_path = output_dir / CALIBRATED_PKL
        isotonic_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(isotonic, str(isotonic_path))
        logger.info(f"  Isotonic calibrator saved: {isotonic_path}")

        # Evaluate calibrated probabilities on test set
        calib_p_hold = apply_isotonic_calibration(
            model.predict_p_hold(X_test), isotonic
        )
        calib_brier = float(np.mean((calib_p_hold - y_test.astype(np.float64)) ** 2))
        calib_ece = compute_ece_binary(y_test, calib_p_hold)
        metrics["calibrated_brier_score_test"] = calib_brier
        metrics["calibrated_ece_test"] = calib_ece

        logger.info(f"  Calibrated Test Brier={calib_brier:.4f}, ECE={calib_ece:.4f}")

        # ECE per regime sau calibration
        if regime_test is not None:
            calib_ece_regime = compute_ece_per_regime(
                y_true=y_test,
                y_pred_proba=calib_p_hold,
                regime_codes=regime_test,
                target_ece=ece_target,
            )
            metrics["calibrated_ece_per_regime_test"] = calib_ece_regime
            logger.info(f"  Calibrated ECE per regime: {calib_ece_regime.get('ece_per_regime', {})}")

        # Apply calibrated threshold
        calib_thresh, calib_fbeta = compute_optimal_threshold(y_val, apply_isotonic_calibration(model.predict_p_hold(X_val), isotonic))
        metrics["calibrated_theta_star"] = calib_thresh
        metrics["calibrated_fbeta"] = calib_fbeta
        logger.info(f"  Calibrated optimal theta*: {calib_thresh:.4f} (F-beta={calib_fbeta:.4f})")

    # --- ECE per regime check ---
    logger.info("[6/6] Đang kiểm tra ECE per regime...")
    ece_check = model.compute_ece_per_regime(X_test, y_test, regime_test, calibrated=True)
    metrics["ece_check"] = ece_check
    if ece_check.get("all_below_target", False):
        logger.success("  ✅ All regimes below ECE target!")
    else:
        logger.warning(
            f"  ⚠️  Some regimes above ECE target ({ece_target}): "
            f"{ece_check.get('regimes_above_target', [])}"
        )

    # --- Save model ---
    logger.info("Đang lưu model...")
    model.save(output_dir / MODEL_JSON)
    metrics["model_path"] = str(output_dir / MODEL_JSON)
    if isotonic_path:
        metrics["calibrator_path"] = str(isotonic_path)

    # --- Save metrics ---
    metrics_path = output_dir / METRICS_JSON
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved: {metrics_path}")

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("HOÀN THÀNH MODEL B TRAINING")
    logger.info("=" * 60)
    logger.info(f"  Model: {output_dir / MODEL_JSON}")
    logger.info(f"  Isotonic: {isotonic_path or 'N/A'}")
    logger.info(f"  Metrics: {metrics_path}")
    logger.info(f"  Theta* optimal: {model.theta_star:.4f}")
    logger.info(f"  Test F1: {test_metrics.get('f1_test', 'N/A'):.4f}")
    logger.info(f"  Test Brier: {test_metrics.get('brier_score_test', 'N/A'):.4f}")
    logger.info(f"  Test ECE: {test_metrics.get('ece_test', 'N/A'):.4f}")

    if ece_check.get("all_below_target", False):
        logger.success("  ✅ All regimes ECE < target!")
    else:
        logger.warning("  ⚠️  Some regimes above ECE target!")


# =============================================================================
# CLI
# =============================================================================

def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Train XGBoost Model B (Phase 6.2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/xgboost",
        help="Thư mục chứa X_B.parquet và y_B.parquet (default: data/xgboost)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/xgboost",
        help="Thư mục lưu model outputs (default: models/xgboost)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Số trees (default: 300)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Max tree depth (default: 4)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate (default: 0.05)",
    )
    parser.add_argument(
        "--theta-star",
        type=float,
        default=0.71,
        help="Initial decision threshold (default: 0.71)",
    )
    parser.add_argument(
        "--scale-pos-weight",
        type=float,
        default=None,
        help="scale_pos_weight (default: auto từ class distribution)",
    )
    parser.add_argument(
        "--ece-target",
        type=float,
        default=0.08,
        help="Target ECE per regime (default: 0.08)",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="Subsample ratio (default: 0.8)",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.7,
        help="Colsample by tree (default: 0.7)",
    )
    parser.add_argument(
        "--no-auto-weight",
        action="store_true",
        help="Tắt auto-compute scale_pos_weight",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    return parser.parse_args(args)


def main() -> None:
    """Entry point cho train_model_b script."""
    args = parse_args()

    # Cấu hình logging
    logger.remove()
    logger.add(sys.stderr, level=args.log_level, format="<level>{level: <8}</level> | {message}")

    base_dir = Path(__file__).resolve().parent.parent

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = base_dir / data_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir

    # Build params override từ CLI args
    params_override: dict[str, Any] = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
    }
    if args.scale_pos_weight is not None:
        params_override["scale_pos_weight"] = args.scale_pos_weight

    try:
        run_training(
            data_dir=data_dir,
            output_dir=output_dir,
            params_override=params_override,
            theta_star=args.theta_star,
            auto_weight=not args.no_auto_weight,
            ece_target=args.ece_target,
        )
    except Exception as e:
        logger.error(f"Model B training thất bại: {e}")
        raise


if __name__ == "__main__":
    main()
