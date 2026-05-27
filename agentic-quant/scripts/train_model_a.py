#!/usr/bin/env python3
"""
train_model_a.py — Train XGBoost Model A (Phase 6.1)

Chức năng:
  - Load pre-built X_A.parquet + y_A.parquet từ data/xgboost/
  - Custom objective (overconfidence penalty) từ model_a.py
  - Train XGBoost classifier, eval IC, Brier Score, ECE
  - Platt Scaling fit trên validation set
  - Save: models/xgboost/model_a.json + model_a_calibrated.pkl + model_a_metrics.json

Usage:
    python scripts/train_model_a.py --data-dir data/xgboost --output-dir models/xgboost

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

from core.ai_engine.xgboost.model_a import (
    XGBoostModelA,
    ModelAConfig,
    compute_ic,
    compute_ic_per_class,
    compute_brier_score,
    compute_ece,
    fit_platt_calibration,
    apply_platt_calibration,
    DEFAULT_MODEL_A_PARAMS,
)


# =============================================================================
# Constants
# =============================================================================

X_A_FILENAME: str = "X_A.parquet"
Y_A_FILENAME: str = "y_A.parquet"

MODEL_JSON: str = "model_a.json"
CALIBRATED_PKL: str = "model_a_calibrated.pkl"
METRICS_JSON: str = "model_a_metrics.json"

# Default training params (can be overridden via CLI)
DEFAULT_PARAMS: dict[str, Any] = dict(DEFAULT_MODEL_A_PARAMS)


# =============================================================================
# Data Loading
# =============================================================================

def load_data(
    data_dir: Path,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """Load X và y từ parquet files.

    Args:
        data_dir: Thư mục chứa X_A.parquet và y_A.parquet
        split: Chỉ định split (train, val, test).
               Mặc định tìm file {split}/X_A.parquet, fallback về X_A.parquet

    Returns:
        (X, y) tuple: features và labels

    Raises:
        FileNotFoundError: Nếu không tìm thấy file
    """
    # Try split-specific directory first
    x_path = data_dir / split / X_A_FILENAME
    y_path = data_dir / split / Y_A_FILENAME

    if not x_path.exists() or not y_path.exists():
        # Fallback to direct files
        x_path = data_dir / X_A_FILENAME
        y_path = data_dir / Y_A_FILENAME

    if not x_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy X_A: {x_path}. "
            f"Các files trong {data_dir.resolve()}: {list(data_dir.iterdir())}"
        )
    if not y_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy y_A: {y_path}"
        )

    logger.info(f"Đang load X_A: {x_path}")
    X_df = pd.read_parquet(x_path)
    logger.info(f"Đang load y_A: {y_path}")
    y_df = pd.read_parquet(y_path)

    X = X_df.values.astype(np.float64)
    y = y_df.values.ravel().astype(np.int32)

    logger.info(f"  X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"  y distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_all_metrics(
    model: XGBoostModelA,
    X: np.ndarray,
    y: np.ndarray,
    label: str = "train",
) -> dict[str, Any]:
    """Tính toán tất cả metrics cho model evaluation.

    Args:
        model: Trained XGBoostModelA instance
        X: Feature matrix
        y: True labels
        label: Label cho logging

    Returns:
        Dict chứa các metrics
    """
    y_pred_proba = model.predict_proba(X)
    y_pred = model.predict(X)

    accuracy = float(np.mean(y_pred == y))
    ic = compute_ic(y, y_pred_proba)
    ic_per_class = compute_ic_per_class(y, y_pred_proba)
    brier = compute_brier_score(y, y_pred_proba)
    ece = compute_ece(y, y_pred_proba)

    metrics: dict[str, Any] = {
        f"accuracy_{label}": accuracy,
        f"ic_{label}": ic,
        f"ic_per_class_{label}": ic_per_class,
        f"brier_score_{label}": brier,
        f"ece_{label}": ece,
        f"n_samples_{label}": len(y),
    }

    logger.info(f"  [{label}] accuracy={accuracy:.4f}, IC={ic:.4f}, "
                f"Brier={brier:.4f}, ECE={ece:.4f}")
    logger.info(f"  [{label}] IC per class: {ic_per_class}")

    return metrics


# =============================================================================
# Main Training Pipeline
# =============================================================================

def run_training(
    data_dir: Path,
    output_dir: Path,
    params_override: dict[str, Any] | None = None,
    early_stopping_rounds: int = 50,
    fine_tune_rounds: int = 50,
    calibration_cv: int = 5,
    ic_target_min: float = 0.05,
    penalty_lambda: float = 0.5,
    prob_threshold: float = 0.85,
) -> None:
    """Run the full Model A training pipeline.

    Args:
        data_dir: Thư mục chứa dataset parquet files
        output_dir: Thư mục lưu model outputs
        params_override: Override các XGBoost params
        early_stopping_rounds: Early stopping rounds
        fine_tune_rounds: Fine-tuning rounds với custom objective
        calibration_cv: Số fold cho Platt scaling CV
        ic_target_min: IC target minimum
        penalty_lambda: Overconfidence penalty lambda
        prob_threshold: Overconfidence probability threshold
    """
    logger.info("=" * 60)
    logger.info("MODEL A TRAINING PIPELINE")
    logger.info("=" * 60)

    # --- Load data ---
    logger.info("[1/6] Đang load dataset...")
    X_train, y_train = load_data(data_dir, split="train")
    X_val, y_val = load_data(data_dir, split="val")
    X_test, y_test = load_data(data_dir, split="test")

    # --- Build config ---
    logger.info("[2/6] Đang cấu hình model...")
    params = dict(DEFAULT_PARAMS)
    if params_override:
        params.update(params_override)

    config = ModelAConfig(
        params=params,
        early_stopping_rounds=early_stopping_rounds,
        model_path=str(output_dir / MODEL_JSON),
        calibration_method="sigmoid",
        calibration_cv=calibration_cv,
        ic_target_min=ic_target_min,
        penalty_lambda=penalty_lambda,
        prob_threshold=prob_threshold,
    )

    model = XGBoostModelA(config=config)
    logger.info(f"  Params: {params}")
    logger.info(f"  Penalty lambda={penalty_lambda}, prob_threshold={prob_threshold}")

    # --- Train model ---
    logger.info("[3/6] Đang train XGBoost Model A...")
    model.fit(X_train, y_train, X_val, y_val)
    logger.info(f"  Train IC: {model.last_ic:.4f}")
    logger.info(f"  Train IC per class: {model.last_ic_per_class}")

    # --- Fine-tune with overconfidence penalty ---
    logger.info("[4/6] Đang fine-tune với overconfidence penalty...")
    try:
        model.fine_tune_with_overconfidence_penalty(
            X_train, y_train, X_val, y_val,
            n_rounds=fine_tune_rounds,
        )
        logger.info(f"  Fine-tuned IC: {model.last_ic:.4f}")
    except Exception as e:
        logger.warning(f"Fine-tuning thất bại (sẽ dùng model gốc): {e}")

    # --- Evaluate ---
    logger.info("[5/6] Đang evaluate model...")
    metrics: dict[str, Any] = {}

    train_metrics = compute_all_metrics(model, X_train, y_train, label="train")
    val_metrics = compute_all_metrics(model, X_val, y_val, label="val")
    test_metrics = compute_all_metrics(model, X_test, y_test, label="test")

    metrics.update(train_metrics)
    metrics.update(val_metrics)
    metrics.update(test_metrics)
    metrics["penalty_lambda"] = penalty_lambda
    metrics["prob_threshold"] = prob_threshold
    metrics["params"] = params

    # --- Calibration: Platt Scaling ---
    logger.info("[6/6] Đang fit Platt Scaling calibration...")
    calibrator = model.fit_calibration(X_val, y_val)
    calibrator_path: Path | None = None
    if calibrator is not None:
        calibrator_path = output_dir / CALIBRATED_PKL
        calibrator_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(calibrator, str(calibrator_path))
        logger.info(f"  Calibrator saved: {calibrator_path}")

        # Evaluate calibrated probabilities on test set
        calib_probas = apply_platt_calibration(model.predict_proba(X_test), calibrator)
        calib_brier = compute_brier_score(y_test, calib_probas)
        calib_ece = compute_ece(y_test, calib_probas)
        metrics["calibrated_brier_score_test"] = calib_brier
        metrics["calibrated_ece_test"] = calib_ece
        logger.info(f"  Calibrated Test Brier={calib_brier:.4f}, ECE={calib_ece:.4f}")

    # --- Save model ---
    logger.info("Đang lưu model...")
    model.save(output_dir / MODEL_JSON)
    metrics["model_path"] = str(output_dir / MODEL_JSON)
    if calibrator_path:
        metrics["calibrator_path"] = str(calibrator_path)

    # --- Save metrics ---
    metrics_path = output_dir / METRICS_JSON
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved: {metrics_path}")

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("HOÀN THÀNH MODEL A TRAINING")
    logger.info("=" * 60)
    logger.info(f"  Model: {output_dir / MODEL_JSON}")
    logger.info(f"  Calibrator: {calibrator_path or 'N/A'}")
    logger.info(f"  Metrics: {metrics_path}")
    logger.info(f"  Test IC: {test_metrics.get('ic_test', 'N/A'):.4f}")
    logger.info(f"  Test Brier: {test_metrics.get('brier_score_test', 'N/A'):.4f}")
    logger.info(f"  Test ECE: {test_metrics.get('ece_test', 'N/A'):.4f}")

    # IC target check
    ic_test = test_metrics.get("ic_test", 0.0)
    if ic_test < ic_target_min:
        logger.warning(
            f"Test IC ({ic_test:.4f}) < target ({ic_target_min}). "
            "Cân nhắc retrain với params khác."
        )


# =============================================================================
# CLI
# =============================================================================

def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Train XGBoost Model A (Phase 6.1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/xgboost",
        help="Thư mục chứa X_A.parquet và y_A.parquet (default: data/xgboost)",
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
        default=6,
        help="Max tree depth (default: 6)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate (default: 0.05)",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=50,
        help="Early stopping rounds (default: 50)",
    )
    parser.add_argument(
        "--fine-tune-rounds",
        type=int,
        default=50,
        help="Fine-tuning rounds với custom objective (default: 50)",
    )
    parser.add_argument(
        "--penalty-lambda",
        type=float,
        default=0.5,
        help="Overconfidence penalty lambda (default: 0.5)",
    )
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.85,
        help="Overconfidence probability threshold (default: 0.85)",
    )
    parser.add_argument(
        "--calibration-cv",
        type=int,
        default=5,
        help="Số fold cho Platt scaling CV (default: 5)",
    )
    parser.add_argument(
        "--ic-target",
        type=float,
        default=0.05,
        help="IC target minimum (default: 0.05)",
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
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    return parser.parse_args(args)


def main() -> None:
    """Entry point cho train_model_a script."""
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

    try:
        run_training(
            data_dir=data_dir,
            output_dir=output_dir,
            params_override=params_override,
            early_stopping_rounds=args.early_stopping,
            fine_tune_rounds=args.fine_tune_rounds,
            calibration_cv=args.calibration_cv,
            ic_target_min=args.ic_target,
            penalty_lambda=args.penalty_lambda,
            prob_threshold=args.prob_threshold,
        )
    except Exception as e:
        logger.error(f"Model A training thất bại: {e}")
        raise


if __name__ == "__main__":
    main()
