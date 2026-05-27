# =============================================================================
# AGENTIC-QUANT — Evaluate Models Script (Phase 6.7)
# Evaluation pipeline: IC, Brier, ECE, SHAP, overfitting check, deploy-if-pass
# =============================================================================
"""
Evaluate trained XGBoost models and decide whether to deploy.

Usage:
    python scripts/evaluate_models.py \\
        --model-a-path models/xgboost/model_a.json \\
        --model-b-path models/xgboost/model_b.json \\
        --test-data data/processed/test_features.parquet \\
        --output-dir models/evaluation/ \\
        --deploy-if-pass

Output:
    - evaluation_report.json: day du metrics
    - deploy/ directory: copy model files neu pass threshold
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

# ── Optional imports ──────────────────────────────────────────────────────
try:
    import xgboost as xgb
except ImportError:
    xgb = None  # type: ignore[assignment]

try:
    import shap
except ImportError:
    shap = None  # type: ignore[assignment]

try:
    from sklearn.metrics import brier_score_loss
except ImportError:
    brier_score_loss = None  # type: ignore[assignment]

# ── Core module imports ───────────────────────────────────────────────────
try:
    from core.backtesting.ic_calculator import ICCalculator
except ImportError:
    ICCalculator = None  # type: ignore[assignment]


# =============================================================================
# Constants
# =============================================================================
IC_TARGET_MIN: float = 0.05
BRIER_TARGET_MAX: float = 0.25
ECE_TARGET_MAX: float = 0.05
OVERFITTING_GAP_WARN: float = 0.15
ECE_N_BINS: int = 10

CLASS_LABELS = ["BSL", "SSL", "LATERAL"]


# =============================================================================
# Evaluation Functions
# =============================================================================

def compute_ic(
    y_pred_proba: np.ndarray,
    y_true: np.ndarray,
    ic_calculator: ICCalculator | None = None,
) -> dict[str, float]:
    """Tinh IC (Spearman) cho Model A predictions.

    Args:
        y_pred_proba: [n_samples, 3] probability matrix (BSL, SSL, LATERAL)
        y_true: [n_samples] integer labels (0=BSL, 1=SSL, 2=LATERAL)
        ic_calculator: ICCalculator instance (optional)

    Returns:
        dict: {'overall_ic', 'ic_bsl', 'ic_ssl', 'ic_lateral'}
    """
    if ic_calculator is None:
        ic_calc = ICCalculator() if ICCalculator is not None else None
    else:
        ic_calc = ic_calculator

    # Signed prediction: P_BSL - P_SSL (positive = bullish bias)
    y_hat_signed = y_pred_proba[:, 0] - y_pred_proba[:, 1]

    # Encode actual: BSL_HIT(0) -> +1, SSL_HIT(1) -> -1, LATERAL(2) -> 0
    y_actual_signed = np.where(y_true == 0, 1.0, np.where(y_true == 1, -1.0, 0.0))

    results: dict[str, float] = {}

    if ic_calc is not None:
        try:
            overall_ic = ic_calc.compute_ic(y_hat_signed, y_actual_signed)
            results["overall_ic"] = float(overall_ic)
        except Exception as e:
            logger.warning(f"IC computation failed: {e}")
            results["overall_ic"] = 0.0
    else:
        # Fallback: manual Spearman
        from scipy.stats import spearmanr
        corr, _ = spearmanr(y_hat_signed, y_actual_signed)
        results["overall_ic"] = float(corr) if not np.isnan(corr) else 0.0

    # Per-class IC
    for i, label in enumerate(CLASS_LABELS):
        mask = y_true == i
        if mask.sum() < 3:
            results[f"ic_{label.lower()}"] = 0.0
            continue
        if ic_calc is not None:
            try:
                ic_val = ic_calc.compute_ic(y_pred_proba[mask, i], y_true[mask].astype(float))
                results[f"ic_{label.lower()}"] = float(ic_val)
            except Exception:
                results[f"ic_{label.lower()}"] = 0.0
        else:
            from scipy.stats import spearmanr
            corr, _ = spearmanr(y_pred_proba[mask, i], y_true[mask].astype(float))
            results[f"ic_{label.lower()}"] = float(corr) if not np.isnan(corr) else 0.0

    return results


def compute_brier(y_pred_proba: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    """Tinh Brier Score cho multiclass.

    Brier = (1/N) * sum_c sum_i (p_ic - y_ic)^2
    Target: < 0.25

    Args:
        y_pred_proba: [n_samples, 3]
        y_true: [n_samples] integer labels

    Returns:
        dict: {'brier_score', 'brier_bsl', 'brier_ssl', 'brier_lateral'}
    """
    n = len(y_true)
    if n == 0:
        return {"brier_score": 1.0, "brier_bsl": 1.0, "brier_ssl": 1.0, "brier_lateral": 1.0}

    # One-hot encode true labels
    y_true_onehot = np.zeros((n, 3), dtype=float)
    y_true_onehot[np.arange(n), y_true] = 1.0

    # Overall Brier
    brier = float(np.mean(np.sum((y_pred_proba - y_true_onehot) ** 2, axis=1)))
    results: dict[str, float] = {"brier_score": brier}

    # Per-class Brier
    for i, label in enumerate(CLASS_LABELS):
        brier_c = float(np.mean((y_pred_proba[:, i] - y_true_onehot[:, i]) ** 2))
        results[f"brier_{label.lower()}"] = brier_c

    return results


def compute_ece(
    y_pred_proba: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = ECE_N_BINS,
) -> dict[str, float]:
    """Tinh Expected Calibration Error (ECE).

    ECE = sum_k (w_k * |acc_k - conf_k|)
    Target: < 0.05

    Args:
        y_pred_proba: [n_samples, 3]
        y_true: [n_samples] integer labels
        n_bins: So bin cho calibration histogram

    Returns:
        dict: {'ece', 'ece_bsl', 'ece_ssl', 'ece_lateral', 'ece_max', 'ece_mean'}
    """
    n = len(y_true)
    if n == 0:
        return {"ece": 1.0, "ece_bsl": 1.0, "ece_ssl": 1.0, "ece_lateral": 1.0,
                "ece_max": 1.0, "ece_mean": 1.0}

    y_true_onehot = np.zeros((n, 3), dtype=float)
    y_true_onehot[np.arange(n), y_true] = 1.0

    ece_values: list[float] = []

    for i in range(3):
        conf = y_pred_proba[:, i]
        acc = y_true_onehot[:, i]

        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(conf, bin_boundaries, right=False) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        ece = 0.0
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if mask.sum() == 0:
                continue
            bin_conf = conf[mask].mean()
            bin_acc = acc[mask].mean()
            w_k = mask.sum() / n
            ece += w_k * abs(bin_acc - bin_conf)

        ece_values.append(ece)

    results = {
        "ece": float(np.mean(ece_values)),
        "ece_bsl": float(ece_values[0]),
        "ece_ssl": float(ece_values[1]),
        "ece_lateral": float(ece_values[2]),
        "ece_max": float(max(ece_values)),
        "ece_mean": float(np.mean(ece_values)),
    }
    return results


def compute_shap_importance(
    model: Any,
    X_test: np.ndarray,
    feature_names: list[str] | None = None,
    max_samples: int = 500,
) -> dict[str, list[dict[str, float]]]:
    """Tinh SHAP feature importance.

    Args:
        model: XGBoost Booster or sklearn model
        X_test: [n_samples, n_features]
        feature_names: Optional feature names
        max_samples: Max samples for SHAP (performance)

    Returns:
        dict: {
            'mean_abs_shap': [{feature: importance}],
            'top_20_features': [feature_name]
        }
    """
    if shap is None:
        return {"mean_abs_shap": [], "top_20_features": [], "warning": "shap not installed"}

    if len(X_test) > max_samples:
        idx = np.random.choice(len(X_test), max_samples, replace=False)
        X_sample = X_test[idx]
    else:
        X_sample = X_test

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # shap_values shape: (n_samples, n_features) for binary
        # or list of arrays for multiclass
        if isinstance(shap_values, list):
            # Multiclass: average absolute SHAP across classes
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)

        n_features = len(mean_shap)
        if feature_names is None:
            feature_names = [f"f_{i}" for i in range(n_features)]

        # Sort by importance
        importance_list = [
            {"name": feature_names[i], "importance": float(mean_shap[i])}
            for i in range(n_features)
        ]
        importance_list.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "mean_abs_shap": importance_list,
            "top_20_features": [imp["name"] for imp in importance_list[:20]],
        }

    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return {"mean_abs_shap": [], "top_20_features": [], "error": str(e)}


def check_overfitting(
    ic_backtest: float,
    ic_forward: float,
    threshold: float = OVERFITTING_GAP_WARN,
) -> dict[str, Any]:
    """Kiem tra overfitting: gap giua IC backtest vs IC forward.

    Args:
        ic_backtest: IC tren backtest period
        ic_forward: IC tren forward/out-of-sample period
        threshold: Canh bao neu gap > threshold

    Returns:
        dict: {'gap', 'is_overfitting', 'warning'}
    """
    gap = abs(ic_backtest - ic_forward)
    return {
        "ic_backtest": ic_backtest,
        "ic_forward": ic_forward,
        "gap": gap,
        "is_overfitting": gap > threshold,
        "warning": f"Overfitting risk: IC gap={gap:.4f} > {threshold}"
        if gap > threshold
        else "No overfitting detected",
    }


# =============================================================================
# Deploy Decision
# =============================================================================

def should_deploy(
    ic: float,
    brier: float,
    ece: float,
    ic_target: float = IC_TARGET_MIN,
    brier_target: float = BRIER_TARGET_MAX,
    ece_target: float = ECE_TARGET_MAX,
) -> dict[str, Any]:
    """Quyet dinh co deploy model hay khong.

    Args:
        ic: Overall IC
        brier: Brier Score
        ece: Expected Calibration Error
        ic_target: IC minimum threshold
        brier_target: Brier maximum threshold
        ece_target: ECE maximum threshold

    Returns:
        dict: {'deploy': bool, 'reasons': list[str]}
    """
    reasons: list[str] = []
    deploy = True

    if ic < ic_target:
        reasons.append(f"IC={ic:.4f} < target={ic_target}")
        deploy = False
    if brier > brier_target:
        reasons.append(f"Brier={brier:.4f} > target={brier_target}")
        deploy = False
    if ece > ece_target:
        reasons.append(f"ECE={ece:.4f} > target={ece_target}")
        deploy = False

    return {
        "deploy": deploy and len(reasons) == 0,
        "reasons": reasons if not deploy else ["All metrics pass"],
        "ic": ic,
        "brier": brier,
        "ece": ece,
        "ic_target": ic_target,
        "brier_target": brier_target,
        "ece_target": ece_target,
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def evaluate_model_a(
    model_path: Path,
    test_data_path: Path,
    ic_calculator: ICCalculator | None = None,
) -> dict[str, Any]:
    """Evaluate Model A (multiclass: BSL/SSL/LATERAL).

    Args:
        model_path: Path to model_a.json
        test_data_path: Path to test Parquet with X_A and y_A
        ic_calculator: Optional ICCalculator instance

    Returns:
        dict: All evaluation metrics
    """
    logger.info(f"Evaluating Model A: {model_path}")
    logger.info(f"Test data: {test_data_path}")

    if xgb is None:
        return {"error": "xgboost not installed", "model": "A"}

    # Load model
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    logger.info(f"Model loaded: {model_path}")

    # Load test data
    df = pd.read_parquet(test_data_path)
    if "X_A" not in df.columns or "y_A" not in df.columns:
        # Try column-based format
        feature_cols = [c for c in df.columns if c.startswith("f_")]
        if feature_cols:
            X_test = df[feature_cols].values
        else:
            X_test = df.drop(columns=["y_A", "y_B"]).values if "y_A" in df.columns else df.values
        y_test = df["y_A"].values if "y_A" in df.columns else df.iloc[:, -1].values
    else:
        X_test = np.stack(df["X_A"].values)
        y_test = df["y_A"].values

    logger.info(f"Test samples: {len(X_test)}")

    # Predict
    y_pred_proba = model.predict_proba(X_test)  # [n, 3]

    # Metrics
    ic_results = compute_ic(y_pred_proba, y_test, ic_calculator)
    brier_results = compute_brier(y_pred_proba, y_test)
    ece_results = compute_ece(y_pred_proba, y_test)

    # SHAP
    feature_names = [f"f_{i}" for i in range(X_test.shape[1])]
    shap_results = compute_shap_importance(model.get_booster(), X_test, feature_names)

    report = {
        "model": "A",
        "model_path": str(model_path),
        "test_samples": int(len(X_test)),
        "metrics": {
            **ic_results,
            **brier_results,
            **ece_results,
        },
        "shap": shap_results,
    }

    logger.info(f"Model A: IC={ic_results.get('overall_ic', 0):.4f}, "
                f"Brier={brier_results.get('brier_score', 1):.4f}, "
                f"ECE={ece_results.get('ece', 1):.4f}")
    return report


def evaluate_model_b(
    model_path: Path,
    test_data_path: Path,
) -> dict[str, Any]:
    """Evaluate Model B (binary: P_hold).

    Args:
        model_path: Path to model_b.json
        test_data_path: Path to test Parquet with X_B and y_B

    Returns:
        dict: Evaluation metrics
    """
    logger.info(f"Evaluating Model B: {model_path}")

    if xgb is None:
        return {"error": "xgboost not installed", "model": "B"}

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    df = pd.read_parquet(test_data_path)
    feature_cols = [c for c in df.columns if c.startswith("f_")]
    if feature_cols:
        X_test = df[feature_cols].values
    else:
        X_test = df.drop(columns=["y_A", "y_B"]).values if "y_B" in df.columns else df.values
    y_test = df["y_B"].values if "y_B" in df.columns else df.iloc[:, -1].values

    y_pred_proba = model.predict_proba(X_test)[:, 1]  # P_hold probability

    # Binary Brier
    brier = float(brier_score_loss(y_test, y_pred_proba)) if brier_score_loss is not None else 0.0

    # Binary ECE
    bin_boundaries = np.linspace(0.0, 1.0, ECE_N_BINS + 1)
    bin_indices = np.digitize(y_pred_proba, bin_boundaries, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, ECE_N_BINS - 1)
    ece = 0.0
    n = len(y_test)
    for bin_idx in range(ECE_N_BINS):
        mask = bin_indices == bin_idx
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y_test[mask].mean() - y_pred_proba[mask].mean())

    report = {
        "model": "B",
        "model_path": str(model_path),
        "test_samples": int(len(X_test)),
        "metrics": {
            "brier_score": brier,
            "ece": ece,
        },
    }

    logger.info(f"Model B: Brier={brier:.4f}, ECE={ece:.4f}")
    return report


# =============================================================================
# Deploy
# =============================================================================

def deploy_models(
    report_a: dict[str, Any],
    report_b: dict[str, Any],
    output_dir: Path,
) -> None:
    """Copy model + scaler files to production/ if evaluation passes.

    Args:
        report_a: Model A evaluation report
        report_b: Model B evaluation report
        output_dir: Output directory
    """
    prod_dir = output_dir / "production"
    prod_dir.mkdir(parents=True, exist_ok=True)

    # Determine deploy decision
    metrics_a = report_a.get("metrics", {})
    metrics_b = report_b.get("metrics", {})

    decision_a = should_deploy(
        ic=metrics_a.get("overall_ic", 0.0),
        brier=metrics_a.get("brier_score", 1.0),
        ece=metrics_a.get("ece", 1.0),
    )
    decision_b = should_deploy(
        ic=0.05,  # No IC for binary, use Brier
        brier=metrics_b.get("brier_score", 1.0),
        ece=metrics_b.get("ece", 1.0),
    )

    deploy_actions = []

    if decision_a["deploy"]:
        src = Path(report_a["model_path"])
        if src.exists():
            dst = prod_dir / src.name
            import shutil
            shutil.copy2(src, dst)
            deploy_actions.append(f"Model A deployed: {dst}")
            logger.info(f"Model A deployed to {dst}")

    if decision_b["deploy"]:
        src = Path(report_b["model_path"])
        if src.exists():
            dst = prod_dir / src.name
            import shutil
            shutil.copy2(src, dst)
            deploy_actions.append(f"Model B deployed: {dst}")
            logger.info(f"Model B deployed to {dst}")

    # Write deploy report
    deploy_report = {
        "model_a": decision_a,
        "model_b": decision_b,
        "actions": deploy_actions,
        "timestamp": str(pd.Timestamp.now()),
    }

    deploy_path = prod_dir / "deploy_report.json"
    with open(deploy_path, "w") as f:
        json.dump(deploy_report, f, indent=2)
    logger.info(f"Deploy report saved: {deploy_path}")


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="AGENTIC-QUANT Model Evaluation Pipeline"
    )
    parser.add_argument(
        "--model-a-path",
        type=str,
        default="models/xgboost/model_a.json",
        help="Path to Model A (XGBoost multiclass)",
    )
    parser.add_argument(
        "--model-b-path",
        type=str,
        default="models/xgboost/model_b.json",
        help="Path to Model B (XGBoost binary)",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test Parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/evaluation",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--deploy-if-pass",
        action="store_true",
        help="Copy models to production/ if all metrics pass",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP computation (faster)",
    )
    return parser


def main() -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_data_path = Path(args.test_data)
    if not test_data_path.exists():
        logger.error(f"Test data not found: {test_data_path}")
        sys.exit(1)

    # Evaluate
    report_a = evaluate_model_a(
        model_path=Path(args.model_a_path),
        test_data_path=test_data_path,
    )
    report_b = evaluate_model_b(
        model_path=Path(args.model_b_path),
        test_data_path=test_data_path,
    )

    # Combine report
    full_report = {
        "model_a": report_a,
        "model_b": report_b,
        "timestamp": str(pd.Timestamp.now()),
        "deploy_if_pass": args.deploy_if_pass,
    }

    # Save report
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    logger.info(f"Evaluation report saved: {report_path}")

    # Deploy
    if args.deploy_if_pass:
        deploy_models(report_a, report_b, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    metrics_a = report_a.get("metrics", {})
    print(f"\nModel A (BSL/SSL/LATERAL):")
    print(f"  IC:      {metrics_a.get('overall_ic', 0):.4f}  (target > {IC_TARGET_MIN})")
    print(f"  Brier:   {metrics_a.get('brier_score', 1):.4f}  (target < {BRIER_TARGET_MAX})")
    print(f"  ECE:     {metrics_a.get('ece', 1):.4f}  (target < {ECE_TARGET_MAX})")

    metrics_b = report_b.get("metrics", {})
    print(f"\nModel B (P_hold):")
    print(f"  Brier:   {metrics_b.get('brier_score', 1):.4f}  (target < {BRIER_TARGET_MAX})")
    print(f"  ECE:     {metrics_b.get('ece', 1):.4f}  (target < {ECE_TARGET_MAX})")

    if args.deploy_if_pass:
        print(f"\nDeploy: {output_dir / 'production/'}")

    print("=" * 60)


if __name__ == "__main__":
    main()
