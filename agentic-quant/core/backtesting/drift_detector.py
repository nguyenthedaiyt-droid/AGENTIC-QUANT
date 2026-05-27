# =============================================================================
# AGENTIC-QUANT — Drift Detector (Phase 8)
# Phat hien feature drift bang PSI (Population Stability Index) va FDS
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    pass


# =============================================================================
# Constants
# =============================================================================
_FDS_THRESHOLD_DEFAULT = 0.2
_MODEL_DEGRADED_THRESHOLD_DEFAULT = 0.4
_PSI_BUCKETS_DEFAULT = 10

# PSI interpretation
PSI_INTERPRETATION: dict[str, tuple[float, float, str]] = {
    "LOW": (0.0, 0.1, "Khong co drift hoac rat nho"),
    "MODERATE": (0.1, 0.25, "Drift nhe, can theo doi"),
    "HIGH": (0.25, float("inf"), "Drift manh, can retrain"),
}


# =============================================================================
# DriftDetector
# =============================================================================
class DriftDetector:
    """Phat hien feature drift bang PSI (Population Stability Index).

    PSI do luong su khac biet giua phan phoi reference (training) va current.
    - PSI < 0.1: Khong co drift
    - 0.1 <= PSI < 0.25: Drill nhe
    - PSI >= 0.25: Drift manh

    FDS (Feature Drift Score) la trung binh co trong so cua PSI cac features.

    Vi du:
        detector = DriftDetector()
        psi = detector.compute_psi(ref_data, cur_data)
        fds = detector.compute_fds(psi)
        degraded = detector.check_model_degraded(fds)
    """

    def __init__(self) -> None:
        self._drift_history: list[dict[str, Any]] = []
        self._last_fds: float = 0.0
        self._last_psi: dict[str, float] = {}
        self._last_regime: str = ""
        self._n_checks: int = 0
        self._n_degraded: int = 0

    # -------------------------------------------------------------------------
    # PSI Computation
    # -------------------------------------------------------------------------
    def compute_psi(
        self,
        reference: np.ndarray | list[float],
        current: np.ndarray | list[float],
        buckets: int = _PSI_BUCKETS_DEFAULT,
    ) -> float:
        """Tinh Population Stability Index cho mot feature.

        PSI = sum((p_i - q_i) * ln(p_i / q_i)) voi:
        - p_i: Ty le samples trong bucket i cua current distribution
        - q_i: Ty le samples trong bucket i cua reference distribution

        Args:
            reference: Phan phoi reference (training data)
            current: Phan phoi current (production data)
            buckets: So luong buckets de phan chia (default: 10)

        Returns:
            float: PSI value

        Raises:
            ValueError: Neu reference hoac current trong hoac khong hop le
        """
        reference = np.asarray(reference, dtype=np.float64).flatten()
        current = np.asarray(current, dtype=np.float64).flatten()

        if len(reference) < buckets:
            logger.warning(
                "Reference co {n} samples, it hon buckets ({b})",
                n=len(reference),
                b=buckets,
            )
            return 0.0

        if len(current) == 0:
            logger.warning("Current distribution rong -> PSI = 0.0")
            return 0.0

        # Tao bins tu reference distribution
        # Su dung percentiles de dam bao buckets dong deu
        try:
            bin_edges = np.percentile(
                reference,
                np.linspace(0, 100, buckets + 1),
            )
        except IndexError:
            logger.warning("Khong the tao percentiles -> PSI = 0.0")
            return 0.0

        # Dam bao bin_edges[0] <= min(reference) va bin_edges[-1] >= max(reference)
        bin_edges[0] = min(bin_edges[0], reference.min())
        bin_edges[-1] = max(bin_edges[-1], reference.max())

        # Tranh duplicate edges (truong hop reference co nhieu gia tri giong nhau)
        unique_edges = np.unique(bin_edges)
        if len(unique_edges) < 2:
            logger.warning("Bin edges khong du unique -> PSI = 0.0")
            return 0.0

        # Dem so luong samples trong moi bucket
        ref_counts, _ = np.histogram(reference, bins=unique_edges)
        cur_counts, _ = np.histogram(current, bins=unique_edges)

        # Chuyen thanh ty le
        ref_pct = ref_counts / len(reference)
        cur_pct = cur_counts / len(current)

        # Small sample adjustment: replace 0 voi 0.0001 de tranh log(0)
        ref_pct = np.clip(ref_pct, 1e-10, 1.0)
        cur_pct = np.clip(cur_pct, 1e-10, 1.0)

        # Tinh PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

        return float(psi)

    # -------------------------------------------------------------------------
    # Feature Drift Score (FDS)
    # -------------------------------------------------------------------------
    def compute_fds(
        self,
        psi_values: dict[str, float],
        threshold: float = _FDS_THRESHOLD_DEFAULT,
    ) -> float:
        """Tinh Feature Drift Score.

        FDS = mean(PSI_values) — trung binh cong cua PSI tat ca features.

        Args:
            psi_values: Dict {feature_name: PSI_value}
            threshold: Nguong PSI de coi la feature bi drift (default: 0.2)

        Returns:
            float: FDS value (0.0 -> 1.0, nhung thuong < 1.0)
        """
        if not psi_values:
            logger.warning("PSI values rong -> FDS = 0.0")
            return 0.0

        values = np.array(list(psi_values.values()), dtype=np.float64)
        fds = float(np.mean(values))

        # Dem so features bi drift
        drifted = [name for name, psi in psi_values.items() if psi > threshold]
        self._last_psi = psi_values
        self._last_fds = fds

        logger.debug(
            "FDS = {fds:.4f} | {n_drifted}/{n_total} features drifted ({threshold})",
            fds=fds,
            n_drifted=len(drifted),
            n_total=len(psi_values),
            threshold=threshold,
        )

        return fds

    # -------------------------------------------------------------------------
    # Model Degraded Check
    # -------------------------------------------------------------------------
    def check_model_degraded(
        self,
        fds: float,
        threshold: float = _MODEL_DEGRADED_THRESHOLD_DEFAULT,
    ) -> bool:
        """Kiem tra xem model co bi degraded hay khong.

        Degraded = FDS > threshold (default 0.4).

        Args:
            fds: Feature Drift Score
            threshold: Nguong FDS de coi la degraded (default: 0.4)

        Returns:
            bool: True neu model degraded, False otherwise
        """
        degraded = fds > threshold

        self._n_checks += 1
        if degraded:
            self._n_degraded += 1

        if degraded:
            logger.warning(
                "[DriftDetector] Model DEGRADED: FDS={fds:.4f} > threshold={thresh}",
                fds=fds,
                thresh=threshold,
            )
        else:
            logger.info(
                "[DriftDetector] Model OK: FDS={fds:.4f} <= threshold={thresh}",
                fds=fds,
                thresh=threshold,
            )

        return degraded

    # -------------------------------------------------------------------------
    # Drift Report
    # -------------------------------------------------------------------------
    def log_drift_report(
        self,
        psi_values: dict[str, float],
        fds: float,
        current_regime: str = "",
    ) -> None:
        """Ghi lai drift report.

        Args:
            psi_values: Dict {feature_name: PSI_value}
            fds: Feature Drift Score
            current_regime: Regime hien tai (neu co)
        """
        self._last_regime = current_regime

        # Sap xep PSI values giam dan
        sorted_psi = sorted(psi_values.items(), key=lambda x: x[1], reverse=True)

        # Phan loai PSI
        low_drift = [name for name, psi in sorted_psi if psi < 0.1]
        moderate_drift = [name for name, psi in sorted_psi if 0.1 <= psi < 0.25]
        high_drift = [name for name, psi in sorted_psi if psi >= 0.25]

        report = {
            "fds": round(fds, 4),
            "regime": current_regime,
            "n_features": len(psi_values),
            "low_drift": len(low_drift),
            "moderate_drift": len(moderate_drift),
            "high_drift": len(high_drift),
            "top_drifted": sorted_psi[:5],
            "drifted_features": [name for name, _ in high_drift],
        }

        self._drift_history.append(report)

        # Log summary
        logger.info(
            "[DriftReport] FDS={fds} | Regime={regime} | "
            "Low={low} Moderate={mod} High={high}",
            fds=round(fds, 4),
            regime=current_regime or "N/A",
            low=len(low_drift),
            mod=len(moderate_drift),
            high=len(high_drift),
        )

        if high_drift:
            logger.warning(
                "Features bi drift manh (PSI >= 0.25): {features}",
                features=high_drift[:5],
            )

    # -------------------------------------------------------------------------
    # Stats / History
    # -------------------------------------------------------------------------
    @property
    def last_fds(self) -> float:
        """FDS value gan nhat."""
        return self._last_fds

    @property
    def last_psi(self) -> dict[str, float]:
        """PSI values gan nhat."""
        return dict(self._last_psi)

    @property
    def drift_history(self) -> list[dict[str, Any]]:
        """Lich su cac drift report."""
        return list(self._drift_history)

    def get_degradation_rate(self) -> float:
        """Ty le kiem tra bi degraded.

        Returns:
            float: Degradation rate (0.0 -> 1.0)
        """
        if self._n_checks == 0:
            return 0.0
        return self._n_degraded / self._n_checks

    def get_stats(self) -> dict[str, Any]:
        """Tra ve thong ke.

        Returns:
            Dict statistics
        """
        return {
            "n_checks": self._n_checks,
            "n_degraded": self._n_degraded,
            "degradation_rate": self.get_degradation_rate(),
            "last_fds": self._last_fds,
            "last_regime": self._last_regime,
            "n_drift_reports": len(self._drift_history),
        }
