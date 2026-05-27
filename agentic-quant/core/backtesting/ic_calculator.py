# =============================================================================
# AGENTIC-QUANT — Information Coefficient (IC) Calculator (Phase 8)
# Tinh toan IC, rolling IC, IC phan bo theo regime/session/impact
# =============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger
from scipy.stats import spearmanr

if TYPE_CHECKING:
    pass


# =============================================================================
# ICCalculator
# =============================================================================
class ICCalculator:
    """Tinh toan Information Coefficient cho backtest evaluation.

    IC do luong quality cua prediction (y_hat) so voi actual (y_actual).
    Gia tri IC:
    - 1.0: Perfect prediction
    - 0.0: Random
    - -1.0: Nguoc chieu hoan toan

    Vi du:
        ic_calc = ICCalculator()
        ic = ic_calc.compute_ic(y_hat=[0.1, 0.2, 0.3], y_actual=[0.15, 0.18, 0.32])
        rolling = ic_calc.compute_rolling_ic(y_hats, y_actuals, window=20)
    """

    def __init__(self) -> None:
        self._n_calls: int = 0
        self._total_ic: float = 0.0

    # -------------------------------------------------------------------------
    # Core IC computation
    # -------------------------------------------------------------------------
    def compute_ic(
        self,
        y_hat: np.ndarray | list[float],
        y_actual: np.ndarray | list[float],
    ) -> float:
        """Tinh Spearman rank correlation giua predicted va actual.

        Args:
            y_hat: Gia tri du doan (predictions)
            y_actual: Gia tri thuc te (actuals)

        Returns:
            Spearman correlation coefficient (IC)

        Raises:
            ValueError: Neu input khong hop le hoac khong du diem de tinh
        """
        y_hat = np.asarray(y_hat, dtype=np.float64)
        y_actual = np.asarray(y_actual, dtype=np.float64)

        if len(y_hat) != len(y_actual):
            raise ValueError(
                f"Do dai y_hat ({len(y_hat)}) va y_actual ({len(y_actual)}) khong match"
            )

        if len(y_hat) < 3:
            logger.warning(
                "Can it nhat 3 samples de tinh IC (co {n})",
                n=len(y_hat),
            )
            return 0.0

        # Kiem tra constant input
        if np.std(y_hat) < 1e-12 or np.std(y_actual) < 1e-12:
            logger.warning("Input constant -> IC = 0.0")
            return 0.0

        result = spearmanr(y_hat, y_actual)
        # Handle both scipy >= 1.8 (namedtuple) and older versions (tuple)
        if isinstance(result, (list, tuple)):
            ic_val = float(result[0])
        else:
            ic_val = float(result.statistic)  # type: ignore[union-attr]

        # Neu NaN (truong hop edge), fallback ve 0
        if np.isnan(ic_val):
            logger.warning("IC = NaN (edge case), fallback ve 0.0")
            ic_val = 0.0

        self._n_calls += 1
        self._total_ic += ic_val

        return ic_val

    # -------------------------------------------------------------------------
    # Rolling IC
    # -------------------------------------------------------------------------
    def compute_rolling_ic(
        self,
        y_hats: np.ndarray | list[float],
        y_actuals: np.ndarray | list[float],
        window: int = 20,
    ) -> np.ndarray:
        """Tinh rolling IC voi window co dinh.

        Args:
            y_hats: Mang cac predictions
            y_actuals: Mang cac actuals
            window: Kich thuoc rolling window (default: 20)

        Returns:
            np.ndarray rolling IC values (NaN cho cac index < window)
        """
        y_hats = np.asarray(y_hats, dtype=np.float64)
        y_actuals = np.asarray(y_actuals, dtype=np.float64)

        if len(y_hats) != len(y_actuals):
            raise ValueError(
                f"Do dai y_hats ({len(y_hats)}) va y_actuals ({len(y_actuals)}) khong match"
            )

        n = len(y_hats)
        if n < window:
            logger.warning(
                "So luong mau ({n}) < window ({w}), tra ve array rong",
                n=n,
                w=window,
            )
            return np.array([])

        rolling_ic = np.full(n, np.nan)

        for i in range(window - 1, n):
            start = i - window + 1
            ic_val = self.compute_ic(
                y_hats[start : i + 1],
                y_actuals[start : i + 1],
            )
            rolling_ic[i] = ic_val

        return rolling_ic

    # -------------------------------------------------------------------------
    # IC by regime
    # -------------------------------------------------------------------------
    def compute_ic_by_regime(
        self,
        ic_values: np.ndarray | list[float],
        regimes: np.ndarray | list[str],
    ) -> dict[str, float]:
        """Tinh trung binh IC theo tung regime.

        Args:
            ic_values: Mang IC values
            regimes: Mang regime labels tuong ung (VD: TRENDING_LV, CHOPPY_HV, NORMAL)

        Returns:
            Dict[regime, mean_ic] — IC trung binh cho tung regime
        """
        ic_values = np.asarray(ic_values, dtype=np.float64)
        regimes = np.asarray(regimes)

        if len(ic_values) != len(regimes):
            raise ValueError("Do dai ic_values va regimes khong match")

        unique_regimes = np.unique(regimes)
        result: dict[str, float] = {}

        for regime in unique_regimes:
            mask = regimes == regime
            regime_ics = ic_values[mask]
            regime_ics = regime_ics[~np.isnan(regime_ics)]

            if len(regime_ics) > 0:
                result[str(regime)] = float(np.mean(regime_ics))
            else:
                result[str(regime)] = 0.0

        return result

    # -------------------------------------------------------------------------
    # IC by session
    # -------------------------------------------------------------------------
    def compute_ic_by_session(
        self,
        ic_values: np.ndarray | list[float],
        sessions: np.ndarray | list[str],
    ) -> dict[str, float]:
        """Tinh trung binh IC theo tung phien giao dich.

        Args:
            ic_values: Mang IC values
            sessions: Mang session labels (VD: ASIA, LONDON, NEW_YORK, ASIA_LONDON_OVERLAP)

        Returns:
            Dict[session, mean_ic] — IC trung binh cho tung session
        """
        ic_values = np.asarray(ic_values, dtype=np.float64)
        sessions = np.asarray(sessions)

        if len(ic_values) != len(sessions):
            raise ValueError("Do dai ic_values va sessions khong match")

        unique_sessions = np.unique(sessions)
        result: dict[str, float] = {}

        for session in unique_sessions:
            mask = sessions == session
            session_ics = ic_values[mask]
            session_ics = session_ics[~np.isnan(session_ics)]

            if len(session_ics) > 0:
                result[str(session)] = float(np.mean(session_ics))
            else:
                result[str(session)] = 0.0

        return result

    # -------------------------------------------------------------------------
    # IC by impact
    # -------------------------------------------------------------------------
    def compute_ic_by_impact(
        self,
        ic_values: np.ndarray | list[float],
        impact_levels: np.ndarray | list[str],
    ) -> dict[str, float]:
        """Tinh trung binh IC theo tung impact level.

        Args:
            ic_values: Mang IC values
            impact_levels: Mang impact levels (VD: LOW, MEDIUM, HIGH, NONE)

        Returns:
            Dict[impact, mean_ic] — IC trung binh cho tung impact level
        """
        ic_values = np.asarray(ic_values, dtype=np.float64)
        impact_levels = np.asarray(impact_levels)

        if len(ic_values) != len(impact_levels):
            raise ValueError("Do dai ic_values va impact_levels khong match")

        unique_levels = np.unique(impact_levels)
        result: dict[str, float] = {}

        for level in unique_levels:
            mask = impact_levels == level
            level_ics = ic_values[mask]
            level_ics = level_ics[~np.isnan(level_ics)]

            if len(level_ics) > 0:
                result[str(level)] = float(np.mean(level_ics))
            else:
                result[str(level)] = 0.0

        return result

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------
    @property
    def n_calls(self) -> int:
        """So lan compute_ic da duoc goi."""
        return self._n_calls

    @property
    def average_ic(self) -> float:
        """IC trung binh cua tat ca cac lan compute_ic."""
        if self._n_calls == 0:
            return 0.0
        return self._total_ic / self._n_calls

    def get_stats(self) -> dict[str, Any]:
        """Tra ve thong tin thong ke.

        Returns:
            Dict statistics
        """
        return {
            "n_calls": self._n_calls,
            "average_ic": self.average_ic,
        }
