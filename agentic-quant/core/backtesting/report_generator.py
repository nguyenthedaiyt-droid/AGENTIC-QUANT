# =============================================================================
# AGENTIC-QUANT — Backtest Report Generator (Phase 8)
# Tong hop IC stats, drift stats, regime shift log, overfitting check
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from core.backtesting.ic_calculator import ICCalculator
    from core.backtesting.drift_detector import DriftDetector
    from core.backtesting.regime_shift_detector import RegimeShiftDetector


# =============================================================================
# Constants
# =============================================================================
_OVERFITTING_WARNING_THRESHOLD = 0.15  # Gap > 0.15 -> overfitting warning


# =============================================================================
# BacktestReport DataClass
# =============================================================================
@dataclass
class BacktestReport:
    """Bao cao day du cho mot lan backtest.

    Attributes:
        report_id: ID duy nhat cua report
        generated_at: Thoi gian tao report
        symbol: Ma symbol
        timeframe: Khung thoi gian (neu co)
        start_time: Thoi gian bat dau backtest
        end_time: Thoi gian ket thuc backtest
        ic_stats: Thong ke IC (mean, std, min, max, rolling)
        drift_stats: Thong ke drift (FDS, PSI values, features)
        regime_shift_log: Lich su regime shift
        overfitting_check: Ket qua kiem tra overfitting
        trades: Danh sach trades (neu co)
    """
    report_id: str = ""
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Meta
    symbol: str = ""
    timeframe: str = ""
    start_time: datetime | None = None
    end_time: datetime | None = None

    # IC
    ic_stats: dict[str, Any] = field(default_factory=dict)
    ic_by_regime: dict[str, float] = field(default_factory=dict)
    ic_by_session: dict[str, float] = field(default_factory=dict)
    ic_by_impact: dict[str, float] = field(default_factory=dict)
    rolling_ic: list[float] = field(default_factory=list)

    # Drift
    drift_stats: dict[str, Any] = field(default_factory=dict)
    psi_values: dict[str, float] = field(default_factory=dict)
    drift_features: list[str] = field(default_factory=list)
    model_degraded: bool = False

    # Regime
    regime_shift_log: list[dict] = field(default_factory=list)
    regime_changes: int = 0

    # Overfitting
    overfitting_check: dict[str, Any] = field(default_factory=dict)

    # Trades
    trades: list[dict] = field(default_factory=list)
    total_trades: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Chuyen report thanh dict de serialize.

        Returns:
            Dict chua tat ca fields
        """
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "ic_stats": self.ic_stats,
            "ic_by_regime": self.ic_by_regime,
            "ic_by_session": self.ic_by_session,
            "ic_by_impact": self.ic_by_impact,
            "rolling_ic": self.rolling_ic,
            "drift_stats": self.drift_stats,
            "psi_values": self.psi_values,
            "drift_features": self.drift_features,
            "model_degraded": self.model_degraded,
            "regime_shift_log": self.regime_shift_log,
            "regime_changes": self.regime_changes,
            "overfitting_check": self.overfitting_check,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
        }

    def summary(self) -> str:
        """Tao summary text cho report.

        Returns:
            String summary
        """
        lines = [
            f"=== BacktestReport: {self.symbol} ===",
            f"Period: {self.start_time} -> {self.end_time}",
            f"Generated: {self.generated_at}",
            "",
            "--- IC Stats ---",
            f"  Mean IC: {self.ic_stats.get('mean_ic', 'N/A')}",
            f"  IC Std:  {self.ic_stats.get('std_ic', 'N/A')}",
            f"  IC > 0:  {self.ic_stats.get('ic_positive_ratio', 'N/A') * 100:.1f}%",
            "",
            "--- Drift ---",
            f"  FDS: {self.drift_stats.get('fds', 'N/A')}",
            f"  Degraded: {self.model_degraded}",
            f"  Drifted features: {len(self.drift_features)}",
            "",
            "--- Regime ---",
            f"  Regime changes: {self.regime_changes}",
            "",
            "--- Overfitting Check ---",
        ]
        if self.overfitting_check:
            of = self.overfitting_check
            lines.append(f"  IC backtest: {of.get('ic_backtest', 'N/A')}")
            lines.append(f"  IC forward:  {of.get('ic_forward', 'N/A')}")
            lines.append(f"  Gap: {of.get('gap', 'N/A')}")
            lines.append(f"  Warning: {of.get('warning', 'N/A')}")
        else:
            lines.append("  (Not checked)")

        lines.extend([
            "",
            "--- Trades ---",
            f"  Total: {self.total_trades}",
            f"  Win rate: {self.win_rate:.2%}",
            f"  Sharpe: {self.sharpe_ratio:.2f}",
            f"  Max DD: {self.max_drawdown:.2%}",
        ])

        return "\n".join(lines)


# =============================================================================
# BacktestReportGenerator
# =============================================================================
class BacktestReportGenerator:
    """Tao BacktestReport tu cac component: IC, drift, regime, trades.

    Args:
        ic_calculator: ICCalculator instance
        drift_detector: DriftDetector instance
        regime_detector: RegimeShiftDetector instance
    """

    def __init__(
        self,
        ic_calculator: ICCalculator | None = None,
        drift_detector: DriftDetector | None = None,
        regime_detector: RegimeShiftDetector | None = None,
    ) -> None:
        self._ic_calc = ic_calculator
        self._drift = drift_detector
        self._regime = regime_detector

        # Internal state
        self._symbol: str = ""
        self._timeframe: str = ""
        self._start_time: datetime | None = None
        self._end_time: datetime | None = None

        # IC data
        self._ic_values: list[float] = []
        self._ic_values_forward: list[float] = []
        self._ic_values_backtest: list[float] = []
        self._ic_by_regime: dict[str, list[float]] = {}
        self._ic_by_session: dict[str, list[float]] = {}
        self._ic_by_impact: dict[str, list[float]] = {}
        self._rolling_ic: list[float] = []

        # Trades
        self._trades: list[dict] = []

    # -------------------------------------------------------------------------
    # Public API: Gather data
    # -------------------------------------------------------------------------
    def set_meta(
        self,
        symbol: str,
        timeframe: str = "",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> None:
        """Set thong tin meta cho report.

        Args:
            symbol: Ma symbol
            timeframe: Khung thoi gian
            start_time: Thoi gian bat dau
            end_time: Thoi gian ket thuc
        """
        self._symbol = symbol
        self._timeframe = timeframe
        self._start_time = start_time
        self._end_time = end_time

    def add_ic_values(
        self,
        y_hats: list[float],
        y_actuals: list[float],
        regimes: list[str] | None = None,
        sessions: list[str] | None = None,
        impacts: list[str] | None = None,
    ) -> None:
        """Them IC values tu predictions.

        Args:
            y_hats: Predictions
            y_actuals: Actual values
            regimes: Regime labels tuong ung (optional)
            sessions: Session labels tuong ung (optional)
            impacts: Impact levels tuong ung (optional)
        """
        if not self._ic_calc:
            logger.warning("ICCalculator khong co san, bo qua add_ic_values")
            return

        if not y_hats or not y_actuals:
            return

        # Tinh IC cho tung cap
        for i in range(len(y_hats)):
            ic = self._ic_calc.compute_ic(
                [y_hats[i]],
                [y_actuals[i]],
            )
            self._ic_values.append(ic)

            # Phan loai theo regime
            if regimes and i < len(regimes):
                regime = regimes[i]
                if regime not in self._ic_by_regime:
                    self._ic_by_regime[regime] = []
                self._ic_by_regime[regime].append(ic)

            # Phan loai theo session
            if sessions and i < len(sessions):
                session = sessions[i]
                if session not in self._ic_by_session:
                    self._ic_by_session[session] = []
                self._ic_by_session[session].append(ic)

            # Phan loai theo impact
            if impacts and i < len(impacts):
                impact = impacts[i]
                if impact not in self._ic_by_impact:
                    self._ic_by_impact[impact] = []
                self._ic_by_impact[impact].append(ic)

    def add_rolling_ic(self, rolling_ic: list[float]) -> None:
        """Them rolling IC values.

        Args:
            rolling_ic: List rolling IC values
        """
        self._rolling_ic = list(rolling_ic)

    def add_trades(self, trades: list[dict]) -> None:
        """Them danh sach trades.

        Args:
            trades: List trade dicts
        """
        self._trades = list(trades)

    def set_ic_splits(
        self,
        backtest: list[float],
        forward: list[float],
    ) -> None:
        """Set IC values cho backtest va forward period.

        Args:
            backtest: IC values trong backtest period
            forward: IC values trong forward/out-of-sample period
        """
        self._ic_values_backtest = list(backtest)
        self._ic_values_forward = list(forward)

    # -------------------------------------------------------------------------
    # Build Report
    # -------------------------------------------------------------------------
    def build(self) -> BacktestReport:
        """Xay dung BacktestReport tu cac du lieu da gather.

        Returns:
            BacktestReport: Bao cao day du

        Raises:
            RuntimeError: Neu thieu du lieu can thiet
        """
        from datetime import datetime, timezone

        report = BacktestReport(
            report_id=f"BT_{self._symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.now(timezone.utc),
            symbol=self._symbol,
            timeframe=self._timeframe,
            start_time=self._start_time,
            end_time=self._end_time,
        )

        # --- IC Stats ---
        report.ic_stats = self._compute_ic_stats()
        report.ic_by_regime = self._compute_grouped_ic(self._ic_by_regime)
        report.ic_by_session = self._compute_grouped_ic(self._ic_by_session)
        report.ic_by_impact = self._compute_grouped_ic(self._ic_by_impact)
        report.rolling_ic = self._rolling_ic

        # --- Drift Stats ---
        if self._drift:
            report.drift_stats = {
                "fds": self._drift.last_fds,
                "n_checks": self._drift.get_stats().get("n_checks", 0),
                "n_degraded": self._drift.get_stats().get("n_degraded", 0),
                "degradation_rate": self._drift.get_degradation_rate(),
            }
            report.psi_values = dict(self._drift.last_psi)
            report.drift_features = [
                name for name, psi in self._drift.last_psi.items()
                if psi > 0.2
            ]
            report.model_degraded = self._drift.check_model_degraded(
                self._drift.last_fds,
            )
        else:
            report.drift_stats = {}
            report.psi_values = {}
            report.drift_features = []
            report.model_degraded = False

        # --- Regime Shift Log ---
        if self._regime:
            report.regime_shift_log = self._regime.shift_log
            report.regime_changes = self._regime.shift_count
        else:
            report.regime_shift_log = []
            report.regime_changes = 0

        # --- Overfitting Check ---
        report.overfitting_check = self._overfitting_check(
            self._ic_values_backtest,
            self._ic_values_forward,
        )

        # --- Trades ---
        report.trades = self._trades
        report.total_trades = len(self._trades)
        report.win_rate = self._compute_win_rate()
        report.sharpe_ratio = self._compute_sharpe()
        report.max_drawdown = self._compute_max_drawdown()

        logger.info(
            "[ReportGenerator] Report {id}: IC_mean={ic:.4f}, "
            "trades={t}, FDS={fds}",
            id=report.report_id,
            ic=report.ic_stats.get("mean_ic", 0.0),
            t=report.total_trades,
            fds=report.drift_stats.get("fds", 0.0),
        )

        return report

    # -------------------------------------------------------------------------
    # Internal: IC stats
    # -------------------------------------------------------------------------
    def _compute_ic_stats(self) -> dict[str, Any]:
        """Tinh thong ke IC.

        Returns:
            Dict: mean_ic, std_ic, min_ic, max_ic, ic_positive_ratio
        """
        if not self._ic_values:
            return {
                "mean_ic": 0.0,
                "std_ic": 0.0,
                "min_ic": 0.0,
                "max_ic": 0.0,
                "positive_count": 0,
                "total_count": 0,
                "ic_positive_ratio": 0.0,
            }

        import numpy as np

        arr = np.array(self._ic_values, dtype=np.float64)

        n_positive = int(np.sum(arr > 0))
        n_total = len(arr)

        return {
            "mean_ic": float(np.mean(arr)),
            "std_ic": float(np.std(arr)),
            "min_ic": float(np.min(arr)),
            "max_ic": float(np.max(arr)),
            "positive_count": n_positive,
            "total_count": n_total,
            "ic_positive_ratio": n_positive / n_total if n_total > 0 else 0.0,
        }

    def _compute_grouped_ic(
        self,
        grouped_data: dict[str, list[float]],
    ) -> dict[str, float]:
        """Tinh trung binh IC cho tung nhom.

        Args:
            grouped_data: Dict {group_name: [IC_values]}

        Returns:
            Dict {group_name: mean_IC}
        """
        import numpy as np

        result: dict[str, float] = {}
        for group_name, values in grouped_data.items():
            if values:
                arr = np.array(values, dtype=np.float64)
                result[group_name] = float(np.mean(arr))
            else:
                result[group_name] = 0.0
        return result

    # -------------------------------------------------------------------------
    # Internal: Overfitting Check
    # -------------------------------------------------------------------------
    def _overfitting_check(
        self,
        ic_backtest: list[float],
        ic_forward: list[float],
    ) -> dict[str, Any]:
        """Kiem tra overfitting bang so sanh IC backtest vs forward.

        Warning duoc phat ra neu gap > 0.15.

        Args:
            ic_backtest: IC values trong backtest (in-sample)
            ic_forward: IC values trong forward (out-of-sample)

        Returns:
            Dict: ic_backtest, ic_forward, gap, warning
        """
        import numpy as np

        if not ic_backtest:
            return {
                "ic_backtest": 0.0,
                "ic_forward": 0.0,
                "gap": 0.0,
                "gap_exceeds_threshold": False,
                "warning": "Khong co IC backtest data",
            }

        backtest_mean = float(np.mean(ic_backtest))
        forward_mean = float(np.mean(ic_forward)) if ic_forward else 0.0
        gap = abs(backtest_mean - forward_mean)

        warning_flag = gap > _OVERFITTING_WARNING_THRESHOLD
        warning_msg = (
            f"CANH BAO OVERFITTING: Gap IC = {gap:.4f} > "
            f"{_OVERFITTING_WARNING_THRESHOLD}. "
            f"Backtest IC = {backtest_mean:.4f}, Forward IC = {forward_mean:.4f}"
        ) if warning_flag else (
            f"OK: Gap IC = {gap:.4f} <= {_OVERFITTING_WARNING_THRESHOLD}. "
            f"Backtest IC = {backtest_mean:.4f}, Forward IC = {forward_mean:.4f}"
        )

        if warning_flag:
            logger.warning(warning_msg)
        else:
            logger.info(warning_msg)

        return {
            "ic_backtest": round(backtest_mean, 4),
            "ic_forward": round(forward_mean, 4),
            "gap": round(gap, 4),
            "gap_exceeds_threshold": warning_flag,
            "threshold": _OVERFITTING_WARNING_THRESHOLD,
            "warning": warning_msg,
            "n_backtest": len(ic_backtest),
            "n_forward": len(ic_forward),
        }

    # -------------------------------------------------------------------------
    # Internal: Trade metrics
    # -------------------------------------------------------------------------
    def _compute_win_rate(self) -> float:
        """Tinh win rate tu danh sach trades.

        Returns:
            float: Win rate (0.0 -> 1.0)
        """
        if not self._trades:
            return 0.0

        wins = sum(
            1 for t in self._trades
            if t.get("pnl", 0) > 0 or t.get("outcome", "") in ("WIN", "BSL_HIT", "PROFIT")
        )
        return wins / len(self._trades)

    def _compute_sharpe(self) -> float:
        """Tinh Sharpe ratio tu danh sach trades.

        Returns:
            float: Sharpe ratio
        """
        import numpy as np

        if len(self._trades) < 2:
            return 0.0

        pnls = np.array(
            [t.get("pnl", 0) for t in self._trades],
            dtype=np.float64,
        )
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)

        if std_pnl < 1e-12:
            return 0.0

        # Sharpe = mean(pnl) / std(pnl) * sqrt(periods)
        # Mac dinh period scaling = 1 (raw)
        return float(mean_pnl / std_pnl)

    def _compute_max_drawdown(self) -> float:
        """Tinh max drawdown tu danh sach trades.

        Returns:
            float: Max drawdown (0.0 -> 1.0)
        """
        import numpy as np

        if not self._trades:
            return 0.0

        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0

        for t in self._trades:
            pnl = t.get("pnl", 0)
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            dd = (peak - cumulative) / (peak + 1e-12)
            if dd > max_dd:
                max_dd = dd

        return float(max_dd)
