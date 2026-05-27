# =============================================================================
# AGENTIC-QUANT — Backtesting & Drift Evaluation Module (Phase 8)
# =============================================================================

from .ic_calculator import ICCalculator
from .drift_detector import DriftDetector
from .regime_shift_detector import RegimeShiftDetector
from .report_generator import BacktestReport, BacktestReportGenerator
from .event_driven_simulator import EventDrivenSimulator
from .fine_tuning import FineTuningPipeline

# BacktestEngine la lop tong hop — export tu day
# (se duoc implement trong file rieng neu can mo rong sau nay)
class BacktestEngine:
    """BacktestEngine tong hop: EventDrivenSimulator + ICCalculator + DriftDetector + ReportGenerator.

    Day la facade de chay toan bo pipeline backtest tu A->Z.
    """

    def __init__(
        self,
        simulator: EventDrivenSimulator | None = None,
        ic_calculator: ICCalculator | None = None,
        drift_detector: DriftDetector | None = None,
        regime_detector: RegimeShiftDetector | None = None,
        report_generator: BacktestReportGenerator | None = None,
    ) -> None:
        self.simulator = simulator or EventDrivenSimulator()
        self.ic_calculator = ic_calculator or ICCalculator()
        self.drift_detector = drift_detector or DriftDetector()
        self.regime_detector = regime_detector or RegimeShiftDetector()
        self.report_generator = report_generator or BacktestReportGenerator()

    async def run(
        self,
        symbol: str,
        start: str,
        end: str,
        **kwargs,
    ) -> BacktestReport:
        """Chay toan bo backtest pipeline.

        Args:
            symbol: Ma symbol (VD: XAUUSD)
            start: Thoi gian bat dau (YYYY-MM-DD)
            end: Thoi gian ket thuc (YYYY-MM-DD)
            **kwargs: Them tham so cho simulator

        Returns:
            BacktestReport: Bao cao day du
        """
        # 1. Replay ticks
        from loguru import logger
        logger.info(f"BacktestEngine.run({symbol}, {start}, {end})")
        await self.simulator.run(symbol, start, end, **kwargs)
        return self.report_generator.build()


__all__ = [
    "BacktestEngine",
    "EventDrivenSimulator",
    "ICCalculator",
    "DriftDetector",
    "RegimeShiftDetector",
    "BacktestReport",
    "BacktestReportGenerator",
    "FineTuningPipeline",
]
