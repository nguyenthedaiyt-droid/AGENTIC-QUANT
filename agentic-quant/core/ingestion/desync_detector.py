# =============================================================================
# AGENTIC-QUANT — Timeframe Desync Detector
# Phat hien va xu ly desync giua cac khung thoi gian
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from loguru import logger

from core.ingestion import TIMEFRAME_SECONDS, BarState


class DesyncSeverity(str, Enum):
    """Muc do nghiem trong cua desync."""

    NONE = "NONE"
    MINOR = "MINOR"      # 1-2 bars thieu
    MODERATE = "MODERATE"  # 3-5 bars thieu
    SEVERE = "SEVERE"    # > 5 bars thieu hoac TF chinh bi desync


@dataclass
class DesyncReport:
    """Bao cao desync cho mot timeframe."""

    timeframe: str
    severity: DesyncSeverity
    missing_bars: int
    expected_count: int
    actual_count: int
    first_missing_bucket: int
    last_missing_bucket: int
    interpolated: bool  # Da noi suy chua chua
    reliability: str    # "RELIABLE" | "INTERPOLATED" | "UNRELIABLE"


class TimeframeDesyncDetector:
    """
    Phat hien va xu ly desync giua cac timeframe.

    Co che hoat dong:
    1. Tinh so bar da dong mong doi trong khoang thoi gian
    2. So sanh voi so bar thuc te co
    3. Neu thieu <= 5 bars: noi suy tuyến tinh
    4. Neu thieu > 5 bars: danh dau UNRELIABLE

    Args:
        max_interpolation: So bars toi da de noi suy (default: 5)
        desync_threshold_minor: So bars thieu de thanh MINOR (default: 1)
        desync_threshold_moderate: So bars thieu de thanh MODERATE (default: 3)
    """

    def __init__(
        self,
        max_interpolation: int = 5,
        desync_threshold_minor: int = 1,
        desync_threshold_moderate: int = 3,
    ) -> None:
        self._max_interpolation = max_interpolation
        self._threshold_minor = desync_threshold_minor
        self._threshold_moderate = desync_threshold_moderate

        # Cache desync reports
        self._reports: dict[str, DesyncReport] = {}
        self._last_check_time: int = 0

    def check_desync(
        self,
        timeframe: str,
        bars: dict[int, BarState],
        reference_time: int,
        lookback_buckets: int,
    ) -> DesyncReport:
        """
        Kiem tra desync cho mot timeframe.

        Args:
            timeframe: Ten timeframe (M1, M5, ...)
            bars: Dict tat ca bars da dong {bucket_time: BarState}
            reference_time: Timestamp hien tai (seconds)
            lookback_buckets: So bucket de kiem tra

        Returns:
            DesyncReport
        """
        tf_sec = TIMEFRAME_SECONDS.get(timeframe, 60)
        latest_bucket = reference_time - (reference_time % tf_sec)

        # Tinh cac bucket mong muon
        expected_buckets = set()
        for i in range(lookback_buckets):
            bucket = latest_bucket - (lookback_buckets - 1 - i) * tf_sec
            expected_buckets.add(bucket)

        # Tim cac bucket thuc te
        actual_buckets = {
            b.bucket_time for b in bars.values() if b.is_closed
        }

        # Tim gap
        missing_buckets = expected_buckets - actual_buckets

        missing_count = len(missing_buckets)

        # Xac dinh severity
        if missing_count == 0:
            severity = DesyncSeverity.NONE
            reliability = "RELIABLE"
        elif missing_count <= self._threshold_minor:
            severity = DesyncSeverity.MINOR
            reliability = "RELIABLE"
        elif missing_count <= self._threshold_moderate:
            severity = DesyncSeverity.MODERATE
            reliability = "RELIABLE"
        elif missing_count <= self._max_interpolation:
            severity = DesyncSeverity.MODERATE
            reliability = "INTERPOLATED"
        else:
            severity = DesyncSeverity.SEVERE
            reliability = "UNRELIABLE"

        first_missing = min(missing_buckets) if missing_buckets else latest_bucket
        last_missing = max(missing_buckets) if missing_buckets else latest_bucket

        report = DesyncReport(
            timeframe=timeframe,
            severity=severity,
            missing_bars=missing_count,
            expected_count=lookback_buckets,
            actual_count=len(actual_buckets),
            first_missing_bucket=first_missing,
            last_missing_bucket=last_missing,
            interpolated=severity != DesyncSeverity.NONE and missing_count <= self._max_interpolation,
            reliability=reliability,
        )

        self._reports[timeframe] = report
        return report

    def check_all_timeframes(
        self,
        all_bars: dict[str, dict[int, BarState]],
        reference_time: int,
        lookback_buckets: int = 100,
    ) -> dict[str, DesyncReport]:
        """
        Kiem tra desync cho tat ca timeframe.

        Args:
            all_bars: Dict tat ca bars theo timeframe
            reference_time: Timestamp hien tai
            lookback_buckets: So bucket de kiem tra

        Returns:
            Dict reports cho moi timeframe
        """
        results = {}
        for tf, bars in all_bars.items():
            report = self.check_desync(tf, bars, reference_time, lookback_buckets)
            results[tf] = report

            if report.severity != DesyncSeverity.NONE:
                logger.warning(
                    "Desync detected: {tf} missing {n}/{e} bars "
                    "({severity}) — reliability={rel}",
                    tf=tf,
                    n=report.missing_bars,
                    e=report.expected_count,
                    severity=report.severity.value,
                    rel=report.reliability,
                )

        return results

    def interpolate_missing_bar(
        self,
        prev_bar: BarState,
        next_bar: BarState,
        missing_bucket_time: int,
    ) -> BarState:
        """
        Noi suy bar thieu bang linear interpolation.

        Cong thuc:
        open = prev_close
        close = next_open
        high = max(prev_close, next_open)
        low = min(prev_close, next_open)
        volume = (prev.volume + next.volume) / 2

        Args:
            prev_bar: Bar truoc gap
            next_bar: Bar sau gap
            missing_bucket_time: Bucket time cua bar thieu

        Returns:
            BarState da noi suy
        """
        return BarState(
            open=prev_bar.close,
            high=max(prev_bar.close, next_bar.open),
            low=min(prev_bar.close, next_bar.open),
            close=next_bar.open,
            volume=(prev_bar.volume + next_bar.volume) / 2,
            tick_count=0,
            bucket_time=missing_bucket_time,
            is_closed=True,
        )

    def fill_gaps(
        self,
        bars: dict[int, BarState],
        tf_sec: int,
        reference_time: int,
    ) -> dict[int, BarState]:
        """
        Dien gap trong bars voi interpolated bars.

        Chi goi khi reliability la INTERPOLATED.

        Args:
            bars: Dict bars hien co
            tf_sec: So giay cua timeframe
            reference_time: Timestamp hien tai

        Returns:
            Bars da duoc fill gap
        """
        if not bars:
            return bars

        result = dict(bars)
        sorted_bars = sorted(bars.items(), key=lambda x: x[0])

        # Tim gap
        for i in range(len(sorted_bars) - 1):
            curr_time, curr_bar = sorted_bars[i]
            next_time, next_bar = sorted_bars[i + 1]

            expected_next = curr_time + tf_sec
            if next_time > expected_next:
                # Co gap
                missing_count = (next_time - expected_next) // tf_sec
                if missing_count <= self._max_interpolation:
                    for j in range(missing_count):
                        gap_bucket = expected_next + j * tf_sec
                        interp_bar = self.interpolate_missing_bar(
                            curr_bar, next_bar, gap_bucket
                        )
                        result[gap_bucket] = interp_bar
                    logger.debug(
                        "Interpolated {n} bars between {t1} and {t2}",
                        n=missing_count,
                        t1=curr_time,
                        t2=next_time,
                    )

        return result

    def get_unreliable_timeframes(self) -> list[str]:
        """Tra ve danh sach cac TF co reliability = UNRELIABLE."""
        return [
            tf for tf, report in self._reports.items()
            if report.reliability == "UNRELIABLE"
        ]

    @property
    def reports(self) -> dict[str, DesyncReport]:
        """Tat ca reports."""
        return self._reports
