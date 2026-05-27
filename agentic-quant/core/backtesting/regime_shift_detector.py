# =============================================================================
# AGENTIC-QUANT — Regime Shift Detector (Phase 8)
# Phat hien su thay doi regime (Trending / Choppy / Normal) dua vao vol_ratio va ADX
# =============================================================================

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from loguru import logger

from core.utils.events import EventBus, EventType, RegimeChangeEvent

if TYPE_CHECKING:
    pass


# =============================================================================
# Enum: Regime types
# =============================================================================
class RegimeType(str, Enum):
    """Cac loai regime cua thi truong."""

    TRENDING_LV = "TRENDING_LV"  # Xu huong, bien dong thap (low volatility)
    TRENDING_HV = "TRENDING_HV"  # Xu huong, bien dong cao (high volatility)
    CHOPPY_HV = "CHOPPY_HV"      # Di ngang, bien dong cao (sideways, high vol)
    NORMAL = "NORMAL"            # Binh thuong, khong dac biet


# =============================================================================
# Constants / Thresholds
# =============================================================================
# Nguong ADX phan biet trending vs choppy
_ADX_TRENDING_THRESHOLD = 25.0

# Nguong Volatility Ratio (current_vol / average_vol)
_VOL_RATIO_HIGH = 1.5  # Bien dong cao gap 1.5 lan trung binh
_VOL_RATIO_LOW = 0.7   # Bien dong thap = 0.7 lan trung binh

# Symbol mapping
_REGIME_TO_EVENT_TYPE = {
    RegimeType.TRENDING_LV: "TRENDING_LV",
    RegimeType.TRENDING_HV: "TRENDING_HV",
    RegimeType.CHOPPY_HV: "CHOPPY_HV",
    RegimeType.NORMAL: "NORMAL",
}


# =============================================================================
# RegimeShiftDetector
# =============================================================================
class RegimeShiftDetector:
    """Phat hien regime shift dua vao volatility ratio va ADX.

    Thresholds (co the tuy chinh):
    - ADX >= 25: Trending
        - Vol ratio >= 1.5: TRENDING_HV
        - Vol ratio < 1.5: TRENDING_LV
    - ADX < 25: Non-trending
        - Vol ratio >= 1.5: CHOPPY_HV
        - Vol ratio < 1.5: NORMAL

    Args:
        event_bus: EventBus instance de publish REGIME_SHIFT_DETECTED
    """

    def __init__(
        self,
        event_bus: EventBus | None = None,
    ) -> None:
        self._bus = event_bus
        self._current_regime: RegimeType = RegimeType.NORMAL
        self._prev_regime: RegimeType = RegimeType.NORMAL
        self._shift_count: int = 0
        self._shift_log: list[dict] = []

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    @property
    def current_regime(self) -> str:
        """Regime hien tai."""
        return self._current_regime.value

    @property
    def previous_regime(self) -> str:
        """Regime truoc do."""
        return self._prev_regime.value

    @property
    def shift_count(self) -> int:
        """So lan regime shift da phat hien."""
        return self._shift_count

    def detect_regime(
        self,
        vol_ratio: float,
        adx_14: float,
    ) -> RegimeType:
        """Phat hien regime dua vao volatility ratio va ADX.

        Args:
            vol_ratio: Ty le volatility hien tai so voi trung binh
                       (current_vol / average_vol)
            adx_14: ADX 14 period value (0-100)

        Returns:
            RegimeType: Loai regime phat hien duoc

        Raises:
            ValueError: Neu vol_ratio hoac adx_14 khong hop le
        """
        if vol_ratio < 0:
            raise ValueError(f"vol_ratio khong duoc am: {vol_ratio}")
        if not 0 <= adx_14 <= 100:
            raise ValueError(
                f"ADX phai nam trong [0, 100], nhan duoc: {adx_14}"
            )

        # Decision tree
        if adx_14 >= _ADX_TRENDING_THRESHOLD:
            # Trending
            if vol_ratio >= _VOL_RATIO_HIGH:
                regime = RegimeType.TRENDING_HV
            else:
                regime = RegimeType.TRENDING_LV
        else:
            # Non-trending
            if vol_ratio >= _VOL_RATIO_HIGH:
                regime = RegimeType.CHOPPY_HV
            else:
                regime = RegimeType.NORMAL

        return regime

    def check_shift(
        self,
        new_regime: RegimeType,
        prev_regime: RegimeType | None = None,
    ) -> bool:
        """Kiem tra co regime shift khong, publish event neu co.

        Args:
            new_regime: Regime moi phat hien
            prev_regime: Regime truoc do (None = dung self._current_regime)

        Returns:
            bool: True neu co shift, False neu khong
        """
        if prev_regime is None:
            prev_regime = self._current_regime

        if new_regime == prev_regime:
            # Khong co shift
            return False

        # Co regime shift
        self._prev_regime = prev_regime
        self._current_regime = new_regime
        self._shift_count += 1

        # Log shift
        shift_info = {
            "shift_number": self._shift_count,
            "prev_regime": prev_regime.value,
            "new_regime": new_regime.value,
        }
        self._shift_log.append(shift_info)

        logger.info(
            "[RegimeShift] #{n}: {old} -> {new}",
            n=self._shift_count,
            old=prev_regime.value,
            new=new_regime.value,
        )

        # Publish REGIME_SHIFT_DETECTED event
        if self._bus:
            try:
                event = RegimeChangeEvent(
                    previous_regime=prev_regime.value,
                    new_regime=new_regime.value,
                    trigger_reason="backtest_regime_shift",
                )
                self._bus.publish(event)

                logger.debug(
                    "[RegimeShift] Published REGIME_CHANGE event: {old} -> {new}",
                    old=prev_regime.value,
                    new=new_regime.value,
                )
            except Exception as exc:
                logger.warning(
                    "[RegimeShift] Loi publish event: {exc}",
                    exc=exc,
                )

        return True

    # -------------------------------------------------------------------------
    # Shift Log
    # -------------------------------------------------------------------------
    @property
    def shift_log(self) -> list[dict]:
        """Lich su cac regime shift.

        Returns:
            List dict cac shift da xay ra
        """
        return list(self._shift_log)

    def reset(self) -> None:
        """Reset detector ve trang thai ban dau."""
        self._current_regime = RegimeType.NORMAL
        self._prev_regime = RegimeType.NORMAL
        self._shift_count = 0
        self._shift_log.clear()
        logger.info("[RegimeShift] Reset complete")
