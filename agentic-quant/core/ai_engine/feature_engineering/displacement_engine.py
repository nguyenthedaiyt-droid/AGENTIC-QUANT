"""Displacement Engine - SMC Displacement Detection.

Port tu Pine Script: f_highlightDisplacement(), GetFVGDisplacementLevel.
Source: pinescript/InstitutionalOrderFlow.pine (lines 456-474, 484-492)

Thuat toan:
  body  = |open - close|
  std   = ta.stdev(body, length=100)
  D     = body[1] / (std * factor)
  displaced = body[1] > std * factor
  Bullish FVG: low > high[2]
  Bearish FVG: high < low[2]
"""
from __future__ import annotations

import numpy as np

from core.ai_engine.feature_engineering.types import (
    DisplacementConfig,
    DisplacementResult,
)


def compute_displacement(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    config: DisplacementConfig,
) -> DisplacementResult:
    """Tinh displacement cho candle hien tai.

    Pine Script: f_highlightDisplacement()
      candle_range = math.abs(open - close)
      fvg = close[1] > open[1] ? high[2] < low : low[2] > high
      displaced = displacement_fvg
        ? candle_range[1] > std[1] * displacement_factor and fvg
        : candle_range > std

    Args:
        opens:   mang open prices, index 0 = current bar
        highs:   mang high prices
        lows:    mang low prices
        closes:  mang close prices
        config:  displacement config (length, factor, require_fvg)
    """
    n = len(opens)
    if n < 3:
        return DisplacementResult()

    # body[i] = |open[i] - close[i]|
    body = np.abs(opens - closes)

    # std = stdev(body, length) - tao rolling window
    length = config.length
    if n >= length:
        recent_bodies = body[:length]
    else:
        recent_bodies = body

    std_val = float(np.std(recent_bodies)) if len(recent_bodies) > 1 else 0.0

    # Candle direction: bull = open < close (green), bear = open > close (red)
    bullish_candle = opens[1] < closes[1]  # bar[1] = previous bar

    # D_strength = body[1] / (std * factor)
    d_strength = 0.0
    if std_val > 0 and config.factor > 0:
        d_strength = body[1] / (std_val * config.factor)

    # Displacement: body[1] > std * factor
    displaced = body[1] > std_val * config.factor

    # FVG confirmation
    bullish_fvg = lows[1] > highs[2]
    bearish_fvg = highs[1] < lows[2]
    fvg_confirmed = bullish_fvg or bearish_fvg

    # Pine Script: displacement_fvg ? (displaced AND fvg) : displaced
    if config.require_fvg:
        displaced = displaced and fvg_confirmed

    return DisplacementResult(
        is_displaced=bool(displaced),
        is_bullish=bool(bullish_candle),
        d_strength=float(d_strength),
        fvg_confirmed=bool(fvg_confirmed),
    )


def compute_d_strength_vector(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    configs: list[DisplacementConfig],
) -> np.ndarray:
    """Tinh D_strength vector (5 chiều) cho 5 displacement configs.

    Pine Script: D_strength_vector (5 dims, each config = {length, factor})

    Args:
        configs: list of 5 DisplacementConfig (length, factor)

    Returns:
        np.ndarray[5] - d_strength values
    """
    results = [
        compute_displacement(opens, highs, lows, closes, cfg)
        for cfg in configs
    ]
    return np.array([r.d_strength for r in results], dtype=np.float64)


def rolling_std_bodies(
    opens: np.ndarray,
    closes: np.ndarray,
    length: int = 100,
) -> np.ndarray:
    """Rolling standard deviation cua body sizes.

    Pine Script: std = ta.stdev(body, length)
    Tra ve mang std theo tung vi tri (dung cho FVG displacement filter).
    """
    body = np.abs(opens - closes)
    n = len(body)
    stds = np.zeros(n, dtype=np.float64)

    for i in range(length, n):
        window = body[i - length:i]
        stds[i] = float(np.std(window))

    return stds


class DisplacementEngine:
    """Displacement Engine - wrapper cho displacement calculations.

    Pine Script: displacement settings + f_highlightDisplacement()
    """

    def __init__(
        self,
        length: int = 100,
        factor: int = 2,
        require_fvg: bool = True,
    ) -> None:
        self.config = DisplacementConfig(
            length=length,
            factor=factor,
            require_fvg=require_fvg,
        )
        self._body_std_cache: np.ndarray | None = None

    def compute(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> DisplacementResult:
        """Tinh displacement cho current bar."""
        return compute_displacement(opens, highs, lows, closes, self.config)

    def compute_vector(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> np.ndarray:
        """Tinh displacement vector (5 configs)."""
        configs = [
            DisplacementConfig(length=100, factor=1, require_fvg=False),
            DisplacementConfig(length=100, factor=2, require_fvg=False),
            DisplacementConfig(length=100, factor=3, require_fvg=False),
            DisplacementConfig(length=100, factor=1, require_fvg=True),
            DisplacementConfig(length=100, factor=2, require_fvg=True),
        ]
        return compute_d_strength_vector(opens, highs, lows, closes, configs)

    @staticmethod
    def fvg_displacement_level(level: int) -> DisplacementConfig:
        """Factory: FVG displacement level 1-4 (tu GetFVGDisplacementLevel)."""
        return DisplacementConfig.fvg_level(level)

    @staticmethod
    def fvg_level_from_setting(setting: str, displacement_factor: int) -> DisplacementConfig:
        """Parse tu Pine Script fvg_type setting.

        Pine Script GetFVGDisplacementLevel:
          'Same As Displacement' -> displacement_factor
          'Level 1' -> 1, 'Level 2' -> 2, etc.
        """
        mapping = {
            "Same As Displacement": displacement_factor,
            "Level 1": 1,
            "Level 2": 2,
            "Level 3": 3,
            "Level 4": 4,
        }
        factor = mapping.get(setting, displacement_factor)
        return DisplacementConfig(factor=factor, require_fvg=True)
