# =============================================================================
# AGENTIC-QUANT — XGBoost Feature Builder (Phase 6.0)
# Build X_A[648] and X_B[560] vectors from raw features
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


# =============================================================================
# Dimension constants
# =============================================================================
D_Z = 512        # LSTM latent embedding
D_LIQ = 24       # F_liq — liquidity features
D_FLOW = 20      # F_flow — flow features (CVD, III, DIV, OBI, etc.)
D_MACRO = 12     # F_macro — macro features (news, surprise, session, regime, etc.)
D_STRUCT = 64    # F_struct — structure features (FVG, EQ, displacement, etc.)
D_AGG = 16       # F_agg — aggregated features
D_ZONE = 16      # F_zone — zone features (zone size, age, type, p_hold, etc.)
D_CONTACT = 20   # F_contact — contact features (CE, body, CVD at contact, etc.)

# Derived sizes
D_X_A = D_Z + D_LIQ + D_FLOW + D_MACRO + D_STRUCT + D_AGG   # 512+24+20+12+64+16 = 648
D_X_B = D_Z + D_ZONE + D_CONTACT + D_MACRO                   # 512+16+20+12 = 560


@dataclass
class XGBoostFeatures:
    """Container for Model A and Model B feature vectors.

    Attributes:
        X_A: Feature vector for Model A — shape (648,)
              concat(z[512], F_liq[24], F_flow[20], F_macro[12], F_struct[64], F_agg[16])
        X_B: Feature vector for Model B — shape (560,)
              concat(z[512], F_zone[16], F_contact[20], F_macro[12])
        symbol: Trading symbol
        timeframe: Timeframe string
        timestamp_ms: Bar close timestamp
    """
    X_A: np.ndarray = field(default_factory=lambda: np.zeros(D_X_A, dtype=np.float64))
    X_B: np.ndarray = field(default_factory=lambda: np.zeros(D_X_B, dtype=np.float64))
    symbol: str = ""
    timeframe: str = ""
    timestamp_ms: int = 0


class XGBoostFeatureBuilder:
    """Build X_A and X_B feature vectors for XGBoost models.

    Phan III.8 — Feature Engineering Overview:
      X_A[648] = concat(z[512], F_liq[24], F_flow[20], F_macro[12], F_struct[64], F_agg[16])
      X_B[560] = concat(z[512], F_zone[16], F_contact[20], F_macro[12])

    F_flow = [CVD_norm, III_t, DIV_CVD, OBI...] 20 dims
    F_macro = [I_news, S(surprise), session_code, regime_code...] 12 dims
    F_zone = [zone_size, age_bars, tf_code, type_code, p_hold_history...] 16 dims
    F_contact = [CE_at_contact, body_size_at_contact, CVD_at_contact...] 20 dims
    """

    def __init__(self) -> None:
        self._last_X_A: np.ndarray = np.zeros(D_X_A, dtype=np.float64)
        self._last_X_B: np.ndarray = np.zeros(D_X_B, dtype=np.float64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        z: np.ndarray,               # [512] — LSTM latent
        f_liq: np.ndarray | None,     # [24] — liquidity features
        f_flow: np.ndarray | None,    # [20] — flow features
        f_macro: np.ndarray | None,   # [12] — macro features
        f_struct: np.ndarray | None,  # [64] — structure features
        f_agg: np.ndarray | None,     # [16] — aggregated features
        f_zone: np.ndarray | None,    # [16] — zone features
        f_contact: np.ndarray | None, # [20] — contact features
        symbol: str = "",
        timeframe: str = "",
        timestamp_ms: int = 0,
    ) -> XGBoostFeatures:
        """Build both X_A and X_B feature vectors.

        Args:
            z: LSTM latent embedding [512]
            f_liq: Liquidity features [24] — from BSLSSLRegistry
            f_flow: Flow features [20] — CVD, III, DIV, OBI, etc.
            f_macro: Macro features [12] — news, surprise, session, regime
            f_struct: Structure features [64] — from LiquidityPoolIndexer
            f_agg: Aggregated features [16] — from LiquidityPoolIndexer
            f_zone: Zone features [16] — zone size, age, type, p_hold
            f_contact: Contact features [20] — CE, body, CVD at contact
            symbol: Trading symbol
            timeframe: Timeframe string
            timestamp_ms: Bar close timestamp

        Returns:
            XGBoostFeatures with X_A, X_B, and metadata
        """
        z = self._validate_vector(z, D_Z, "z")

        # X_A = concat(z[512], F_liq[24], F_flow[20], F_macro[12], F_struct[64], F_agg[16])
        f_liq = self._validate_vector(f_liq, D_LIQ, "f_liq")
        f_flow = self._validate_vector(f_flow, D_FLOW, "f_flow")
        f_macro = self._validate_vector(f_macro, D_MACRO, "f_macro")
        f_struct = self._validate_vector(f_struct, D_STRUCT, "f_struct")
        f_agg = self._validate_vector(f_agg, D_AGG, "f_agg")

        X_A = np.concatenate([z, f_liq, f_flow, f_macro, f_struct, f_agg])
        assert X_A.shape[0] == D_X_A, f"X_A shape mismatch: {X_A.shape[0]} != {D_X_A}"

        # X_B = concat(z[512], F_zone[16], F_contact[20], F_macro[12])
        f_zone = self._validate_vector(f_zone, D_ZONE, "f_zone")
        f_contact = self._validate_vector(f_contact, D_CONTACT, "f_contact")
        f_macro_b = self._validate_vector(f_macro, D_MACRO, "f_macro (B)")

        X_B = np.concatenate([z, f_zone, f_contact, f_macro_b])
        assert X_B.shape[0] == D_X_B, f"X_B shape mismatch: {X_B.shape[0]} != {D_X_B}"

        self._last_X_A = X_A.copy()
        self._last_X_B = X_B.copy()

        return XGBoostFeatures(
            X_A=X_A,
            X_B=X_B,
            symbol=symbol,
            timeframe=timeframe,
            timestamp_ms=timestamp_ms,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_vector(arr: np.ndarray | None, expected_d: int, name: str) -> np.ndarray:
        """Validate or create zero-fill vector."""
        if arr is None:
            return np.zeros(expected_d, dtype=np.float64)
        arr = np.asarray(arr, dtype=np.float64).ravel()
        if arr.shape[0] != expected_d:
            # Pad or truncate to expected dimension
            if arr.shape[0] < expected_d:
                padded = np.zeros(expected_d, dtype=np.float64)
                padded[:arr.shape[0]] = arr
                return padded
            return arr[:expected_d]
        return arr

    @property
    def last_X_A(self) -> np.ndarray:
        """Last built X_A vector (copy)."""
        return self._last_X_A.copy()

    @property
    def last_X_B(self) -> np.ndarray:
        """Last built X_B vector (copy)."""
        return self._last_X_B.copy()

    def reset(self) -> None:
        """Reset internal state."""
        self._last_X_A = np.zeros(D_X_A, dtype=np.float64)
        self._last_X_B = np.zeros(D_X_B, dtype=np.float64)

    @classmethod
    def build_synthetic_z(cls, batch_size: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        """Generate synthetic LSTM latent for testing."""
        if rng is None:
            rng = np.random.default_rng(42)
        return rng.standard_normal((batch_size, D_Z), dtype=np.float64)

    @classmethod
    def build_synthetic_f_flow(cls, batch_size: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        """Generate synthetic flow features for testing.

        F_flow = [CVD_norm, III_t, DIV_CVD, OBI...] 20 dims
        """
        if rng is None:
            rng = np.random.default_rng(42)
        features = rng.standard_normal((batch_size, D_FLOW), dtype=np.float64)
        # CVD_norm in [-1, 1]
        features[:, 0] = np.clip(features[:, 0], -1.0, 1.0)
        return features

    @classmethod
    def build_synthetic_f_macro(cls, batch_size: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        """Generate synthetic macro features for testing.

        F_macro = [I_news, S(surprise), session_code, regime_code...] 12 dims
        """
        if rng is None:
            rng = np.random.default_rng(42)
        features = rng.standard_normal((batch_size, D_MACRO), dtype=np.float64)
        session_codes = np.array([0, 1, 2, 3, 4, 5])  # ASIAN, LONDON_OPEN, LONDON, NY_OPEN, NY_AM, NY_PM
        features[:, 2] = rng.choice(session_codes, size=batch_size)
        regime_codes = np.array([0, 1, 2, 3])  # NORMAL, TRENDING_LV, TRENDING_HV, CHOPPY_HV
        features[:, 3] = rng.choice(regime_codes, size=batch_size)
        # I_news in [0, 1]
        features[:, 0] = rng.uniform(0.0, 1.0, size=batch_size)
        return features

    @classmethod
    def build_synthetic_f_zone(cls, batch_size: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        """Generate synthetic zone features for testing.

        F_zone = [zone_size, age_bars, tf_code, type_code, p_hold_history...] 16 dims
        """
        if rng is None:
            rng = np.random.default_rng(42)
        features = rng.standard_normal((batch_size, D_ZONE), dtype=np.float64)
        # p_hold in [0, 1]
        features[:, 4] = np.clip(rng.uniform(0.0, 1.0, size=batch_size), 0.0, 1.0)
        # tf_code one-hot-ish
        tf_codes = np.array([0, 1, 2, 3, 4, 5])
        features[:, 2] = rng.choice(tf_codes, size=batch_size)
        return features

    @classmethod
    def build_synthetic_f_contact(cls, batch_size: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        """Generate synthetic contact features for testing.

        F_contact = [CE_at_contact, body_size_at_contact, CVD_at_contact...] 20 dims
        """
        if rng is None:
            rng = np.random.default_rng(42)
        features = rng.standard_normal((batch_size, D_CONTACT), dtype=np.float64)
        # CE in [0, 100]
        features[:, 0] = np.clip(features[:, 0], 0.0, 100.0)
        return features
