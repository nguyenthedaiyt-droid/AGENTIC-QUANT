"""Liquidity Pool Indexer - Feature Vector Builder.

TODO 4.6: Tong hop BSL/SSL tu tat ca TF, tinh V_acc_zone, III_zone.
TODO 4.6.2: build_f_struct_vector -> np.ndarray[64]
TODO 4.6.3: build_f_agg_vector -> np.ndarray[16]

F_struct[64] = [
  24 dims: F_liq (from BSLSSLRegistry)
  16 dims: structure events (MSS/BOS)
  16 dims: FVG/OB metrics
  8 dims: EQ status
]

F_agg[16] = [
  Zone density, structure alignment score, displacement strength, etc.
]
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from core.ai_engine.feature_engineering.types import (
    StructureMap,
    FVGCollection,
    Imbalance,
    ImbalanceType,
    EqualLevel,
    FeatureVectors,
    FeatureEngineeringConfig,
)
from core.ai_engine.feature_engineering.bsl_ssl_registry import BSLSSLRegistry
from core.ai_engine.feature_engineering.ict_structure_mapper import ICTStructureMapper
from core.ai_engine.feature_engineering.fvg_ob_scanner import FVGOBScanner
from core.ai_engine.feature_engineering.equal_levels_detector import EqualLevelsDetector


@dataclass
class SymbolicFeatureMap:
    """Symbolic feature map (tu TODO 4.6.2).

    Tom tat tat ca features symbolic tu cac detector.
    """
    # Liquidity
    f_liq: np.ndarray | None = None  # [24]

    # Structure
    mss_bullish_count: int = 0
    mss_bearish_count: int = 0
    bos_bullish_count: int = 0
    bos_bearish_count: int = 0
    structure_score: float = 0.0  # MSS * 2 + BOS * 1

    # FVG/OB
    fvg_bullish_active: int = 0
    fvg_bearish_active: int = 0
    fvg_mitigated: int = 0
    ifvg_active: int = 0
    ob_bullish_active: int = 0
    ob_bearish_active: int = 0
    fvg_avg_strength: float = 0.0
    fvg_avg_range: float = 0.0
    zone_premium_count: int = 0
    zone_discount_count: int = 0

    # Equal Levels
    eq_total: int = 0
    eq_claimed: int = 0
    eq_unclaimed: int = 0
    eq_high_count: int = 0
    eq_low_count: int = 0

    # Displacement
    d_strength_mean: float = 0.0
    d_strength_max: float = 0.0

    # Price context
    current_price: float = 0.0
    equilibrium: float = 0.0
    atr: float = 0.0

    def to_f_struct(self) -> np.ndarray:
        """Convert sang F_struct[64] theo agentic_quant_full_plan.md.

        Dims [0–23]:  F_liq[24] (Tier-based ST/IT/LT)
        Dims [24–39]: Structure events (16 dims)
        Dims [40–55]: FVG/OB metrics (16 dims)
        Dims [56–63]: EQ status (8 dims)
        """
        vec = np.zeros(64, dtype=np.float64)
        idx = 0

        # [0-23] F_liq[24] - da duoc populate tu SymbolicFeatureMap.f_liq
        if self.f_liq is not None and len(self.f_liq) == 24:
            vec[0:24] = self.f_liq
        idx = 24

        # [24-39] Structure events (16 dims)
        # MSS_bull_count, MSS_bear_count, BOS_bull_count, BOS_bear_count
        vec[idx] = float(self.mss_bullish_count); idx += 1
        vec[idx] = float(self.mss_bearish_count); idx += 1
        vec[idx] = float(self.bos_bullish_count); idx += 1
        vec[idx] = float(self.bos_bearish_count); idx += 1

        # last_MSS_strength, last_MSS_bars_ago, HTF_trend_H4, HTF_trend_H1
        vec[idx] = self.structure_score; idx += 1
        vec[idx] = 0.0; idx += 1  # bars_ago - can tracking
        vec[idx] = 0.0; idx += 1  # HTF_trend_H4 - can HTF data
        vec[idx] = 0.0; idx += 1  # HTF_trend_H1

        # fib_382_dist, fib_618_dist, eq_dist, premium_flag, discount_flag
        vec[idx] = 0.0; idx += 1  # fib_382_dist
        vec[idx] = 0.0; idx += 1  # fib_618_dist
        eq_dist = (self.current_price - self.equilibrium) / (self.equilibrium + 1e-9) if self.equilibrium > 0 else 0.0
        vec[idx] = eq_dist; idx += 1
        vec[idx] = 1.0 if eq_dist > 0 else 0.0; idx += 1  # premium_flag
        vec[idx] = 1.0 if eq_dist < 0 else 0.0; idx += 1  # discount_flag

        # structure_alignment_HTF, structure_alignment_MTF, reserved
        vec[idx] = self.structure_score / max(self.mss_bullish_count + self.mss_bearish_count, 1); idx += 1
        vec[idx] = self.structure_score / max(self.bos_bullish_count + self.bos_bearish_count, 1); idx += 1
        idx += 1  # reserved
        idx = 40

        # [40-55] FVG/OB metrics (16 dims)
        # fvg_bull_count, fvg_bear_count, nearest_fvg_dist, nearest_fvg_strength
        vec[idx] = float(self.fvg_bullish_active); idx += 1
        vec[idx] = float(self.fvg_bearish_active); idx += 1
        vec[idx] = 0.0; idx += 1  # nearest_fvg_dist - can tracking
        vec[idx] = self.fvg_avg_strength; idx += 1

        # nearest_fvg_ce, fvg_p_hold, ob_bull_count, ob_bear_count
        vec[idx] = 0.0; idx += 1  # nearest_fvg_ce - can tracking
        vec[idx] = 0.5; idx += 1  # fvg_p_hold - default
        vec[idx] = float(self.ob_bullish_active); idx += 1
        vec[idx] = float(self.ob_bearish_active); idx += 1

        # nearest_ob_dist, ob_p_hold, ifvg_count, mitigation_rate_last20
        vec[idx] = 0.0; idx += 1  # nearest_ob_dist
        vec[idx] = 0.5; idx += 1  # ob_p_hold
        vec[idx] = float(self.ifvg_active); idx += 1
        total = self.fvg_bullish_active + self.fvg_bearish_active + self.fvg_mitigated + 1
        vec[idx] = float(self.fvg_mitigated) / total; idx += 1

        # displacement_avg, w_zone_fvg, w_zone_ob, reserved
        vec[idx] = self.d_strength_mean; idx += 1
        vec[idx] = 1.5 if self.zone_premium_count > self.zone_discount_count else 1.0; idx += 1  # w_zone_fvg
        vec[idx] = 1.5 if self.zone_premium_count > self.zone_discount_count else 1.0; idx += 1  # w_zone_ob
        idx += 1  # reserved
        idx = 56

        # [56-63] EQ status (8 dims)
        # eq_high_count, eq_low_count, nearest_eq_high_dist, nearest_eq_low_dist
        vec[idx] = float(self.eq_high_count); idx += 1
        vec[idx] = float(self.eq_low_count); idx += 1
        vec[idx] = 0.0; idx += 1  # nearest_eq_high_dist
        vec[idx] = 0.0; idx += 1  # nearest_eq_low_dist

        # eq_high_age_bars, eq_low_age_bars, eq_high_claimed_ratio, eq_low_claimed_ratio
        vec[idx] = 0.0; idx += 1  # eq_high_age_bars
        vec[idx] = 0.0; idx += 1  # eq_low_age_bars
        total_eq = self.eq_high_count + self.eq_low_count + 1
        vec[idx] = float(self.eq_claimed) / total_eq; idx += 1
        vec[idx] = float(self.eq_unclaimed) / total_eq; idx += 1

        return vec

    def to_f_agg(self) -> np.ndarray:
        """Convert sang F_agg[16] theo agentic_quant_full_plan.md.

        [zone_density, structure_alignment_score, displacement_strength_avg,
         bsl_density_score, ssl_density_score, premium_discount_bias,
         news_regime_factor, session_weight_ltf, session_weight_htf,
         cvd_alignment_score, iii_zone_max, claimed_rate_acceleration,
         mss_recency_score, ob_confluence_score, fvg_confluence_score, reserved]
        """
        vec = np.zeros(16, dtype=np.float64)
        idx = 0

        # [0] zone_density - tong so zones
        total_zones = (self.fvg_bullish_active + self.fvg_bearish_active +
                       self.ob_bullish_active + self.ob_bearish_active)
        vec[idx] = float(total_zones); idx += 1

        # [1] structure_alignment_score
        bullish_struct = self.mss_bullish_count * 2 + self.bos_bullish_count
        bearish_struct = self.mss_bearish_count * 2 + self.bos_bearish_count
        vec[idx] = float(bullish_struct - bearish_struct); idx += 1

        # [2] displacement_strength_avg
        vec[idx] = self.d_strength_mean; idx += 1

        # [3] bsl_density_score (can du lieu them tu BSLSSLRegistry)
        vec[idx] = 0.0; idx += 1

        # [4] ssl_density_score
        vec[idx] = 0.0; idx += 1

        # [5] premium_discount_bias
        vec[idx] = float(self.zone_premium_count - self.zone_discount_count); idx += 1

        # [6] news_regime_factor
        vec[idx] = 1.0; idx += 1  # default: normal regime

        # [7] session_weight_ltf
        vec[idx] = 1.0; idx += 1

        # [8] session_weight_htf
        vec[idx] = 1.0; idx += 1

        # [9] cvd_alignment_score
        vec[idx] = 0.0; idx += 1

        # [10] iii_zone_max
        vec[idx] = 0.0; idx += 1

        # [11] claimed_rate_acceleration
        total_pivots = self.eq_total + 1
        vec[idx] = float(self.eq_claimed) / total_pivots; idx += 1

        # [12] mss_recency_score
        vec[idx] = float(self.mss_bullish_count + self.mss_bearish_count); idx += 1

        # [13] ob_confluence_score
        vec[idx] = float(self.ob_bullish_active + self.ob_bearish_active); idx += 1

        # [14] fvg_confluence_score
        total_fvg_active = self.fvg_bullish_active + self.fvg_bearish_active
        vec[idx] = float(total_fvg_active) * self.fvg_avg_strength; idx += 1

        # [15] reserved
        idx += 1

        return vec


class LiquidityPoolIndexer:
    """Liquidity Pool Indexer.

    Tong hop BSL/SSL tu tat ca TFs, tinh V_acc_zone, III_zone.
    Xay dung F_struct[64] va F_agg[16] feature vectors.

    Usage:
      indexer = LiquidityPoolIndexer()
      fmap = indexer.build_symbolic_feature_map(
          pivots=pivots,
          structure_map=structure_map,
          fvg_collection=fvg_collection,
          eq_levels=eq_levels,
          current_price=current_price,
          atr=atr,
      )
      f_struct = indexer.build_f_struct_vector(fmap)
      f_agg = indexer.build_f_agg_vector(fmap)
    """

    def __init__(
        self,
        bsl_ssl_registry: BSLSSLRegistry | None = None,
        ict_mapper: ICTStructureMapper | None = None,
        fvg_scanner: FVGOBScanner | None = None,
        eq_detector: EqualLevelsDetector | None = None,
    ) -> None:
        self.bsl_ssl = bsl_ssl_registry
        self.ict_mapper = ict_mapper
        self.fvg_scanner = fvg_scanner
        self.eq_detector = eq_detector

    def build_symbolic_feature_map(
        self,
        pivots: list | None = None,
        structure_map: StructureMap | None = None,
        fvg_collection: FVGCollection | None = None,
        eq_levels: list[EqualLevel] | None = None,
        current_price: float = 0.0,
        atr: float = 0.0,
        equilibrium: float = 0.0,
        f_liq: np.ndarray | None = None,
        d_strength_mean: float = 0.0,
        d_strength_max: float = 0.0,
    ) -> SymbolicFeatureMap:
        """Build SymbolicFeatureMap (TODO 4.6.2).

        Args:
            pivots: Danh sach pivots tu SwingPointDetector
            structure_map: StructureMap tu ICTStructureMapper
            fvg_collection: FVGCollection tu FVGOBScanner
            eq_levels: EqualLevels tu EqualLevelsDetector
            current_price: Gia hien tai
            atr: Gia tri ATR hien tai
            equilibrium: Equilibrium tu ICTStructureMapper
            f_liq: F_liq[24] tu BSLSSLRegistry
            d_strength_mean: D_strength trung binh
            d_strength_max: D_strength max

        Returns:
            SymbolicFeatureMap day du
        """
        fmap = SymbolicFeatureMap(
            current_price=current_price,
            equilibrium=equilibrium,
            atr=atr,
            f_liq=f_liq,
            d_strength_mean=d_strength_mean,
            d_strength_max=d_strength_max,
        )

        # Structure events
        if structure_map:
            mss_bull = sum(
                1 for e in structure_map.mss_events
                if e.event_type.value.startswith("BULLISH")
            )
            mss_bear = sum(
                1 for e in structure_map.mss_events
                if e.event_type.value.startswith("BEARISH")
            )
            bos_bull = sum(
                1 for e in structure_map.bos_events
                if e.event_type.value.startswith("BULLISH")
            )
            bos_bear = sum(
                1 for e in structure_map.bos_events
                if e.event_type.value.startswith("BEARISH")
            )
            fmap.mss_bullish_count = mss_bull
            fmap.mss_bearish_count = mss_bear
            fmap.bos_bullish_count = bos_bull
            fmap.bos_bearish_count = bos_bear
            fmap.structure_score = float(mss_bull * 2 + mss_bear * 2 + bos_bull + bos_bear)
            fmap.zone_premium_count = len(structure_map.premium_zones)
            fmap.zone_discount_count = len(structure_map.discount_zones)

        # FVG/OB
        if fvg_collection:
            fvgs = fvg_collection.fvgs
            ifvgs = fvg_collection.ifvgs

            fmap.fvg_bullish_active = sum(
                1 for f in fvgs if f.is_bullish and not f.mitigated and not f.inverted
            )
            fmap.fvg_bearish_active = sum(
                1 for f in fvgs if not f.is_bullish and not f.mitigated and not f.inverted
            )
            fmap.fvg_mitigated = sum(1 for f in fvgs if f.mitigated)
            fmap.ifvg_active = sum(1 for f in ifvgs if not f.mitigated)

            active = [f for f in fvgs if not f.mitigated]
            if active:
                fmap.fvg_avg_strength = float(
                    np.mean([f.strength for f in active])
                )
                fmap.fvg_avg_range = float(
                    np.mean([f.range_size for f in active])
                )

        # Equal Levels
        if eq_levels:
            fmap.eq_total = len(eq_levels)
            fmap.eq_claimed = sum(1 for e in eq_levels if e.is_claimed)
            fmap.eq_unclaimed = sum(1 for e in eq_levels if not e.is_claimed)
            fmap.eq_high_count = sum(1 for e in eq_levels if e.is_high)
            fmap.eq_low_count = sum(1 for e in eq_levels if e.is_low)

        return fmap

    def build_f_struct_vector(
        self,
        feature_map: SymbolicFeatureMap,
    ) -> np.ndarray:
        """Build F_struct[64] (TODO 4.6.2)."""
        return feature_map.to_f_struct()

    def build_f_agg_vector(
        self,
        feature_map: SymbolicFeatureMap,
    ) -> np.ndarray:
        """Build F_agg[16] (TODO 4.6.3)."""
        return feature_map.to_f_agg()

    def apply_session_weights(
        self,
        f_struct: np.ndarray,
        config: FeatureEngineeringConfig,
    ) -> np.ndarray:
        """Apply session weights theo Overview doc Phan IV.1.

        Theo agentic_quant_full_plan.md:
          - FVG/OB LTF × ltf_signal_weight
          - FVG/OB HTF × htf_signal_weight

        F_struct[40] = fvg_p_hold
        F_struct[45] = fvg_p_hold (theo mapping moi)
        F_struct[50] = ob_p_hold

        Args:
            f_struct: F_struct[64] vector
            config: FeatureEngineeringConfig voi session weights

        Returns:
            F_struct[64] da duoc apply weights
        """
        if f_struct is None or len(f_struct) < 64:
            return f_struct

        result = f_struct.copy()

        # fvg_p_hold (index 45)
        result[45] *= config.session_ltf_weight
        # ob_p_hold (index 49)
        result[49] *= config.session_ltf_weight

        # F_agg session weights (index 7-8)
        # (f_agg duoc tinh rieng, khong nam trong f_struct)

        return result

    def build_feature_vectors(
        self,
        feature_map: SymbolicFeatureMap,
    ) -> FeatureVectors:
        """Build all feature vectors at once."""
        return FeatureVectors(
            f_struct=self.build_f_struct_vector(feature_map).tolist(),
            f_agg=self.build_f_agg_vector(feature_map).tolist(),
            f_liq=feature_map.f_liq.tolist() if feature_map.f_liq is not None else [],
            d_strength=[feature_map.d_strength_mean, feature_map.d_strength_max],
        )

    def compute_v_acc_zone(
        self,
        pivots: list,
        cvd_values: np.ndarray,
        epsilon_pct: float = 0.001,
    ) -> dict[str, float]:
        """Tinh V_acc_zone: sum CVD trong vung ±ε quanh moi liquidity level.

        Args:
            pivots: List BSL/SSL pivots
            cvd_values: Mang CVD values
            epsilon_pct: % khoang cach quanh level

        Returns:
            Dict zone_id -> V_acc value
        """
        v_acc = {}
        if not pivots or len(cvd_values) == 0:
            return v_acc

        for pivot in pivots:
            epsilon = pivot.price * epsilon_pct
            # Find bars in ±ε around pivot.price
            start_idx = 0
            end_idx = len(cvd_values)

            # V_acc = sum of CVD changes in zone
            total_cvd = float(np.sum(np.abs(np.diff(cvd_values[start_idx:end_idx]))))
            v_acc[pivot.price] = total_cvd

        return v_acc

    def compute_iii_zone(
        self,
        fvgs: list[Imbalance],
        iii_values: np.ndarray,
    ) -> dict[str, float]:
        """Tinh III_zone: sum III trong vung FVG/OB.

        Args:
            fvgs: List FVG/OB imbalances
            iii_values: Mang III values

        Returns:
            Dict zone_id -> III value
        """
        iii_acc = {}
        if not fvgs or len(iii_values) == 0:
            return iii_acc

        for fvg in fvgs:
            # Approximate: III trong zone = sum III values
            total_iii = float(np.sum(iii_values))
            iii_acc[fvg.id] = total_iii

        return iii_acc
