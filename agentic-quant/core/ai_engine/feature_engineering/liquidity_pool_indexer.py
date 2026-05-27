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
        """Convert sang F_struct[64]."""
        vec = np.zeros(64, dtype=np.float64)
        idx = 0

        # [0-23] F_liq[24]
        if self.f_liq is not None and len(self.f_liq) == 24:
            vec[0:24] = self.f_liq
        idx = 24

        # [24-31] Structure events (8 dims)
        vec[idx] = float(self.mss_bullish_count)
        idx += 1
        vec[idx] = float(self.mss_bearish_count)
        idx += 1
        vec[idx] = float(self.bos_bullish_count)
        idx += 1
        vec[idx] = float(self.bos_bearish_count)
        idx += 1
        vec[idx] = self.structure_score
        idx += 1
        vec[idx] = float(self.mss_bullish_count - self.mss_bearish_count)  # bias
        idx += 1
        vec[idx] = float(self.bos_bullish_count - self.bos_bearish_count)  # bias
        idx += 1
        vec[idx] = 1.0 if self.structure_score > 0 else 0.0  # bullish context
        idx += 1

        # [32-47] FVG/OB metrics (16 dims)
        vec[idx] = float(self.fvg_bullish_active)
        idx += 1
        vec[idx] = float(self.fvg_bearish_active)
        idx += 1
        vec[idx] = float(self.fvg_mitigated)
        idx += 1
        vec[idx] = float(self.ifvg_active)
        idx += 1
        vec[idx] = float(self.ob_bullish_active)
        idx += 1
        vec[idx] = float(self.ob_bearish_active)
        idx += 1
        vec[idx] = self.fvg_avg_strength
        idx += 1
        vec[idx] = self.fvg_avg_range
        idx += 1
        vec[idx] = float(self.zone_premium_count)
        idx += 1
        vec[idx] = float(self.zone_discount_count)
        idx += 1
        # Normalized FVG density
        total_fvg = self.fvg_bullish_active + self.fvg_bearish_active + 1
        vec[idx] = self.fvg_bullish_active / total_fvg
        idx += 1
        vec[idx] = float(self.fvg_mitigated) / max(self.fvg_bullish_active + self.fvg_bearish_active, 1)
        idx += 1
        # FVG position relative to price
        if self.current_price > 0 and self.equilibrium > 0:
            price_ratio = (self.current_price - self.equilibrium) / self.equilibrium
            vec[idx] = float(price_ratio)
            idx += 1
        else:
            idx += 1
        # Unmitigated ratio
        total = self.fvg_bullish_active + self.fvg_bearish_active + self.fvg_mitigated + 1
        vec[idx] = (self.fvg_bullish_active + self.fvg_bearish_active) / total
        idx += 1
        # Bull/Bear imbalance
        total_zones = self.zone_premium_count + self.zone_discount_count + 1
        vec[idx] = self.zone_premium_count / total_zones
        idx += 1

        # [48-63] EQ status (8 dims)
        vec[idx] = float(self.eq_total)
        idx += 1
        vec[idx] = float(self.eq_unclaimed)
        idx += 1
        vec[idx] = float(self.eq_claimed)
        idx += 1
        vec[idx] = float(self.eq_high_count)
        idx += 1
        vec[idx] = float(self.eq_low_count)
        idx += 1
        # EQ bias
        if self.eq_total > 0:
            vec[idx] = self.eq_high_count / self.eq_total
            idx += 1
        else:
            idx += 1
        # EQ claimed ratio
        if self.eq_total > 0:
            vec[idx] = self.eq_claimed / self.eq_total
            idx += 1
        else:
            idx += 1
        # EQ density
        vec[idx] = float(self.eq_total) / max(self.fvg_bullish_active + self.fvg_bearish_active, 1)
        idx += 1

        return vec

    def to_f_agg(self) -> np.ndarray:
        """Convert sang F_agg[16]."""
        vec = np.zeros(16, dtype=np.float64)
        idx = 0

        # [0-3] Zone density metrics
        total_zones = self.fvg_bullish_active + self.fvg_bearish_active + self.ob_bullish_active + self.ob_bearish_active
        vec[idx] = float(total_zones)
        idx += 1
        vec[idx] = self.fvg_avg_strength * total_zones  # weighted strength
        idx += 1
        vec[idx] = self.fvg_avg_range * total_zones
        idx += 1
        vec[idx] = float(self.fvg_bullish_active + self.fvg_bearish_active)
        idx += 1

        # [4-7] Structure alignment
        vec[idx] = self.structure_score
        idx += 1
        bullish_struct = self.mss_bullish_count * 2 + self.bos_bullish_count
        bearish_struct = self.mss_bearish_count * 2 + self.bos_bearish_count
        vec[idx] = float(bullish_struct - bearish_struct)
        idx += 1
        vec[idx] = float(bullish_struct + bearish_struct)  # total structure events
        idx += 1
        vec[idx] = float(bullish_struct) / max(bullish_struct + bearish_struct, 1)
        idx += 1

        # [8-11] Displacement
        vec[idx] = self.d_strength_mean
        idx += 1
        vec[idx] = self.d_strength_max
        idx += 1
        vec[idx] = 1.0 if self.d_strength_max > 1.0 else 0.0  # displaced flag
        idx += 1
        vec[idx] = self.d_strength_mean * self.d_strength_max
        idx += 1

        # [12-15] Price context
        if self.current_price > 0 and self.equilibrium > 0:
            vec[idx] = (self.current_price - self.equilibrium) / self.equilibrium
        idx += 1
        if self.current_price > 0 and self.atr > 0:
            vec[idx] = self.atr / self.current_price  # normalized ATR
        idx += 1
        vec[idx] = float(self.zone_premium_count - self.zone_discount_count)
        idx += 1
        # Combined score
        struct_aligned = self.structure_score * 0.3
        fvg_aligned = (self.fvg_bullish_active - self.fvg_bearish_active) * 0.3
        eq_aligned = (self.eq_high_count - self.eq_low_count) * 0.2
        displacement = self.d_strength_mean * 0.2
        vec[idx] = float(struct_aligned + fvg_aligned + eq_aligned + displacement)
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
