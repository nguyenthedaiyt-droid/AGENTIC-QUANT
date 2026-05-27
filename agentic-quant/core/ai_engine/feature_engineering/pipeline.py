"""FeatureEngineeringPipeline - Orchestrator.

TODO 9: Tong hop tat ca detectors, hook vao BAR_CLOSE event.
Publish zone creation events vao EventBus.

Pipeline flow:
  BAR_CLOSE(M1) event
    -> SwingPointDetector.detect() -> upsert to Zone Registry
    -> ICTStructureMapper.get_structure_map()
    -> FVGOBScanner.scan_fvg() + FVGStateMachine -> persist zones to Redis
    -> EqualLevelsDetector.process_pivots()
    -> DisplacementEngine.compute_vector()
    -> LiquidityPoolIndexer.build_feature_vectors()
    -> Apply session weights
    -> Publish FeatureVectors + Zone events
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.memory.short_term.redis_cache_manager import RedisCacheManager
    from core.memory.short_term.active_zone_registry import ActiveZoneRegistry
    from core.ai_engine.feature_engineering.types import (
        Pivot,
        StructureMap,
        FVGCollection,
        FeatureVectors,
        EqualLevel,
        Imbalance,
        FeatureEngineeringConfig,
    )

from core.ai_engine.feature_engineering.smc_detector import SwingPointDetector
from core.ai_engine.feature_engineering.fvg_ob_scanner import FVGOBScanner, FVGConfig
from core.ai_engine.feature_engineering.equal_levels_detector import EqualLevelsDetector
from core.ai_engine.feature_engineering.displacement_engine import DisplacementEngine
from core.ai_engine.feature_engineering.ict_structure_mapper import ICTStructureMapper
from core.ai_engine.feature_engineering.bsl_ssl_registry import BSLSSLRegistry
from core.ai_engine.feature_engineering.liquidity_pool_indexer import (
    LiquidityPoolIndexer,
    SymbolicFeatureMap,
)
from core.ai_engine.feature_engineering.types import (
    FeatureVectors,
    FeatureEngineeringConfig,
    DisplacementConfig,
)


@dataclass
class PipelineResult:
    """Ket qua cua FeatureEngineeringPipeline.process()."""
    symbol: str
    timeframe: str
    bar_close_time: int

    pivots: list = field(default_factory=list)
    structure_map: object | None = None
    fvg_collection: object | None = None
    equal_levels: list = field(default_factory=list)

    f_struct: np.ndarray | None = None
    f_agg: np.ndarray | None = None
    f_liq: np.ndarray | None = None
    d_strength: np.ndarray | None = None

    zones_created: list = field(default_factory=list)
    pivots_claimed: list = field(default_factory=list)

    current_price: float = 0.0
    equilibrium: float = 0.0
    atr: float = 0.0

    def to_feature_vectors(self) -> FeatureVectors:
        """Convert sang FeatureVectors dataclass."""
        return FeatureVectors(
            f_struct=self.f_struct.tolist() if self.f_struct is not None else [],
            f_agg=self.f_agg.tolist() if self.f_agg is not None else [],
            f_liq=self.f_liq.tolist() if self.f_liq is not None else [],
            d_strength=self.d_strength.tolist() if self.d_strength is not None else [],
        )


class FeatureEngineeringPipeline:
    """Feature Engineering Pipeline Orchestrator.

    Tong hop tat ca SMC detectors thanh mot pipeline xu ly BAR_CLOSE.

    Args:
        redis: RedisCacheManager (optional, cho Zone Registry)
        zone_registry: ActiveZoneRegistry (optional)
        config: FeatureEngineeringConfig
    """

    def __init__(
        self,
        redis: "RedisCacheManager | None" = None,
        zone_registry: "ActiveZoneRegistry | None" = None,
        config: "FeatureEngineeringConfig | None" = None,
    ) -> None:
        self._redis = redis
        self._zone_registry = zone_registry
        self.config = config or FeatureEngineeringConfig()
        self._atr_cache: float = 0.0
        self._news_guardrail_active: bool = False

        # Initialize all detectors
        self._sp_detector = SwingPointDetector()
        fvg_cfg = FVGConfig.from_fe_config(self.config)
        self._fvg_scanner = FVGOBScanner(fvg_cfg)
        self._eq_detector = EqualLevelsDetector(tolerance=self.config.eq_tolerance)
        self._displacement = DisplacementEngine(
            length=self.config.displacement.length,
            factor=self.config.displacement.factor,
            require_fvg=self.config.displacement.require_fvg,
        )
        self._ict_mapper = ICTStructureMapper()
        self._bsl_ssl: BSLSSLRegistry | None = None
        self._liq_indexer = LiquidityPoolIndexer()

        if redis and zone_registry:
            self._bsl_ssl = BSLSSLRegistry(redis, zone_registry)

    def reset(self) -> None:
        """Reset all internal state."""
        self._sp_detector.reset()
        self._fvg_scanner.reset()
        self._eq_detector.reset()
        self._ict_mapper.reset()
        self._atr_cache = 0.0

    def on_news_alert(self, active_guardrail: bool) -> None:
        """Hook vao NEWS_ALERT event (News Guardrail).

        Theo agentic_quant_full_plan.md:
          - Khi NEWS_ALERT active: disable LTF FVG/OB scanning
          - Config.news_guardrail_active = True
        """
        self._news_guardrail_active = active_guardrail
        self.config.news_guardrail_active = active_guardrail
        self._fvg_scanner.on_news_alert(active_guardrail)

    @property
    def news_guardrail_active(self) -> bool:
        return self._news_guardrail_active

    def process_bar(
        self,
        symbol: str,
        timeframe: str,
        bar_close_time: int,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        times: np.ndarray | None = None,
        atr: float = 0.0,
    ) -> PipelineResult:
        """Xu ly mot bar close, tra ve tat ca features (sync).

        Args:
            symbol: Symbol trading (VD: "XAUUSD")
            timeframe: Timeframe (VD: "M15")
            bar_close_time: Timestamp ms khi bar dong
            opens/highs/lows/closes: OHLC arrays (index 0 = current bar)
            times: Array timestamps (ms)
            atr: ATR hien tai

        Returns:
            PipelineResult chua tat ca features
        """
        self._atr_cache = atr
        self._fvg_scanner.set_atr(atr)

        current_price = float(closes[0]) if len(closes) > 0 else 0.0
        result = PipelineResult(
            symbol=symbol,
            timeframe=timeframe,
            bar_close_time=bar_close_time,
            current_price=current_price,
            atr=atr,
        )

        # News Guardrail: neu active, chi xu ly HTF features
        if self._news_guardrail_active:
            return self._compute_htf_only(result, opens, highs, lows, closes, times)

        # 1. Swing Point Detection
        pivots = self._sp_detector.detect(highs, lows, times, bar_close_time)

        # 2. ICT Structure Map
        all_pivots = self._sp_detector.get_all_pivots()
        structure_map = self._ict_mapper.get_structure_map(
            all_pivots, current_price, highs, lows, atr
        )
        result.structure_map = structure_map
        result.equilibrium = structure_map.equilibrium
        result.pivots = all_pivots

        # 3. FVG/OB Scan
        fvg_coll = self._fvg_scanner.scan_fvg(opens, highs, lows, closes, times)
        self._fvg_scanner.detect_ifvg(fvg_coll, opens, closes)
        self._fvg_scanner.check_mitigated(
            fvg_coll, opens, highs, lows, closes, times, bar_close_time
        )
        result.fvg_collection = fvg_coll

        # Classify zone (premium/discount) cho moi FVG
        for fvg in fvg_coll.fvgs:
            zone = self._ict_mapper.classify_fvg_zone(
                fvg.bottom, fvg.top, current_price
            )
            fvg.zone = zone

        # 4. Equal Levels
        sths = self._sp_detector.get_sths()
        stls = self._sp_detector.get_stls()
        self._eq_detector.process_pivots(sths)
        self._eq_detector.process_pivots(stls)
        eq_levels = self._eq_detector.get_equal_levels()
        result.equal_levels = eq_levels

        # 5. Displacement
        d_configs = [
            DisplacementConfig(length=100, factor=1, require_fvg=False),
            DisplacementConfig(length=100, factor=2, require_fvg=False),
            DisplacementConfig(length=100, factor=3, require_fvg=False),
            DisplacementConfig(length=100, factor=1, require_fvg=True),
            DisplacementConfig(length=100, factor=2, require_fvg=True),
        ]
        d_result = self._displacement.compute(opens, highs, lows, closes)
        d_vector = self._displacement.compute_vector(opens, highs, lows, closes)
        result.d_strength = d_vector

        # 6. Feature Vectors
        if self._bsl_ssl:
            f_liq = self._bsl_ssl.get_f_liq_from_detector(
                symbol, current_price, atr, self._sp_detector, 0
            )
        else:
            f_liq = np.zeros(24, dtype=np.float64)

        result.f_liq = f_liq

        fmap = self._liq_indexer.build_symbolic_feature_map(
            pivots=all_pivots,
            structure_map=structure_map,
            fvg_collection=fvg_coll,
            eq_levels=eq_levels,
            current_price=current_price,
            atr=atr,
            equilibrium=structure_map.equilibrium,
            f_liq=f_liq,
            d_strength_mean=float(d_result.d_strength),
            d_strength_max=float(np.max(d_vector)) if len(d_vector) > 0 else 0.0,
        )

        result.f_struct = self._liq_indexer.build_f_struct_vector(fmap)
        result.f_agg = self._liq_indexer.build_f_agg_vector(fmap)

        # Apply session weights
        result.f_struct = self._liq_indexer.apply_session_weights(
            result.f_struct, self.config
        )

        # 7. Zones
        result.zones_created = self._build_zone_list(fvg_coll, all_pivots, current_price)
        result.pivots_claimed = [p for p in all_pivots if p.claimed]

        return result

    def _compute_htf_only(
        self,
        result: PipelineResult,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        times: np.ndarray | None,
    ) -> PipelineResult:
        """Compute HTF-only features khi News Guardrail active.

        Theo agentic_quant_full_plan.md - disable LTF FVG/OB scanning.
        """
        current_price = float(closes[0]) if len(closes) > 0 else 0.0
        result.current_price = current_price

        # Chi detect pivots (IT/LT)
        all_pivots = self._sp_detector.get_all_pivots()
        result.pivots = all_pivots

        # Structure map (chi IT/LT pivots)
        it_lt_pivots = [p for p in all_pivots if p.term.value in ("IT", "LT")]
        structure_map = self._ict_mapper.get_structure_map(
            it_lt_pivots, current_price
        )
        result.structure_map = structure_map
        result.equilibrium = structure_map.equilibrium

        # Khong scan FVG LTF
        result.fvg_collection = None
        result.equal_levels = []
        result.d_strength = np.zeros(5, dtype=np.float64)

        # Feature vectors rỗng
        result.f_struct = np.zeros(64, dtype=np.float64)
        result.f_agg = np.zeros(16, dtype=np.float64)
        result.f_liq = np.zeros(24, dtype=np.float64)

        result.zones_created = []
        return result

    def _build_zone_list(
        self,
        fvg_coll: FVGCollection,
        pivots: list,
        current_price: float,
    ) -> list[dict]:
        """Build zone list cho persistence vao Redis."""
        zones = []

        for fvg in fvg_coll.fvgs:
            if fvg.mitigated:
                continue
            zone_type = "FVG_BULL" if fvg.is_bullish else "FVG_BEAR"
            zone_id = fvg.id or f"fvg_{zone_type.lower()}_{fvg.open_time}"
            zones.append({
                "zone_id": zone_id,
                "zone_type": zone_type,
                "top": fvg.top,
                "bottom": fvg.bottom,
                "ce": fvg.middle,
                "p_hold": 0.5,
                "w_zone": 1.5 if fvg.zone == "premium" else 1.0,
                "iii_formation": fvg.strength,
                "touch_count": 0,
                "status": "UNMITIGATED",
                "zone": fvg.zone,
                "source": "pipeline",
            })

        for fvg in fvg_coll.ifvgs:
            if fvg.mitigated:
                continue
            zone_type = "VI_BULL" if fvg.is_bullish else "VI_BEAR"
            zone_id = fvg.id or f"ifvg_{zone_type.lower()}_{fvg.open_time}"
            zones.append({
                "zone_id": zone_id,
                "zone_type": zone_type,
                "top": fvg.top,
                "bottom": fvg.bottom,
                "ce": fvg.middle,
                "p_hold": 0.3,
                "w_zone": 1.0,
                "iii_formation": 0.0,
                "touch_count": 0,
                "status": "UNMITIGATED",
                "zone": "mid",
                "source": "pipeline",
            })

        return zones

    # =========================================================================
    # Accessors
    # =========================================================================
    @property
    def swing_detector(self) -> SwingPointDetector:
        return self._sp_detector

    @property
    def fvg_scanner(self) -> FVGOBScanner:
        return self._fvg_scanner

    @property
    def eq_detector(self) -> EqualLevelsDetector:
        return self._eq_detector

    @property
    def displacement(self) -> DisplacementEngine:
        return self._displacement

    @property
    def ict_mapper(self) -> ICTStructureMapper:
        return self._ict_mapper

    @property
    def liq_indexer(self) -> LiquidityPoolIndexer:
        return self._liq_indexer
