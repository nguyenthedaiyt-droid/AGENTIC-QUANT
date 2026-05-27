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

    # ------------------------------------------------------------------
    # Static feature builders
    # ------------------------------------------------------------------

    @staticmethod
    def build_f_flow(usv_data: dict | None = None) -> np.ndarray:
        """Xây dung vector F_flow[20] tu USV (Unified Sigma Vector) data.

        Dims:
          [0]:  CVD_norm = CVD_t / V_total_t       — chuan hoa cumulative volume delta
          [1]:  III_t     = (CVD_t / V_avg_30) * (|dP_t| / ATR_14)  — Intraday Intensity Index
          [2]:  DIV_CVD   = sign(dP_t) * sign(CVD_t) € {-1, +1}     — CVD divergence sign
          [3]:  CVD_slope_5bar   — slope cua CVD qua 5 nen
          [4]:  CVD_slope_10bar  — slope cua CVD qua 10 nen
          [5]:  CVD_slope_20bar  — slope cua CVD qua 20 nen
          [6]:  CVD_std_10       — std cua CVD 10 nen
          [7]:  CVD_zscore       — z-score cua CVD hien tai
          [8]:  III_slope_5bar   — slope cua III qua 5 nen
          [9]:  III_slope_10bar  — slope cua III qua 10 nen
          [10]: III_mean_20      — trung binh III 20 nen
          [11]: III_std_20       — std III 20 nen
          [12]: III_zscore       — z-score cua III hien tai
          [13]: OBI = (V_bid_best - V_ask_best) / (V_bid_best + V_ask_best + 1e-9)  — Order Book Imbalance
          [14]: OBI_slope_5bar   — slope cua OBI qua 5 nen
          [15]: OBI_mean_20      — trung binh OBI 20 nen
          [16]: OBI_std_20       — std OBI 20 nen
          [17]: spread_pips      — spread hien tai (pips)
          [18]: spread_zscore    — z-score cua spread
          [19]: reserved         — du phong

        Args:
            usv_data: Dict chua key 'CVD_t', 'V_total_t', 'V_avg_30', 'dP_t', 'ATR_14',
                      'CVD_slope_5', 'CVD_slope_10', 'CVD_slope_20', 'CVD_std_10',
                      'CVD_zscore', 'III_slope_5', 'III_slope_10', 'III_mean_20',
                      'III_std_20', 'III_zscore', 'V_bid_best', 'V_ask_best',
                      'OBI_slope_5', 'OBI_mean_20', 'OBI_std_20', 'spread_pips',
                      'spread_zscore', hoac None.

        Returns:
            np.ndarray shape (20,) — neu input None, tra ve zeros.
        """
        out = np.zeros(D_FLOW, dtype=np.float64)
        if usv_data is None:
            return out

        try:
            # [0] CVD_norm
            cvd_t = usv_data.get("CVD_t", 0.0)
            v_total_t = usv_data.get("V_total_t", 1.0)
            out[0] = cvd_t / v_total_t if v_total_t != 0.0 else 0.0

            # [1] III_t
            v_avg_30 = usv_data.get("V_avg_30", 1.0)
            dP_t = usv_data.get("dP_t", 0.0)
            atr_14 = usv_data.get("ATR_14", 1.0)
            if v_avg_30 != 0.0 and atr_14 != 0.0:
                out[1] = (cvd_t / v_avg_30) * (abs(dP_t) / atr_14)

            # [2] DIV_CVD
            out[2] = float(np.sign(dP_t) * np.sign(cvd_t))

            # [3-7] CVD slopes, std, zscore
            out[3] = usv_data.get("CVD_slope_5", 0.0)
            out[4] = usv_data.get("CVD_slope_10", 0.0)
            out[5] = usv_data.get("CVD_slope_20", 0.0)
            out[6] = usv_data.get("CVD_std_10", 0.0)
            out[7] = usv_data.get("CVD_zscore", 0.0)

            # [8-12] III slopes, mean, std, zscore
            out[8] = usv_data.get("III_slope_5", 0.0)
            out[9] = usv_data.get("III_slope_10", 0.0)
            out[10] = usv_data.get("III_mean_20", 0.0)
            out[11] = usv_data.get("III_std_20", 0.0)
            out[12] = usv_data.get("III_zscore", 0.0)

            # [13] OBI
            v_bid_best = usv_data.get("V_bid_best", 0.0)
            v_ask_best = usv_data.get("V_ask_best", 0.0)
            denom = v_bid_best + v_ask_best + 1e-9
            out[13] = (v_bid_best - v_ask_best) / denom

            # [14-18] OBI slopes, mean, std, spread
            out[14] = usv_data.get("OBI_slope_5", 0.0)
            out[15] = usv_data.get("OBI_mean_20", 0.0)
            out[16] = usv_data.get("OBI_std_20", 0.0)
            out[17] = usv_data.get("spread_pips", 0.0)
            out[18] = usv_data.get("spread_zscore", 0.0)

            # [19] reserved — giu nguyen 0.0
        except Exception:
            # Neu co loi, tra ve zeros (da khoi tao o tren)
            pass

        return out

    @staticmethod
    def build_f_macro(macro_context: dict | None = None) -> np.ndarray:
        """Xay dung vector F_macro[12] tu macro context (lich kinh te, tin toc).

        Dims:
          [0]: I_news — news impact score:
                 Low=0.2, Medium=0.5, High=1.0*(1+0.4*M_bar/ATR_D1)
          [1]: Surprise factor S = (actual - forecast) / sigma_surprise
          [2]: seconds_to_next_event — chuan hoa [0,1]: min(seconds/3600, 1.0)
          [3]: active_guardrail flag — 0.0 hoac 1.0
          [4]: session_onehot[0]    — Asian = [1,0]
          [5]: session_onehot[1]    — London = [0,1]; NY = [0,0]
          [6]: regime_onehot[0]     — NORMAL = [0,0]
          [7]: regime_onehot[1]     — PRE_NEWS = [1,0]; NEWS_WINDOW = [0,1]; POST_NEWS = [0,0]
          [8]: cluster_flag         — 1.0 neu 2+ High events trong 30 phut
          [9]: event_impact_code    — Low=0.2, Medium=0.5, High=1.0
          [10]: reserved
          [11]: reserved

        Args:
            macro_context: Dict chua cac key:
                'news_impact_level' (str: Low/Medium/High),
                'M_bar', 'ATR_D1',
                'actual', 'forecast', 'sigma_surprise',
                'seconds_to_next_event',
                'active_guardrail',
                'session' (str: Asian/London/NY),
                'regime' (str: NORMAL/PRE_NEWS/NEWS_WINDOW/POST_NEWS),
                'cluster_flag',
                'event_impact_code' (str: Low/Medium/High),
                hoac None.

        Returns:
            np.ndarray shape (12,) — neu input None, tra ve zeros.
        """
        out = np.zeros(D_MACRO, dtype=np.float64)
        if macro_context is None:
            return out

        try:
            # [0] I_news — news impact score
            impact_level = macro_context.get("news_impact_level", "Low")
            impact_map = {"low": 0.2, "medium": 0.5, "high": 1.0}
            base = impact_map.get(impact_level.lower(), 0.2)
            if impact_level.lower() == "high":
                m_bar = macro_context.get("M_bar", 0.0)
                atr_d1 = macro_context.get("ATR_D1", 1.0)
                out[0] = base * (1.0 + 0.4 * m_bar / atr_d1) if atr_d1 != 0.0 else base
            else:
                out[0] = base

            # [1] Surprise factor S
            actual = macro_context.get("actual", 0.0)
            forecast = macro_context.get("forecast", 0.0)
            sigma = macro_context.get("sigma_surprise", 1.0)
            out[1] = (actual - forecast) / sigma if sigma != 0.0 else 0.0

            # [2] seconds_to_next_event — chuan hoa [0, 1]
            secs = macro_context.get("seconds_to_next_event", 3600.0)
            out[2] = min(secs / 3600.0, 1.0)

            # [3] active_guardrail
            out[3] = 1.0 if macro_context.get("active_guardrail", False) else 0.0

            # [4-5] session_onehot: Asian=[1,0], London=[0,1], NY=[0,0]
            session = macro_context.get("session", "NY")
            session_lower = session.lower()
            if session_lower == "asian":
                out[4] = 1.0
                out[5] = 0.0
            elif session_lower == "london":
                out[4] = 0.0
                out[5] = 1.0
            else:  # NY
                out[4] = 0.0
                out[5] = 0.0

            # [6-7] regime_onehot: NORMAL=[0,0], PRE_NEWS=[1,0], NEWS_WINDOW=[0,1], POST_NEWS=[0,0]
            regime = macro_context.get("regime", "NORMAL")
            regime_upper = regime.upper()
            if regime_upper == "PRE_NEWS":
                out[6] = 1.0
                out[7] = 0.0
            elif regime_upper == "NEWS_WINDOW":
                out[6] = 0.0
                out[7] = 1.0
            else:  # NORMAL hoac POST_NEWS
                out[6] = 0.0
                out[7] = 0.0

            # [8] cluster_flag
            out[8] = 1.0 if macro_context.get("cluster_flag", False) else 0.0

            # [9] event_impact_code
            impact_code = macro_context.get("event_impact_code", "Low")
            code_map = {"low": 0.2, "medium": 0.5, "high": 1.0}
            out[9] = code_map.get(impact_code.lower(), 0.2)

            # [10-11] reserved — giu nguyen 0.0
        except Exception:
            pass

        return out

    @staticmethod
    def build_f_zone(zone_data: dict | None = None) -> np.ndarray:
        """Xay dung vector F_zone[16] tu zone data (FVG, OB, EQ, iFVG).

        Dims:
          [0]:  zone_size        = (top - bottom) / ATR_14 — chuan hoa zone range
          [1]:  age_bars         = so nen M1 tu khi zone duoc formed
          [2]:  tf_code          = M1=1/6, M5=2/6, M15=3/6, H1=4/6, H4=5/6, D1=6/6
          [3]:  type_code        = FVG=0.0, OB=0.33, EQ=0.67, iFVG=1.0
          [4]:  p_hold_history   = p_hold hien tai (0.0-1.0)
          [5]:  touch_count      = so lan gia cham zone
          [6]:  mitigation_level = NONE=0, WICK_TOUCHED=0.2, WICK_FILLED=0.4,
                                    BODY_FILLED=0.6, WICK_FILLED_HALF=0.8, BODY_FILLED_HALF=1.0
          [7]:  zone_strength    = displacement_factor * body_size
          [8]:  premium_discount = premium=1.0, discount=0.0, mid=0.5
          [9]:  w_zone           = 1.0 hoac 1.5
          [10]: iii_formation    = III value khi zone duoc tao
          [11]: distance_to_price = chuan hoa = |mid - price| / ATR
          [12-15]: reserved

        Args:
            zone_data: Dict chua cac key:
                'zone_top', 'zone_bottom', 'ATR_14',
                'age_bars',
                'tf' (str: M1/M5/M15/H1/H4/D1),
                'type' (str: FVG/OB/EQ/iFVG),
                'p_hold',
                'touch_count',
                'mitigation_level' (str: NONE/WICK_TOUCHED/WICK_FILLED/...),
                'zone_strength' hoac 'displacement_factor', 'body_size',
                'premium_discount' (str: premium/discount/mid),
                'w_zone',
                'iii_formation',
                'zone_mid', 'current_price', 'ATR',
                hoac None.

        Returns:
            np.ndarray shape (16,) — neu input None, tra ve zeros.
        """
        out = np.zeros(D_ZONE, dtype=np.float64)
        if zone_data is None:
            return out

        try:
            # [0] zone_size = (top - bottom) / ATR_14
            zone_top = zone_data.get("zone_top", 0.0)
            zone_bottom = zone_data.get("zone_bottom", 0.0)
            atr_14 = zone_data.get("ATR_14", 1.0)
            out[0] = (zone_top - zone_bottom) / atr_14 if atr_14 != 0.0 else 0.0

            # [1] age_bars
            out[1] = float(zone_data.get("age_bars", 0))

            # [2] tf_code
            tf = zone_data.get("tf", "M1")
            tf_map = {"m1": 1.0 / 6.0, "m5": 2.0 / 6.0, "m15": 3.0 / 6.0,
                      "h1": 4.0 / 6.0, "h4": 5.0 / 6.0, "d1": 6.0 / 6.0}
            out[2] = tf_map.get(tf.lower(), 1.0 / 6.0)

            # [3] type_code
            ztype = zone_data.get("type", "FVG")
            type_map = {"fvg": 0.0, "ob": 0.33, "eq": 0.67, "ifvg": 1.0}
            out[3] = type_map.get(ztype.upper(), 0.0)

            # [4] p_hold_history
            p_hold = zone_data.get("p_hold", 0.0)
            out[4] = np.clip(float(p_hold), 0.0, 1.0)

            # [5] touch_count
            out[5] = float(zone_data.get("touch_count", 0))

            # [6] mitigation_level
            mit_level = zone_data.get("mitigation_level", "NONE")
            mit_map = {
                "none": 0.0,
                "wick_touched": 0.2,
                "wick_filled": 0.4,
                "body_filled": 0.6,
                "wick_filled_half": 0.8,
                "body_filled_half": 1.0,
            }
            out[6] = mit_map.get(mit_level.lower().replace("-", "_"), 0.0)

            # [7] zone_strength = displacement_factor * body_size
            displacement = zone_data.get("displacement_factor", 0.0)
            body_size = zone_data.get("body_size", 0.0)
            out[7] = displacement * body_size
            # Override neu co san zone_strength
            if "zone_strength" in zone_data:
                out[7] = float(zone_data["zone_strength"])

            # [8] premium_discount
            pd = zone_data.get("premium_discount", "mid")
            pd_lower = pd.lower()
            if pd_lower == "premium":
                out[8] = 1.0
            elif pd_lower == "discount":
                out[8] = 0.0
            else:  # mid
                out[8] = 0.5

            # [9] w_zone
            out[9] = float(zone_data.get("w_zone", 1.0))

            # [10] iii_formation
            out[10] = zone_data.get("iii_formation", 0.0)

            # [11] distance_to_price = |mid - price| / ATR
            zone_mid = zone_data.get("zone_mid", 0.0)
            current_price = zone_data.get("current_price", 0.0)
            atr = zone_data.get("ATR", 1.0)
            out[11] = abs(zone_mid - current_price) / atr if atr != 0.0 else 0.0

            # [12-15] reserved — giu nguyen 0.0
        except Exception:
            pass

        return out

    @staticmethod
    def build_f_contact(contact_candle: dict | None = None, zone: dict | None = None) -> np.ndarray:
        """Xay dung vector F_contact[20] tu contact candle va zone data.

        Dims:
          [0]:  CE_at_contact       = (price - middle) / (range/2) € [-1, +1]
          [1]:  body_size_at_contact = |open - close| / ATR_14
          [2]:  CVD_at_contact       — CVD value tai contact bar
          [3]:  CVD_acceleration     = CVD_slope_5bar - CVD_slope_10bar
          [4]:  III_at_contact       — III value tai contact bar
          [5]:  displacement_at_contact = body / (std * 100)
          [6]:  volume_ratio         = V_contact / V_avg_20
          [7]:  spread_at_contact    — spread tai contact
          [8]:  bar_position_in_zone = 0.0 (bottom) -> 1.0 (top)
          [9-19]: reserved

        Args:
            contact_candle: Dict chua cac key:
                'price', 'middle', 'range',
                'open', 'close', 'ATR_14',
                'CVD_value', 'CVD_slope_5', 'CVD_slope_10',
                'III_value', 'body', 'std',
                'V_contact', 'V_avg_20',
                'spread',
                hoac None.
            zone: Dict chua cac key:
                'zone_top', 'zone_bottom',
                hoac None.

        Returns:
            np.ndarray shape (20,) — neu input None, tra ve zeros.
        """
        out = np.zeros(D_CONTACT, dtype=np.float64)
        if contact_candle is None:
            return out

        try:
            # [0] CE_at_contact = (price - middle) / (range/2)
            price = contact_candle.get("price", 0.0)
            middle = contact_candle.get("middle", 0.0)
            rng = contact_candle.get("range", 1.0)
            if rng != 0.0:
                out[0] = (price - middle) / (rng / 2.0)
            else:
                out[0] = 0.0

            # [1] body_size_at_contact = |open - close| / ATR_14
            open_price = contact_candle.get("open", 0.0)
            close = contact_candle.get("close", 0.0)
            atr_14 = contact_candle.get("ATR_14", 1.0)
            out[1] = abs(open_price - close) / atr_14 if atr_14 != 0.0 else 0.0

            # [2] CVD_at_contact
            out[2] = contact_candle.get("CVD_value", 0.0)

            # [3] CVD_acceleration = CVD_slope_5bar - CVD_slope_10bar
            cvd_slope_5 = contact_candle.get("CVD_slope_5", 0.0)
            cvd_slope_10 = contact_candle.get("CVD_slope_10", 0.0)
            out[3] = cvd_slope_5 - cvd_slope_10

            # [4] III_at_contact
            out[4] = contact_candle.get("III_value", 0.0)

            # [5] displacement_at_contact = body / (std * 100)
            body = contact_candle.get("body", 0.0)
            std = contact_candle.get("std", 1.0)
            denom = std * 100.0
            out[5] = body / denom if denom != 0.0 else 0.0

            # [6] volume_ratio = V_contact / V_avg_20
            v_contact = contact_candle.get("V_contact", 0.0)
            v_avg_20 = contact_candle.get("V_avg_20", 1.0)
            out[6] = v_contact / v_avg_20 if v_avg_20 != 0.0 else 0.0

            # [7] spread_at_contact
            out[7] = contact_candle.get("spread", 0.0)

            # [8] bar_position_in_zone: 0.0 (bottom) -> 1.0 (top)
            if zone is not None:
                zone_top = zone.get("zone_top", 0.0)
                zone_bottom = zone.get("zone_bottom", 0.0)
                zrange = zone_top - zone_bottom
                if zrange != 0.0:
                    out[8] = (price - zone_bottom) / zrange
                else:
                    out[8] = 0.0
            else:
                # Fallback: dung middle +/- range tu contact_candle de uoc luong
                mid = contact_candle.get("middle", 0.0)
                r = contact_candle.get("range", 1.0)
                if r != 0.0:
                    zb = mid - r / 2.0
                    zt = mid + r / 2.0
                    out[8] = (price - zb) / (zt - zb)
                else:
                    out[8] = 0.0

            # [9-19] reserved — giu nguyen 0.0
        except Exception:
            pass

        return out

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
