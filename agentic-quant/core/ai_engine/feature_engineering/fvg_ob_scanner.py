"""FVG / OB Scanner - Fair Value Gap & Order Block Detection.

Port tu Pine Script: FindImbalance, CheckMitigated methods.
Source: pinescript/InstitutionalOrderFlow.pine (lines 918-1111)

Thuat toan FVG:
  Bullish FVG:  low[0] > high[2]
  Bearish FVG: high[0] < low[2]
  Displacement filter: body[1] > std * factor
  BO QUA: Gap (low > high[1] OR high < low[1]) AND NOT Gap[1]
  iFVG: bullish FVG bi invalidate khi close < open (bearish candle pha vỡ)

6 Mitigation Types:
  WICK_TOUCHED, WICK_FILLED, BODY_FILLED, WICK_FILLED_HALF, BODY_FILLED_HALF
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from core.ai_engine.feature_engineering.types import (
    FVGCollection,
    Imbalance,
    ImbalanceType,
    MitigationType,
    FeatureEngineeringConfig,
)

if TYPE_CHECKING:
    from core.ai_engine.feature_engineering.types import Pivot
    from core.ai_engine.feature_engineering.types import StructureEvent


@dataclass
class FVGConfig:
    """Config cho FVG/OB scanner (tu Pine Script Imbalance_Settings)."""
    max_count: int = 20
    max_mitigated_count: int = 5
    merge_vi: bool = False
    fvg_type: str = "Always Display"
    mitigated_type: MitigationType = MitigationType.WICK_FILLED
    displacement_factor: int = 2
    displacement_length: int = 100

    @classmethod
    def from_fe_config(cls, cfg: FeatureEngineeringConfig) -> FVGConfig:
        """Factory tu FeatureEngineeringConfig."""
        return cls(
            merge_vi=cfg.fvg_merge_vi,
            displacement_factor=cfg.fvg_displacement_level,
            mitigated_type=cfg.mitigated_type,
        )


class FVGStateMachine:
    """FVG State Machine - 5 states chuyen doi mot chieu.

    Theo agentic_quant_full_plan.md (TODO 4.4):
      UNMITIGATED -> WICK_TOUCHED -> WICK_FILLED -> WICK_FILLED_HALF -> BODY_FILLED

    moi state chi chuyen toi mot state tiep theo (khong nhay lui).
    """

    # Thuc tu 6 types, nhung state machine chi track progression
    _STATE_ORDER = [
        MitigationType.NONE,
        MitigationType.WICK_TOUCHED,
        MitigationType.WICK_FILLED,
        MitigationType.BODY_FILLED,
        MitigationType.WICK_FILLED_HALF,
        MitigationType.BODY_FILLED_HALF,
    ]

    def transition(
        self,
        imb: Imbalance,
        current_bar_open: float,
        current_bar_close: float,
        current_bar_high: float,
        current_bar_low: float,
        settings_type: MitigationType,
    ) -> MitigationType:
        """Kiem tra chuyen doi state dua tren current bar.

        Kiem tra tu nhe den nang: WICK_TOUCHED -> ... -> BODY_FILLED_HALF.
        Chi tra ve state tiep theo trong progression, khong nhay state.

        Args:
            imb: Imbalance hien tai
            current_bar_*: OHLC cua bar hien tai
            settings_type: MitigationType hien tai (WICK_FILLED, BODY_FILLED, ...)

        Returns:
            MitigationType moi (hoac cu neu chua chuyen doi)
        """
        if imb.mitigated:
            return imb.mitigated_type

        bottom = float(imb.bottom)
        top = float(imb.top)
        mid = float(imb.middle)
        body_min = min(current_bar_open, current_bar_close)
        body_max = max(current_bar_open, current_bar_close)

        # Current state index — find where we are in progression
        try:
            current_idx = self._STATE_ORDER.index(settings_type)
        except ValueError:
            current_idx = self._STATE_ORDER.index(MitigationType.WICK_FILLED)

        if imb.is_bullish:
            # Bullish FVG: gia di LEN de fill gap (low pha vỡ bottom từ trên xuống)
            for next_idx in range(current_idx + 1, len(self._STATE_ORDER)):
                next_type = self._STATE_ORDER[next_idx]
                hit = self._check_bullish_hit(next_type, current_bar_low, body_min, bottom, mid)
                if hit:
                    return next_type
        else:
            # Bearish FVG: gia di XUONG de fill gap (high pha vỡ top từ dưới lên)
            for next_idx in range(current_idx + 1, len(self._STATE_ORDER)):
                next_type = self._STATE_ORDER[next_idx]
                hit = self._check_bearish_hit(next_type, current_bar_high, body_max, top, mid)
                if hit:
                    return next_type

        return settings_type  # No progression

    @staticmethod
    def _check_bullish_hit(
        mit_type: MitigationType, low: float, body_min: float,
        bottom: float, mid: float
    ) -> bool:
        """Check bullish FVG mitigation hit."""
        match mit_type:
            case MitigationType.WICK_TOUCHED:
                return low < bottom
            case MitigationType.WICK_FILLED:
                return low <= bottom
            case MitigationType.BODY_FILLED:
                return body_min <= bottom
            case MitigationType.WICK_FILLED_HALF:
                return low <= mid
            case MitigationType.BODY_FILLED_HALF:
                return body_min <= mid
        return False

    @staticmethod
    def _check_bearish_hit(
        mit_type: MitigationType, high: float, body_max: float,
        top: float, mid: float
    ) -> bool:
        """Check bearish FVG mitigation hit."""
        match mit_type:
            case MitigationType.WICK_TOUCHED:
                return high > top
            case MitigationType.WICK_FILLED:
                return high >= top
            case MitigationType.BODY_FILLED:
                return body_max >= top
            case MitigationType.WICK_FILLED_HALF:
                return high >= mid
            case MitigationType.BODY_FILLED_HALF:
                return body_max >= mid
        return False


class FVGOBScanner:
    """ICT FVG / Order Block Scanner.

    Port tu Pine Script: ImbalanceStructure methods (FindImbalance, CheckMitigated).

    Usage:
      scanner = FVGOBScanner(config)
      collection = scanner.scan_fvg(opens, highs, lows, closes, times)
      scanner.check_mitigated(collection, current_price, opens, highs, lows, closes)
    """

    def __init__(self, config: FVGConfig | None = None) -> None:
        self.config = config or FVGConfig()
        self._collection: FVGCollection = FVGCollection()
        self._state_machine = FVGStateMachine()
        # News Guardrail: disable LTF FVG/OB scanning when news is active
        self._news_guardrail_active: bool = False
        self._atr14: float = 0.0

    def reset(self) -> None:
        self._collection = FVGCollection()

    def on_news_alert(self, active_guardrail: bool) -> None:
        """Hook vao NEWS_ALERT event (News Guardrail).

        Theo agentic_quant_full_plan.md - khi co tinieu re su kien,
        disable LTF FVG/OB scanning.
        """
        self._news_guardrail_active = active_guardrail

    @property
    def news_guardrail_active(self) -> bool:
        return self._news_guardrail_active

    def set_atr(self, atr: float) -> None:
        """Set ATR 14 cho OB strength calculation."""
        self._atr14 = atr

    # =========================================================================
    # Internal helpers
    # =========================================================================
    @staticmethod
    def _is_bullish_fvg(highs: np.ndarray, lows: np.ndarray, idx: int) -> bool:
        """Bullish FVG: low[idx] > high[idx-2]."""
        n = len(highs)
        if idx < 2 or idx >= n:
            return False
        return bool(lows[idx] > highs[idx - 2])

    @staticmethod
    def _is_bearish_fvg(highs: np.ndarray, lows: np.ndarray, idx: int) -> bool:
        """Bearish FVG: high[idx] < low[idx-2]."""
        n = len(lows)
        if idx < 2 or idx >= n:
            return False
        return bool(highs[idx] < lows[idx - 2])

    @staticmethod
    def _is_gap(highs: np.ndarray, lows: np.ndarray, idx: int) -> bool:
        """Gap: low[idx] > high[idx-1] OR high[idx] < low[idx-1]."""
        n = len(highs)
        if idx < 1 or idx >= n:
            return False
        return bool(lows[idx] > highs[idx - 1] or highs[idx] < lows[idx - 1])

    @staticmethod
    def _body_size(opens: np.ndarray, closes: np.ndarray, idx: int) -> float:
        if idx >= len(opens) or idx >= len(closes):
            return 0.0
        return abs(opens[idx] - closes[idx])

    @staticmethod
    def _is_bullish_candle(opens: np.ndarray, closes: np.ndarray, idx: int) -> bool:
        if idx >= len(opens) or idx >= len(closes):
            return False
        return bool(closes[idx] > opens[idx])

    # =========================================================================
    # FVG Detection (port tu Pine Script FindImbalance)
    # =========================================================================
    def scan_fvg(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        times: np.ndarray | None = None,
    ) -> FVGCollection:
        """Phat hien FVG Bull/Bear trong day OHLC.

        Pine Script FindImbalance (IS.type == 'FVG'):
          Gap = low > high[1] or high < low[1]
          FVG = high < low[2] or low > high[2]
          if FVG and not Gap and not Gap[1]:
            # Bullish: low > high[2]
            #   close = mergeVI AND min(O,C) > max(O1,C1) ? min(O,C) : low
            #   open  = mergeVI AND min(O1,C1) > max(O2,C2) ? max(O2,C2) : high[2]
            # Bearish: high < low[2]
            #   close = mergeVI AND max(O,C) < min(O1,C1) ? max(O,C) : high
            #   open  = mergeVI AND min(O2,C2) > max(O1,C1) ? min(O2,C2) : low[2]

        Args:
            opens:  mang open prices, index 0 = current bar
            highs:  mang high prices
            lows:   mang low prices
            closes: mang close prices
            times:  mang timestamps (ms)

        Returns:
            FVGCollection chua fvgs va ifvgs
        """
        n = len(opens)
        if n < 3:
            return self._collection

        fvgs: list[Imbalance] = []
        i_fvgs: list[Imbalance] = []

        for i in range(2, n):
            # Kiem tra Gap - FVG phai khong co Gap[0] va khong co Gap[1]
            has_gap = self._is_gap(highs, lows, i)
            has_gap_prev = self._is_gap(highs, lows, i - 1) if i - 1 > 0 else False
            if has_gap or has_gap_prev:
                continue

            bullish_fvg = self._is_bullish_fvg(highs, lows, i)
            bearish_fvg = self._is_bearish_fvg(highs, lows, i)

            if not bullish_fvg and not bearish_fvg:
                continue

            # Displacement filter (chi cho "Level 1-4" hoac "Same As Displacement")
            if self.config.fvg_type != "Always Display":
                displacement_ok = self._check_displacement(
                    opens, closes, highs, lows, i - 1, self.config.displacement_factor
                )
                if not displacement_ok:
                    continue

            imb = self._create_fvg(
                opens, highs, lows, closes, times, i, bullish_fvg
            )
            if imb:
                fvgs.append(imb)

        self._collection.fvgs = fvgs
        self._collection.ifvgs = i_fvgs
        return self._collection

    def _check_displacement(
        self,
        opens: np.ndarray,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        idx: int,
        factor: int,
    ) -> bool:
        """Kiem tra displacement filter: body > std * factor."""
        n = len(opens)
        length = self.config.displacement_length
        if n < length or idx < 0 or idx >= n:
            return True  # Pass through if not enough data

        body = np.abs(opens - closes)
        recent = body[max(0, n - length):n]
        std_val = float(np.std(recent)) if len(recent) > 1 else 0.0

        body_val = abs(opens[idx] - closes[idx])
        return bool(body_val > std_val * factor)

    def _create_fvg(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        times: np.ndarray | None,
        idx: int,
        bullish: bool,
    ) -> Imbalance | None:
        """Tao Imbalance tu OHLC data (port tu Pine Script lines 1064-1091)."""
        n = len(opens)
        if idx < 2 or idx >= n:
            return None

        if bullish:
            # Bullish FVG: low[idx] > high[idx-2]
            # open  = gap zone bottom = high[idx-2]
            # close = gap zone top  = low[idx]
            gap_bottom = highs[idx - 2]
            gap_top = lows[idx]

            if self.config.merge_vi:
                # Neu merge VI, mo rong FVG
                if (idx - 1 >= 0 and
                    min(opens[idx - 1], closes[idx - 1]) > max(opens[idx - 2], closes[idx - 2])):
                    gap_top = min(opens[idx - 1], closes[idx - 1])
                if (idx - 2 >= 0 and
                    min(opens[idx - 2], closes[idx - 2]) > max(opens[idx - 3], closes[idx - 3])):
                    gap_bottom = max(opens[idx - 2], closes[idx - 2])

            imb = Imbalance(
                imb_type=ImbalanceType.FVG,
                open_time=int(times[idx - 1]) if times is not None else 0,
                close_time=int(times[idx]) if times is not None else 0,
                top=gap_top,
                bottom=gap_bottom,
                # is_bullish derived from top/bottom
                displacement_factor=float(self.config.displacement_factor),
                body_size=self._body_size(opens, closes, idx - 1),
            )
            imb.__post_init__()
            return imb
        else:
            # Bearish FVG: high[idx] < low[idx-2]
            gap_top = lows[idx - 2]
            gap_bottom = highs[idx]

            if self.config.merge_vi:
                if (idx - 1 >= 0 and
                    max(opens[idx - 1], closes[idx - 1]) < min(opens[idx - 2], closes[idx - 2])):
                    gap_bottom = max(opens[idx - 1], closes[idx - 1])
                if (idx - 2 >= 0 and
                    min(opens[idx - 2], closes[idx - 2]) > max(opens[idx - 3], closes[idx - 3])):
                    gap_top = min(opens[idx - 2], closes[idx - 2])

            imb = Imbalance(
                imb_type=ImbalanceType.FVG,
                open_time=int(times[idx - 1]) if times is not None else 0,
                close_time=int(times[idx]) if times is not None else 0,
                top=gap_top,
                bottom=gap_bottom,
                # is_bullish derived from top/bottom
                displacement_factor=float(self.config.displacement_factor),
                body_size=self._body_size(opens, closes, idx - 1),
            )
            imb.__post_init__()
            return imb

    # =========================================================================
    # FVG Invalidation (iFVG) - port tu Pine Script CheckMitigated
    # =========================================================================
    def detect_ifvg(
        self,
        collection: FVGCollection,
        opens: np.ndarray,
        closes: np.ndarray,
    ) -> FVGCollection:
        """Phat hien FVG bi invalidate (chuyen thanh iFVG).

        Theo agentic_quant_full_plan.md (TODO 4.4 - iFVG Inversion Logic):
          Bullish FVG -> iFVG: bearish candle CLOSES BELOW fvg.bottom
            = (close < open) AND (close < fvg.bottom)
          Bearish FVG -> iFVG: bullish candle CLOSES ABOVE fvg.top
            = (close > open) AND (close > fvg.top)

        Dung hơn Pine Script vi Pine Script chi kiem tra close < open,
        khong kiem tra gia tri close so voi fvg boundary.
        """
        if not collection.fvgs:
            return collection

        curr_open = float(opens[0]) if len(opens) > 0 else 0.0
        curr_close = float(closes[0]) if len(closes) > 0 else 0.0

        # Check invertability: bullish != previous.bullish
        for i, fvg in enumerate(collection.fvgs):
            if i + 1 < len(collection.fvgs):
                prev = collection.fvgs[i + 1]
                fvg.invertable = fvg.is_bullish != prev.is_bullish

        # Check inversion (iFVG)
        for fvg in collection.fvgs:
            if fvg.invertable and not fvg.inverted:
                # Bullish FVG bi invalidate: bearish candle (close < open) AND close < fvg.bottom
                if fvg.is_bullish and curr_close < curr_open and curr_close < fvg.bottom:
                    fvg.inverted = True
                    ifvg = Imbalance(
                        imb_type=ImbalanceType.IFVG,
                        open_time=fvg.close_time,
                        close_time=0,
                        top=fvg.bottom,
                        bottom=fvg.top,
                        strength=fvg.strength,
                        displacement_factor=fvg.displacement_factor,
                    )
                    ifvg.__post_init__()
                    collection.ifvgs.append(ifvg)
                # Bearish FVG bi invalidate: bullish candle (close > open) AND close > fvg.top
                elif not fvg.is_bullish and curr_close > curr_open and curr_close > fvg.top:
                    fvg.inverted = True
                    ifvg = Imbalance(
                        imb_type=ImbalanceType.IFVG,
                        open_time=fvg.close_time,
                        close_time=0,
                        top=fvg.top,
                        bottom=fvg.bottom,
                        strength=fvg.strength,
                        displacement_factor=fvg.displacement_factor,
                    )
                    ifvg.__post_init__()
                    collection.ifvgs.append(ifvg)

        return collection

    # =========================================================================
    # Mitigation Check (port tu Pine Script CheckMitigated, lines 1027-1051)
    # =========================================================================
    def check_mitigated(
        self,
        collection: FVGCollection,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        times: np.ndarray | None = None,
        current_time: int | None = None,
    ) -> list[Imbalance]:
        """Kiem tra FVG/VI/GAP co bi mitigated chua.

        Pine Script CheckMitigated:
          switch mitigated_type:
            'Wick Touched':    imb.mitigated := imb.isbullish ? low < imb.close : high > imb.close
            'Wick filled':       imb.mitigated := imb.isbullish ? low <= imb.open : high >= imb.open
            'Body filled':       imb.mitigated := imb.isbullish ? min(open,close) <= imb.open : max(open,close) >= imb.open
            'Wick filled half':  imb.mitigated := imb.isbullish ? low <= imb.middle : high >= imb.middle
            'Body filled half':  imb.mitigated := imb.isbullish ? min(open,close) <= imb.middle : max(open,close) >= imb.middle

        Args:
            collection: FVGCollection hien tai
            opens/highs/lows/closes: OHLC data
            current_time: timestamp hien tai

        Returns:
            List cac FVG bi mitigated
        """
        if current_time is None:
            current_time = int(times[0]) if times is not None else 0

        mitigated_list: list[Imbalance] = []

        all_imbs = collection.fvgs + collection.ifvgs
        for imb in all_imbs:
            if imb.mitigated:
                continue

            mit_type = self._get_mitigation_type(imb, opens, highs, lows, closes)
            if mit_type != MitigationType.NONE:
                imb.mitigated = True
                imb.mitigated_type = mit_type
                imb.mitigated_time = current_time
                mitigated_list.append(imb)

        return mitigated_list

    def _get_mitigation_type(
        self,
        imb: Imbalance,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> MitigationType:
        """Tinh mitigation type cho mot imbalance.

        Pine Script CheckMitigated:
          Bullish FVG (top < bottom): wick/body goes UP to fill the gap
            - Wick Touched:   low < bottom   (wick touches close line)
            - Wick Filled:     low <= bottom   (wick goes through close line)
            - Body Filled:    min(O,C) <= bottom (body closes through close line)
            - Wick Filled Half: low <= middle   (wick through midpoint)
            - Body Filled Half: min(O,C) <= middle

          Bearish FVG (top > bottom): wick/body goes DOWN to fill the gap
            - Wick Touched:   high > top
            - Wick Filled:   high >= top
            - Body Filled:    max(O,C) >= top
            - Wick Filled Half: high >= middle
            - Body Filled Half: max(O,C) >= middle
        """
        if imb.mitigated:
            return imb.mitigated_type

        mit = self.config.mitigated_type
        if mit == MitigationType.NONE:
            return MitigationType.NONE

        # Current candle values
        curr_open = float(opens[0]) if len(opens) > 0 else 0.0
        curr_close = float(closes[0]) if len(closes) > 0 else 0.0
        curr_high = float(highs[0]) if len(highs) > 0 else 0.0
        curr_low = float(lows[0]) if len(lows) > 0 else 0.0
        curr_body_min = min(curr_open, curr_close)
        curr_body_max = max(curr_open, curr_close)

        bottom = float(imb.bottom)
        top = float(imb.top)
        mid = float(imb.middle)

        if imb.is_bullish:
            # Bullish FVG: price goes UP to fill gap
            # bottom = gap bottom, mid = gap midpoint
            match mit:
                case MitigationType.WICK_TOUCHED:
                    return MitigationType.WICK_TOUCHED if curr_low < bottom else MitigationType.NONE
                case MitigationType.WICK_FILLED:
                    return MitigationType.WICK_FILLED if curr_low <= bottom else MitigationType.NONE
                case MitigationType.BODY_FILLED:
                    return MitigationType.BODY_FILLED if curr_body_min <= bottom else MitigationType.NONE
                case MitigationType.WICK_FILLED_HALF:
                    return MitigationType.WICK_FILLED_HALF if curr_low <= mid else MitigationType.NONE
                case MitigationType.BODY_FILLED_HALF:
                    return MitigationType.BODY_FILLED_HALF if curr_body_min <= mid else MitigationType.NONE
        else:
            # Bearish FVG: price goes DOWN to fill gap
            # top = gap top, mid = gap midpoint
            match mit:
                case MitigationType.WICK_TOUCHED:
                    return MitigationType.WICK_TOUCHED if curr_high > top else MitigationType.NONE
                case MitigationType.WICK_FILLED:
                    return MitigationType.WICK_FILLED if curr_high >= top else MitigationType.NONE
                case MitigationType.BODY_FILLED:
                    return MitigationType.BODY_FILLED if curr_body_max >= top else MitigationType.NONE
                case MitigationType.WICK_FILLED_HALF:
                    return MitigationType.WICK_FILLED_HALF if curr_high >= mid else MitigationType.NONE
                case MitigationType.BODY_FILLED_HALF:
                    return MitigationType.BODY_FILLED_HALF if curr_body_max >= mid else MitigationType.NONE

        return MitigationType.NONE

    def get_mitigation_type(
        self,
        imb: Imbalance,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> MitigationType:
        """Tinh mitigation type (read-only)."""
        return self._get_mitigation_type(imb, opens, highs, lows, closes)

    # =========================================================================
    # Order Block Detection
    # =========================================================================
    def scan_ob(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        bos_trigger_idx: int | None = None,
        bos_trigger_price: float = 0.0,
        atr: float = 0.0,
    ) -> list[Imbalance]:
        """Phat hien Order Block truoc BOS/MSS trigger.

        Theo agentic_quant_full_plan.md (TODO 4.4):
          - Tim last bearish candle truoc trigger -> Bullish OB
          - Tim last bullish candle truoc trigger -> Bearish OB
          - strength = |high[bos] - low[bos]| / ATR_14

        Args:
            opens/highs/lows/closes: OHLC data
            bos_trigger_idx: chi so bar trigger BOS/MSS (optional)
            bos_trigger_price: gia trigger (optional)
            atr: gia tri ATR hien tai (cho strength)
        """
        obs: list[Imbalance] = []
        n = len(opens)

        if bos_trigger_idx is None:
            bos_trigger_idx = 0

        if bos_trigger_idx >= n or bos_trigger_idx < 1:
            return obs

        # Neu chua co atr, su dung atr14 tu instance
        atr_val = atr if atr > 0 else self._atr14

        # Bullish OB: tim last bearish candle truoc bos_trigger_idx
        for i in range(bos_trigger_idx - 1, max(0, bos_trigger_idx - 20), -1):
            if i >= len(opens):
                continue
            if closes[i] < opens[i]:
                strength = (highs[bos_trigger_idx] - lows[bos_trigger_idx]) / atr_val if atr_val > 0 else 0.0
                ob = Imbalance(
                    imb_type=ImbalanceType.FVG,
                    open_time=0,
                    close_time=0,
                    top=highs[i],
                    bottom=lows[i],
                    strength=float(strength),
                )
                ob.__post_init__()
                obs.append(ob)
                break

        # Bearish OB: tim last bullish candle truoc bos_trigger_idx
        for i in range(bos_trigger_idx - 1, max(0, bos_trigger_idx - 20), -1):
            if i >= len(opens):
                continue
            if closes[i] > opens[i]:
                strength = (highs[bos_trigger_idx] - lows[bos_trigger_idx]) / atr_val if atr_val > 0 else 0.0
                ob = Imbalance(
                    imb_type=ImbalanceType.FVG,
                    open_time=0,
                    close_time=0,
                    top=highs[i],
                    bottom=lows[i],
                    strength=float(strength),
                )
                ob.__post_init__()
                obs.append(ob)
                break

        return obs

    def scan_ob_from_events(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        structure_events: list,
        atr: float = 0.0,
    ) -> list[Imbalance]:
        """Phat hien OB tu danh sach StructureEvents.

        Theo agentic_quant_full_plan.md - dung structure_events lam trigger.

        Args:
            opens/highs/lows/closes: OHLC data
            structure_events: list[StructureEvent] tu ICTStructureMapper
            atr: gia tri ATR hien tai

        Returns:
            list[Imbalance] chua Order Blocks
        """
        all_obs: list[Imbalance] = []
        for event in structure_events:
            trigger_idx = getattr(event, 'trigger_index', 0)
            obs = self.scan_ob(opens, highs, lows, closes, trigger_idx, atr=atr)
            all_obs.extend(obs)
        return all_obs

    # =========================================================================
    # Zone Classification
    # =========================================================================
    def classify_zone(
        self,
        imb: Imbalance,
        current_price: float,
        equilibrium: float = 0.0,
    ) -> str:
        """Classify FVG/OB zone = premium/discount.

        Premium: gia > EQ
        Discount: gia < EQ
        """
        fvg_mid = (imb.top + imb.bottom) / 2.0
        ref_price = equilibrium if equilibrium != 0.0 else fvg_mid

        if fvg_mid > ref_price:
            return "premium"
        elif fvg_mid < ref_price:
            return "discount"
        return "mid"

    # =========================================================================
    # Public API
    # =========================================================================
    def get_fvgs(self) -> list[Imbalance]:
        return list(self._collection.fvgs)

    def get_ifvgs(self) -> list[Imbalance]:
        return list(self._collection.ifvgs)

    def get_active_fvgs(self) -> list[Imbalance]:
        """FVG chua bi mitigated va chua bi invalidate."""
        return [f for f in self._collection.fvgs if not f.mitigated and not f.inverted]

    @property
    def fvg_count(self) -> int:
        return len(self._collection.fvgs)

    @property
    def ifvg_count(self) -> int:
        return len(self._collection.ifvgs)
