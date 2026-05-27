# =============================================================================
# AGENTIC-QUANT — Khung du lieu cho mot tick tu MT5
# =============================================================================

from __future__ import annotations

import struct
from dataclasses import dataclass


# =============================================================================
# TickFrame: binary format tu MT5 EA
# =============================================================================
# Schema (little-endian):
#   symbol[12]   - 12 bytes, null-terminated string
#   timestamp    - Q (unsigned long long, 8 bytes) microseconds since epoch
#   bid          - d (unsigned char, 1 byte) + scaling
#   ask          - d (unsigned char, 1 byte) + scaling
#   last         - d (unsigned char, 1 byte) + scaling
#   volume       - Q (unsigned long long, 8 bytes) in units (e.g. 100 oz)
#   flags        - I (unsigned int, 4 bytes) bitfield
#
# Bitfield flags:
#   bit 0: buy_volume_increased
#   bit 1: sell_volume_increased
#   bit 2: spread_high (spread > threshold)
#   bit 3: price_jump (price moved > 2 pips from previous)
# =============================================================================

_TICK_PACK_FMT = "<12s Q d d d Q I"
_TICK_STRUCT_SIZE = struct.calcsize(_TICK_PACK_FMT)  # 56 bytes

# Scaling factor: MT5 stores price as integer (pips × 10)
# e.g. 2500.00 -> 25000000
_PRICE_SCALE = 100000.0


@dataclass(slots=True)
class TickFrame:
    """
    Parsed tick data tu MT5 Expert Advisor.

    Thuong duoc deserialize tu binary ZeroMQ message.
    """

    symbol: str
    timestamp_us: int  # microseconds since epoch
    bid: float
    ask: float
    last: float
    volume: float
    flags: int

    @property
    def spread_pips(self) -> float:
        """Spread tính bằng pips (4 chữ số thập phân)."""
        return (self.ask - self.bid) * 10000.0

    @property
    def mid_price(self) -> float:
        """Giá trung binh bid-ask."""
        return (self.bid + self.ask) / 2.0

    @property
    def timestamp_sec(self) -> int:
        """Timestamp làm tròn xuống giây (dùng cho OHLCV)."""
        return self.timestamp_us // 1_000_000

    def is_abnormal_spread(self, threshold_pips: float = 0.5) -> bool:
        """Kiem tra spread bat thuong."""
        return self.spread_pips > threshold_pips

    def aggressor_side(self, prev_price: float | None = None) -> str:
        """
        Xac dinh bên mua hoac bán là aggressor (taker).

        Quy tac:
        - Neu last >= ask -> BUY (người mua lấy liquidity phía ask)
        - Neu last <= bid -> SELL (người bán lấy liquidity phía bid)
        - Nguoc lai: UNKNOWN

        Args:
            prev_price: Gia last cua tick truoc (optional, de xu ly retest)
        """
        if prev_price is not None:
            # Neu price quay lai gan bid/ask sau khi vượt qua
            if prev_price > self.last and self.last <= self.bid * 1.0001:
                return "SELL"
            if prev_price < self.last and self.last >= self.ask * 0.9999:
                return "BUY"

        if self.last >= self.ask:
            return "BUY"
        if self.last <= self.bid:
            return "SELL"
        return "UNKNOWN"

    @classmethod
    def from_binary(cls, data: bytes) -> "TickFrame":
        """Deserialize tu binary ZeroMQ message."""
        if len(data) < _TICK_STRUCT_SIZE:
            raise ValueError(
                f"Binary data ngan hon expected ({len(data)} < {_TICK_STRUCT_SIZE})"
            )

        (
            symbol_bytes,
            timestamp_us,
            bid_scaled,
            ask_scaled,
            last_scaled,
            volume,
            flags,
        ) = struct.unpack(_TICK_PACK_FMT, data[:_TICK_STRUCT_SIZE])

        symbol = symbol_bytes.rstrip(b"\x00").decode("utf-8", errors="replace")

        return cls(
            symbol=symbol,
            timestamp_us=timestamp_us,
            bid=bid_scaled / _PRICE_SCALE,
            ask=ask_scaled / _PRICE_SCALE,
            last=last_scaled / _PRICE_SCALE,
            volume=float(volume),
            flags=flags,
        )

    def to_binary(self) -> bytes:
        """Serialize thanh binary cho ZeroMQ."""
        symbol_padded = self.symbol.encode("utf-8").ljust(12, b"\x00")[:12]
        return struct.pack(
            _TICK_PACK_FMT,
            symbol_padded,
            self.timestamp_us,
            int(self.bid * _PRICE_SCALE),
            int(self.ask * _PRICE_SCALE),
            int(self.last * _PRICE_SCALE),
            int(self.volume),
            self.flags,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "TickFrame":
        """Tu dictionary (thường từ JSON webhook)."""
        return cls(
            symbol=str(d.get("symbol", "")),
            timestamp_us=int(d.get("timestamp_us", 0)),
            bid=float(d.get("bid", 0.0)),
            ask=float(d.get("ask", 0.0)),
            last=float(d.get("last", 0.0)),
            volume=float(d.get("volume", 0.0)),
            flags=int(d.get("flags", 0)),
        )
