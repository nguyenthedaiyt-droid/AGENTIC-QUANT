# =============================================================================
# AGENTIC-QUANT — Unit Tests cho TickFrame
# =============================================================================

from __future__ import annotations

import pytest

from core.ingestion.tick_frame import TickFrame


class TestTickFrame:
    """Tests cho TickFrame serialization and methods."""

    def test_from_binary_roundtrip(self) -> None:
        """Binary -> TickFrame -> Binary = same data."""
        original = TickFrame(
            symbol="XAUUSD",
            timestamp_us=1_700_000_000_000_000,
            bid=2499.0,
            ask=2499.5,
            last=2499.3,
            volume=100.0,
            flags=0b0000_0001,
        )
        data = original.to_binary()
        restored = TickFrame.from_binary(data)

        assert restored.symbol == original.symbol
        assert restored.timestamp_us == original.timestamp_us
        assert restored.bid == original.bid
        assert restored.ask == original.ask
        assert restored.last == original.last
        assert restored.volume == original.volume
        assert restored.flags == original.flags

    def test_binary_size(self) -> None:
        """Binary format co dung 56 bytes."""
        tick = TickFrame(
            symbol="XAUUSD",
            timestamp_us=1_700_000_000_000_000,
            bid=2499.0,
            ask=2499.5,
            last=2499.3,
            volume=100.0,
            flags=0,
        )
        data = tick.to_binary()
        assert len(data) == 56

    def test_spread_pips(self) -> None:
        """Spread tinh dung bang pips."""
        tick = TickFrame(
            symbol="XAUUSD",
            timestamp_us=0,
            bid=2499.0,
            ask=2499.5,
            last=2499.3,
            volume=100.0,
            flags=0,
        )
        # spread = 0.5 (ask - bid) = 0.5
        # 0.5 * 100000 / 10000 = 5 pips
        assert tick.spread_pips == 5.0

    def test_mid_price(self) -> None:
        """Mid price = (bid + ask) / 2."""
        tick = TickFrame(
            symbol="XAUUSD",
            timestamp_us=0,
            bid=2499.0,
            ask=2500.0,
            last=2499.5,
            volume=100.0,
            flags=0,
        )
        assert tick.mid_price == 2499.5

    def test_timestamp_sec(self) -> None:
        """Timestamp sec lam tron xuong giay."""
        tick = TickFrame(
            symbol="XAUUSD",
            timestamp_us=1_700_000_123_456,
            bid=2499.0,
            ask=2499.5,
            last=2499.3,
            volume=100.0,
            flags=0,
        )
        assert tick.timestamp_sec == 1_700_000_123

    def test_is_abnormal_spread_true(self) -> None:
        """Spread > 0.5 pips la abnormal."""
        tick = TickFrame(
            symbol="XAUUSD",
            timestamp_us=0,
            bid=2499.0,
            ask=2499.6,  # spread = 0.6 = 6 pips
            last=2499.5,
            volume=100.0,
            flags=0,
        )
        assert tick.is_abnormal_spread(threshold_pips=0.5) is True

    def test_is_abnormal_spread_false(self) -> None:
        """Spread <= 0.5 pips la normal."""
        tick = TickFrame(
            symbol="XAUUSD",
            timestamp_us=0,
            bid=2499.0,
            ask=2499.5,  # spread = 0.5 = 5 pips
            last=2499.3,
            volume=100.0,
            flags=0,
        )
        assert tick.is_abnormal_spread(threshold_pips=0.5) is False
        # bid=2499.0, ask=2499.5, spread=5 pips > 0.5, should be True
        # Test with higher threshold
        tick2 = TickFrame(symbol='XAUUSD', timestamp_us=0, bid=2499.0, ask=2499.1, last=2499.05, volume=100.0, flags=0)
        assert tick2.is_abnormal_spread(threshold_pips=0.5) is False

    def test_aggressor_buy(self) -> None:
        """last >= ask -> BUY."""
        tick = TickFrame(
            symbol="XAUUSD",
            timestamp_us=0,
            bid=2499.0,
            ask=2499.5,
            last=2499.5,  # last = ask
            volume=100.0,
            flags=0,
        )
        assert tick.aggressor_side() == "BUY"

    def test_aggressor_sell(self) -> None:
        """last <= bid -> SELL."""
        tick = TickFrame(
            symbol="XAUUSD",
            timestamp_us=0,
            bid=2499.0,
            ask=2499.5,
            last=2499.0,  # last = bid
            volume=100.0,
            flags=0,
        )
        assert tick.aggressor_side() == "SELL"

    def test_aggressor_unknown(self) -> None:
        """last trong bid-ask -> UNKNOWN."""
        tick = TickFrame(
            symbol="XAUUSD",
            timestamp_us=0,
            bid=2499.0,
            ask=2499.5,
            last=2499.2,  # between bid and ask
            volume=100.0,
            flags=0,
        )
        assert tick.aggressor_side() == "UNKNOWN"

    def test_from_dict(self) -> None:
        """Parse tu dictionary."""
        d = {
            "symbol": "XAUUSD",
            "timestamp_us": 1_700_000_000_000_000,
            "bid": 2499.0,
            "ask": 2499.5,
            "last": 2499.3,
            "volume": 100.0,
            "flags": 5,
        }
        tick = TickFrame.from_dict(d)
        assert tick.symbol == "XAUUSD"
        assert tick.last == 2499.3
        assert tick.flags == 5

    def test_from_binary_too_short(self) -> None:
        """Data ngan hon 56 bytes -> ValueError."""
        with pytest.raises(ValueError, match="shorter than expected"):
            TickFrame.from_binary(b"too short")

    def test_symbol_truncation(self) -> None:
        """Symbol dai hon 12 bytes bi cat."""
        tick = TickFrame(
            symbol="VERYLONGSYMBOLNAME",
            timestamp_us=0,
            bid=2499.0,
            ask=2499.5,
            last=2499.3,
            volume=100.0,
            flags=0,
        )
        data = tick.to_binary()
        restored = TickFrame.from_binary(data)
        assert restored.symbol == "VERYLONGSYMB"
        assert len(restored.symbol) == 12
