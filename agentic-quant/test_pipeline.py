import asyncio
import sys
sys.path.insert(0, '.')

# Windows: cho zmq hoat dong voi asyncio
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from core.ingestion import (
    MT5TickSimulator,
    TickReceiver,
    OHLCVAggregator,
    MTFSynchronizer,
    TIMEFRAME_SECONDS,
)
from core.utils.events import EventBus, EventType


async def test_full_pipeline():
    print("=== Testing MT5 Pipeline ===\n")

    # 1. Simulator
    sim = MT5TickSimulator(
        symbol="XAUUSD",
        base_price=4450.0,
        volatility=0.05,   # ~0.5 pip per tick
        spread=0.5,        # ~5 pips spread
        push_address="tcp://127.0.0.1:5558",
        tick_interval_ms=0,
    )

    # 2. TickReceiver
    bus = EventBus()
    receiver = TickReceiver(port=5558, event_bus=bus, abnormal_spread_threshold=10.0)

    # 3. OHLCV Aggregator
    ohlcv = OHLCVAggregator(event_bus=bus)

    # 4. MTF Synchronizer
    mtf = MTFSynchronizer(symbol="XAUUSD")

    # Subscribe: tick -> aggregator
    received_ticks = []

    async def on_tick(event):
        received_ticks.append(event)
        closed = ohlcv.process_tick(event)
        for tf in TIMEFRAME_SECONDS:
            bar = ohlcv.get_latest_bar("XAUUSD", tf)
            if bar:
                mtf.update_bar(tf, bar)

    bus.subscribe(EventType.TICK_RECEIVED, on_tick)

    # Start
    await sim.start()
    await receiver.start()

    # Wait for some ticks
    print("Dang cho tick...")
    for i in range(50):
        await asyncio.sleep(0.01)

    await sim.stop()
    await receiver.stop()

    # Results
    print(f"\nKet qua:")
    print(f"  Simulator ticks generated: {sim.tick_count}")
    print(f"  Receiver ticks received: {receiver.tick_count}")
    print(f"  EventBus ticks: {len(received_ticks)}")

    if received_ticks:
        t = received_ticks[-1]
        print(f"\nTick cuoi cung:")
        print(f"  bid={t.bid:.2f} ask={t.ask:.2f} last={t.last:.2f}")
        print(f"  spread={t.spread_pips:.1f}pips aggressor={t.aggressor}")

    m1 = ohlcv.get_latest_bar("XAUUSD", "M1")
    if m1:
        print(f"\nM1 bar hien tai:")
        print(f"  O={m1.open:.2f} H={m1.high:.2f} L={m1.low:.2f} C={m1.close:.2f}")
        print(f"  ticks={m1.tick_count} volume={m1.volume:.0f}")

    usv = mtf.build_usv(received_ticks[-1])
    print(f"\nUSV:")
    print(f"  symbol={usv.symbol}")
    print(f"  bars count={len(usv.bars)}")
    print(f"  price={usv.current_price:.2f}")

    print("\n=== Test PASSED ===")


if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
