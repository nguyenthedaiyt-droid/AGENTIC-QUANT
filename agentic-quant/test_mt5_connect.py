import MetaTrader5 as mt5
import time

mt5.initialize()

sym = 'XAUUSD'
info = mt5.symbol_info(sym)
print(f"{sym}: bid={info.bid}, ask={info.ask}, spread={info.spread} points")

# Lay tick moi nhat
tick = mt5.symbol_info_tick(sym)
if tick:
    print(f"Last tick: time_msc={tick.time_msc}, bid={tick.bid}, ask={tick.ask}, last={tick.last}, volume={tick.volume}")
else:
    print("Khong lay duoc tick")

# Lay lich su ticks
ticks = mt5.copy_ticks_from(sym, 0, 100, mt5.COPY_TICKS_ALL)
if ticks is not None:
    print(f"Da lay {len(ticks)} ticks tu MT5")
    for t in ticks[-3:]:
        print(f"  ts={t['time']} bid={t['bid']} ask={t['ask']} last={t['last']} vol={t['volume']}")
else:
    print("Khong lay ticks:", mt5.last_error())

mt5.shutdown()
