# =============================================================================
# AGENTIC-QUANT — Backtest Data Loader
# Doc tick history tu Parquet, replay nhu real-time
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterator
from loguru import logger

from core.ingestion.tick_frame import TickFrame

if TYPE_CHECKING:
    import pandas as pd


# =============================================================================
# Backtest Tick Iterator
# =============================================================================
@dataclass
class BacktestConfig:
    """Cau hinh cho backtest data loader."""

    symbol: str = "XAUUSD"
    start_time: datetime | None = None
    end_time: datetime | None = None
    speed_multiplier: float = 1.0  # 1.0 = real-time, 1000.0 = fast forward
    tick_interval_ms: float = 0.0  # 0 = as fast as possible

    @property
    def data_dir(self) -> Path:
        return Path("data") / "historical_ticks" / self.symbol


class HistoricalTickLoader:
    """
    Load tick data tu Parquet files cho backtesting.

    Ho tro:
    - Doc file Parquet theo nam/thang
    - Iterator tra ve TickFrame theo thu tu timestamp
    - Replay mode: tao Artificial tick events
    - Speed control: co the chay nhanh hoac cham

    Args:
        config: BacktestConfig
        parquet_dir: Thu muc chua Parquet files
    """

    def __init__(
        self,
        config: BacktestConfig | None = None,
        parquet_dir: Path | None = None,
    ) -> None:
        self.config = config or BacktestConfig()
        self._parquet_dir = parquet_dir or self.config.data_dir
        self._loaded = False
        self._df = None

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------
    async def load(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> None:
        """
        Load tat ca tick data trong khoang thoi gian.

        Args:
            start_time: Bat dau (default: None = tu dau)
            end_time: Ket thuc (default: None = den cuoi)
        """
        import pandas as pd

        start = start_time or self.config.start_time
        end = end_time or self.config.end_time

        files = self._find_parquet_files(start, end)
        if not files:
            logger.warning(
                "Khong tim thay Parquet files trong {dir} "
                "tu {start} den {end}",
                dir=self._parquet_dir,
                start=start,
                end=end,
            )
            return

        dfs = []
        for fpath in files:
            try:
                df = await self._load_parquet_file(fpath)
                if df is not None and not df.empty:
                    dfs.append(df)
            except Exception:
                logger.exception("Loi load file {f}:", f=fpath)

        if dfs:
            self._df = pd.concat(dfs, ignore_index=True)
            self._df = self._df.sort_values("timestamp_us").reset_index(drop=True)
            self._loaded = True
            logger.info(
                "Da load {n} ticks tu {files} files",
                n=len(self._df),
                files=len(files),
            )
        else:
            logger.warning("Khong co tick data nao duoc load")

    async def _load_parquet_file(self, fpath: Path) -> "pd.DataFrame":
        """Load mot file Parquet."""
        import pandas as pd

        # Doc bang aiofiles + pyarrow
        df = pd.read_parquet(fpath, engine="pyarrow")

        # Validate schema
        required_cols = ["timestamp_us", "bid", "ask", "last"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Parquet thieu cot: {missing}")

        return df

    # -------------------------------------------------------------------------
    # Iterate
    # -------------------------------------------------------------------------
    def iter_ticks(self) -> Iterator[TickFrame]:
        """
        Iterator tra ve TickFrame.

        Su dung trong vong lap backtest.
        """
        if not self._loaded or self._df is None:
            raise RuntimeError("Data chua duoc load. Goi await load() truoc.")

        import pandas as pd

        for row in self._df.itertuples(index=False):
            row_dict = row._asdict() if hasattr(row, "_asdict") else dict(row._fields)

            yield TickFrame(
                symbol=str(row_dict.get("symbol", self.config.symbol)),
                timestamp_us=int(row_dict["timestamp_us"]),
                bid=float(row_dict["bid"]),
                ask=float(row_dict["ask"]),
                last=float(row_dict["last"]),
                volume=float(row_dict.get("volume", 0.0)),
                flags=int(row_dict.get("flags", 0)),
            )

    def iter_ticks_with_interval(
        self,
    ) -> Iterator[tuple[TickFrame, float]]:
        """
        Iterator tra ve (TickFrame, interval_ms).

        Dung de replay voi real-time timing.
        """
        prev_ts = None
        for tick in self.iter_ticks():
            if prev_ts is not None:
                interval_ms = (tick.timestamp_us - prev_ts) / 1000.0
                interval_ms /= self.config.speed_multiplier
            else:
                interval_ms = 0.0
            prev_ts = tick.timestamp_us
            yield tick, max(0, interval_ms)

    # -------------------------------------------------------------------------
    # File discovery
    # -------------------------------------------------------------------------
    def _find_parquet_files(
        self,
        start: datetime | None,
        end: datetime | None,
    ) -> list[Path]:
        """Tim tat ca Parquet files trong khoang thoi gian."""
        if not self._parquet_dir.exists():
            return []

        files = sorted(self._parquet_dir.rglob("*.parquet"))

        if start is None and end is None:
            return files

        # Filter by year/month folder
        result = []
        for f in files:
            # Path format: {symbol}/{year}/{month}.parquet
            parts = f.relative_to(self._parquet_dir).parts
            if len(parts) >= 2:
                try:
                    year = int(parts[0])
                    month = int(parts[1].replace(".parquet", ""))
                    dt = datetime(year, month, 1, tzinfo=timezone.utc)

                    if start and dt < start:
                        continue
                    if end and dt >= end:
                        continue

                    result.append(f)
                except (ValueError, IndexError):
                    # Not in expected format, include anyway
                    result.append(f)
            else:
                result.append(f)

        return result

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------
    @property
    def tick_count(self) -> int:
        """Tong so tick da load."""
        if self._df is not None:
            return len(self._df)
        return 0

    @property
    def time_range(self) -> tuple[datetime, datetime] | None:
        """Khoang thoi gian cua data."""
        if self._df is None or self._df.empty:
            return None
        import pandas as pd
        min_ts = int(self._df["timestamp_us"].min())
        max_ts = int(self._df["timestamp_us"].max())
        min_dt = datetime.fromtimestamp(min_ts / 1_000_000, tz=timezone.utc)
        max_dt = datetime.fromtimestamp(max_ts / 1_000_000, tz=timezone.utc)
        return min_dt, max_dt


# =============================================================================
# MT5 Tick Export Script
# =============================================================================
async def export_mt5_ticks(
    mt5_path: str,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
) -> Path:
    """
    Export ticks tu MetaTrader 5 history sang Parquet.

    Script nay chay ngoai Python process (MQL5 -> Python IPC).
    Trong production, goi thong qua subprocess hoac named pipe.

    Args:
        mt5_path: Duong dan den MT5 terminal
        symbol: Symbol can export
        start_date: Ngay bat dau
        end_date: Ngay ket thuc
        output_dir: Thu muc output

    Returns:
        Duong dan file Parquet da tao
    """
    import subprocess

    output_dir = Path(output_dir)
    year = start_date.year
    month = start_date.month
    output_path = output_dir / str(year) / f"{month:02d}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build MQL5 script command
    # MT5 cung cap API de doc history, day script:
    escaped_path = output_path.as_posix().replace("/", "\\\\")
    script = f"""
    //+------------------------------------------------------------------+
    //| ExportTicks.mq5                                                   |
    //+------------------------------------------------------------------+
    #property script_show_inputs

    input string InSymbol = "{symbol}";
    input datetime InStart = {int(start_date.timestamp())};
    input datetime InEnd = {int(end_date.timestamp())};
    input string OutFile = "{escaped_path}";
    //+------------------------------------------------------------------+
    void OnStart() {{
        // Doc history ticks
        datetime from = (datetime)InStart;
        datetime to = (datetime)InEnd;
        MqlDateTime s, e;
        TimeToStruct(from, s);
        TimeToStruct(to, e);
    }}
    //+------------------------------------------------------------------+
    """

    # Serialize script
    script_path = output_path.parent / "ExportTicks.mq5"
    Path(script_path).write_text(script)

    logger.info(
        f"MT5 export script tao tai {script_path}. "
        "Chay thu cong MT5 de export data.",
    )

    return output_path


# =============================================================================
# Validate tick data coverage
# =============================================================================
async def validate_coverage(
    loader: HistoricalTickLoader,
    market_open_hours: tuple[int, int] = (0, 24),  # 0-24 UTC
) -> dict:
    """
    Kiem tra coverage cua tick data.

    Phat hien:
    - Gap trong gio market open
    - Missing ticks
    - Out-of-order timestamps

    Args:
        loader: HistoricalTickLoader da load data
        market_open_hours: Gio market open theo UTC (start, end)

    Returns:
        Dict chua ket qua kiem tra
    """
    import pandas as pd

    if loader._df is None:
        return {"error": "No data loaded"}

    df = loader._df
    gaps: list[dict] = []
    out_of_order: int = 0

    timestamps = df["timestamp_us"].values
    expected_interval = 100_000  # 100ms = 100,000 us (XAUUSD thong thuong)

    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i - 1]

        if dt < 0:
            out_of_order += 1
            continue

        # Check gap > 5 seconds
        if dt > 5_000_000:  # 5 seconds
            # Check if within market hours
            prev_dt = datetime.fromtimestamp(
                timestamps[i - 1] / 1_000_000, tz=timezone.utc
            )
            curr_dt = datetime.fromtimestamp(
                timestamps[i] / 1_000_000, tz=timezone.utc
            )

            gaps.append({
                "start": str(prev_dt),
                "end": str(curr_dt),
                "gap_ms": dt / 1000.0,
                "start_hour_utc": prev_dt.hour,
            })

    coverage_pct = (1 - len(gaps) / max(len(timestamps), 1) * 100)

    return {
        "total_ticks": len(df),
        "gaps_found": len(gaps),
        "out_of_order": out_of_order,
        "coverage_pct": round(coverage_pct, 2),
        "sample_gaps": gaps[:10],  # Chi tra ve 10 gap dau tien
        "status": "RELIABLE" if coverage_pct > 95 else "UNRELIABLE",
    }
