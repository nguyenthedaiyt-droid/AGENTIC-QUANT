# =============================================================================
# AGENTIC-QUANT — Backtesting CLI Entry Point (Phase 8)
# Chay backtest, benchmark, hoac stress test tu command line
# =============================================================================
# Cach dung:
#   python -m core.backtesting.run --symbol XAUUSD --start 2024-01-01 --end 2024-03-31
#   python -m core.backtesting.run --symbol XAUUSD --start 2024-01-01 --end 2024-03-31 --mode benchmark
#   python -m core.backtesting.run --symbol XAUUSD --start 2024-01-01 --end 2024-03-31 --mode stress --output /tmp/report.json
# =============================================================================

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


# =============================================================================
# Constants
# =============================================================================
_DEFAULT_OUTPUT = "backtest_report.json"
_SUPPORTED_MODES = ("backtest", "benchmark", "stress")


# =============================================================================
# Argument Parser
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    """Xay dung argparse.ArgumentParser cho CLI backtest.

    Returns:
        argparse.ArgumentParser: Parser da cau hinh
    """
    parser = argparse.ArgumentParser(
        prog="python -m core.backtesting.run",
        description="Backtesting CLI — chay backtest, benchmark, hoac stress test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Vi du:\n"
            "  python -m core.backtesting.run --symbol XAUUSD --start 2024-01-01 --end 2024-03-31\n"
            "  python -m core.backtesting.run --symbol XAUUSD --start 2024-01-01 --end 2024-03-31 --mode benchmark\n"
            "  python -m core.backtesting.run --symbol XAUUSD --start 2024-01-01 --end 2024-03-31 --mode stress --output /tmp/report.json\n"
        ),
    )

    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Ma symbol (VD: XAUUSD, BTCUSD)",
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Ngay bat dau (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="Ngay ket thuc (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=_DEFAULT_OUTPUT,
        help=f"Duong dan file output JSON (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="backtest",
        choices=_SUPPORTED_MODES,
        help="Che do chay: backtest (mac dinh), benchmark, stress",
    )
    parser.add_argument(
        "--tick-dir",
        type=str,
        default="data/ticks/parquet",
        help="Thu muc chua Parquet tick files (default: data/ticks/parquet)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=10_000,
        help="So dong doc moi batch tu Parquet (default: 10000)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help="Toc do replay (seconds delay, 0 = max speed, default: 0)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="In log chi tiet",
    )

    return parser


# =============================================================================
# Async Main
# =============================================================================
async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    """Chay backtest pipeline async.

    Args:
        args: Namespace tu argparse

    Returns:
        Dict: Ket qua backtest (report dict)

    Raises:
        ImportError: Neu thieu module core.backtesting
    """
    # Cau hinh logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    logger.info(
        "Backtest CLI: symbol={symbol}, start={start}, end={end}, mode={mode}",
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        mode=args.mode,
    )

    # --- Import cac module can thiet ---
    try:
        from core.backtesting.event_driven_simulator import EventDrivenSimulator
        from core.backtesting.report_generator import BacktestReport, BacktestReportGenerator
        from core.backtesting.ic_calculator import ICCalculator
        from core.utils.events import EventBus
    except ImportError as exc:
        logger.error("Loi import module core.backtesting: {exc}", exc=exc)
        raise

    # --- Tao components ---
    benchmark_mode = args.mode == "benchmark"
    event_bus = EventBus()

    simulator = EventDrivenSimulator(
        event_bus=event_bus,
        tick_dir=args.tick_dir,
        benchmark_mode=benchmark_mode,
        chunksize=args.chunksize,
    )

    ic_calculator = ICCalculator()
    report_generator = BacktestReportGenerator(
        ic_calculator=ic_calculator,
    )

    # Set meta
    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    report_generator.set_meta(
        symbol=args.symbol,
        timeframe="tick",
        start_time=start_dt,
        end_time=end_dt,
    )

    # --- Chay simulator ---
    logger.info(
        "[CLI] Bat dau simulator.run(mode={mode}, speed={speed})",
        mode=args.mode,
        speed=args.speed,
    )

    stats = await simulator.run(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        speed=args.speed,
    )

    logger.info(
        "[CLI] Simulator hoan thanh: {ticks} ticks in {elapsed:.2f}s",
        ticks=stats.get("ticks_replayed", 0),
        elapsed=stats.get("elapsed_seconds", 0.0),
    )

    # --- Generate report ---
    report = report_generator.build()
    report_dict = report.to_dict()

    # --- Them simulator stats vao report ---
    report_dict["simulator_stats"] = stats
    report_dict["mode"] = args.mode

    logger.info(
        "[CLI] Report generated: trades={trades}, win_rate={wr:.2%}, "
        "sharpe={sharpe:.2f}, max_dd={dd:.2%}",
        trades=report_dict.get("total_trades", 0),
        wr=report_dict.get("win_rate", 0.0),
        sharpe=report_dict.get("sharpe_ratio", 0.0),
        dd=report_dict.get("max_drawdown", 0.0),
    )

    return report_dict


# =============================================================================
# Save Report to JSON
# =============================================================================
def save_report(report_dict: dict[str, Any], output_path: str) -> Path:
    """Luu report ra file JSON.

    Args:
        report_dict: Dict report can luu
        output_path: Duong dan file output

    Returns:
        Path: Duong dan file da luu
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Custom serializer cho datetime
    def json_serializer(obj: Any) -> str:
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    with open(out, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, default=json_serializer)

    logger.info("[CLI] Report saved to {path} ({size} bytes)", path=out, size=out.stat().st_size)
    return out


# =============================================================================
# CLI Entry Point
# =============================================================================
def main() -> None:
    """Entry point chinh cho CLI backtest.

    Parse arguments, chay async main, va luu ket qua.
    """
    parser = build_parser()
    args = parser.parse_args()

    # Validate mode
    if args.mode not in _SUPPORTED_MODES:
        logger.error("Mode khong ho tro: {mode}. Chon: {modes}", mode=args.mode, modes=_SUPPORTED_MODES)
        sys.exit(1)

    # Chay async main
    try:
        report_dict = asyncio.run(async_main(args))
    except KeyboardInterrupt:
        logger.info("[CLI] Bi interrupt boi nguoi dung")
        sys.exit(130)
    except Exception as exc:
        logger.error("[CLI] Loi: {exc}", exc=exc)
        sys.exit(1)

    # Luu report
    save_report(report_dict, args.output)

    # In summary ra console
    print("\n" + "=" * 60)
    print("BACKTEST COMPLETE")
    print("=" * 60)
    print(f"  Symbol:     {args.symbol}")
    print(f"  Period:     {args.start} -> {args.end}")
    print(f"  Mode:       {args.mode}")
    print(f"  Trades:     {report_dict.get('total_trades', 0)}")
    print(f"  Win Rate:   {report_dict.get('win_rate', 0.0):.2%}")
    print(f"  Profit F:   {report_dict.get('profit_factor', 0.0):.2f}")
    print(f"  Sharpe:     {report_dict.get('sharpe_ratio', 0.0):.2f}")
    print(f"  Max DD:     {report_dict.get('max_drawdown', 0.0):.2%}")
    print(f"  Avg Hold:   {report_dict.get('avg_hold_bars', 0.0):.1f} bars")
    print(f"  Output:     {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
