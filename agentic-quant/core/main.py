# =============================================================================
# AGENTIC-QUANT — Backend Entry Point
# Khoi dong toan bo he thong bang: poetry run python -m core.main
# =============================================================================

from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path

# Them project root vao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.utils.logging import setup_logging, get_logger


async def main() -> None:
    """Entry point chinh cua backend.

    Chu trinh khoi dong (cold start):
        T+0.5s: Load config
        T+1.0s: Init Redis, SQLite, VectorDB
        T+3.0s: Connect ZeroMQ
        T+3.5s: Fetch calendar
        T+4.0s: Start WebSocket server
        T+4.1s: Print "AGENTIQ_BACKEND_READY"
    """
    logger = get_logger("core.main")

    logger.info("AGENTIC-QUANT Backend khoi dong...")

    # Load config
    logger.info("Dang tai cau hinh...")
    from core.config import MasterConfig

    try:
        cfg = MasterConfig.from_yaml_files(project_root / "config")
    except Exception:
        cfg = MasterConfig()

    logger.info(f"Cau hinh tai: {cfg.system.name} v{cfg.system.version}")
    logger.info(f"Moi truong: {cfg.system.environment}")
    logger.info(f"Symbol: {cfg.system.symbol}")

    # TODO: Khoi tao Redis connection
    logger.info("Dang ket noi Redis...")

    # TODO: Khoi tao SQLite

    # TODO: Khoi tao VectorDB

    # TODO: Load ML models (LSTM, XGBoost A/B)

    # TODO: Khoi dong ZeroMQ tick receiver

    # TODO: Khoi dong Calendar scraper

    # TODO: Khoi dong WebSocket server
    # port_file = project_root / ".." / ".." / "aq_ws_port.txt"
    # # Ghi port ra file de Tauri doc
    # port_file.parent.mkdir(exist_ok=True)
    # port_file.write_text(str(cfg.system.ports.websocket))

    # Thong bao ready
    print("AGENTIQ_BACKEND_READY", flush=True)
    logger.info("AGENTIC-QUANT Backend san sang!")

    # Keep running
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        logger.info("Backend duoc yeu cau dung.")
        raise


def run() -> None:
    """Ham entry point synchronous."""
    # Setup logging truoc
    setup_logging()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def signal_handler(sig, frame):
        logger = get_logger("core.main")
        logger.warning(f"Nhan signal {sig}, dang dung...")
        loop.call_soon_threadsafe(loop.stop)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        loop.run_until_complete(main())
    finally:
        loop.close()


if __name__ == "__main__":
    run()
