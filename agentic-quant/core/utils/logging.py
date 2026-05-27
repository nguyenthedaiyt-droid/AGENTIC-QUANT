# =============================================================================
# AGENTIC-QUANT — Logging Setup
# =============================================================================

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger

# =============================================================================
# cau hinh logging
# =============================================================================

LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Kich thuoc rotate: 100MB
MAX_LOG_SIZE = 100 * 1024 * 1024
# Giu 7 ngay
RETENTION_DAYS = 7


def setup_logging(
    environment: str = "development",
    log_dir: Path = LOG_DIR,
) -> "Logger":
    """Cau hinh loguru cho he thong.

    Args:
        environment: 'development' | 'staging' | 'production'
        log_dir: Thu muc chua log files

    Returns:
        logger instance da cau hinh
    """
    # Xoa tat ca handlers mac dinh
    logger.remove()

    # Muc do logging theo moi truong
    env_levels = {
        "development": "DEBUG",
        "staging": "INFO",
        "production": "WARNING",
    }
    level = env_levels.get(environment, "DEBUG")

    # --- Console Handler ---
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # --- File Handlers ---
    log_dir.mkdir(exist_ok=True)

    # System log — tat ca module
    logger.add(
        log_dir / "system_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation=MAX_LOG_SIZE,
        retention=RETENTION_DAYS,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        serialize=False,
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    # Model log — chi AI model
    logger.add(
        log_dir / "model_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation=MAX_LOG_SIZE,
        retention=RETENTION_DAYS,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        serialize=False,
        enqueue=True,
        filter=lambda record: record["name"] in (
            "core.ai_engine.neural.model_a",
            "core.ai_engine.neural.model_b",
            "core.ai_engine.neural.lstm",
            "core.ai_engine.multi_agent",
        ),
    )

    # IPC log — chi IPC layer
    logger.add(
        log_dir / "ipc_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation=MAX_LOG_SIZE,
        retention=RETENTION_DAYS,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        serialize=False,
        enqueue=True,
        filter=lambda record: "ipc" in record["name"] or "websocket" in record["name"],
    )

    # --- JSON log cho production ---
    if environment == "production":
        logger.add(
            log_dir / "json_{time:YYYY-MM-DD}.log",
            level="INFO",
            rotation=MAX_LOG_SIZE,
            retention=RETENTION_DAYS,
            serialize=True,
            enqueue=True,
            format="{message}",
        )

    return logger


def get_logger(name: str | None = None) -> "Logger":
    """Lay logger instance, co the chi dinh ten module."""
    if name:
        return logger.bind(name=name)
    return logger
