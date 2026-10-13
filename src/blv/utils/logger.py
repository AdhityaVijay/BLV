"""Centralized logger setup so every module logs consistently."""

import logging
import os
from pathlib import Path


def get_logger(name: str, level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """Return a configured logger.

    Args:
        name: usually __name__ of the calling module.
        level: logging level name, e.g. "INFO", "DEBUG".
        log_file: optional path to also write logs to a file.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured (e.g. imported twice) — don't add duplicate handlers.
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S"
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
