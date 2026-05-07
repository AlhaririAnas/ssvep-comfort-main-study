from __future__ import annotations

"""
Logging setup helpers.

This module only configures Python loggers. It does not write experiment data,
open PsychoPy windows, or connect to Cortex.
"""

import logging
import sys
from pathlib import Path


DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_path: str | Path,
    logger_name: str,
    *,
    file_level: int = logging.DEBUG,
    stream_level: int = logging.INFO,
    reset_handlers: bool = True,
) -> logging.Logger:
    """Configure a logger with one file handler and one stdout handler."""

    target = Path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if reset_handlers:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    file_handler = logging.FileHandler(target, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(stream_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def close_logger_handlers(logger: logging.Logger) -> None:
    """Close and remove all handlers from a logger."""

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


if __name__ == "__main__":
    log_file = Path("data") / "_demo_logging" / "demo.log"
    demo_logger = setup_logging(log_file, "demo_logger")
    demo_logger.info("Logging demo message.")
    close_logger_handlers(demo_logger)

    print("Logging utils module demo")
    print("This demo writes one small log file only.")
    print(f"Log path: {log_file}")
