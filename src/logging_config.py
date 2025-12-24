from __future__ import annotations

import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler

_LOGGER_INITIALIZED = False

DEFAULT_LOG_FILE = os.getenv("SD_LOG_FILE", "app.log")
DEFAULT_LOG_LEVEL = os.getenv("SD_LOG_LEVEL", "INFO").upper()


def init_logging(name: str = "shoplifting_app") -> logging.Logger:
    """Initialize logging once and return a logger for the app.

    Environment variables:
      - SD_LOG_LEVEL: DEBUG|INFO|WARNING|ERROR|CRITICAL (default: INFO)
      - SD_LOG_DIR: directory to write logs (default: logs)
      - SD_LOG_FILE: filename (default: app.log)
      - SD_LOG_CONSOLE: '1' to also log to console (default: '1')
    """
    global _LOGGER_INITIALIZED
    logger = logging.getLogger(name)
    if _LOGGER_INITIALIZED:
        return logger

    level_name = os.getenv("SD_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    level = getattr(logging, level_name, logging.INFO)

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler (avoid duplicates)
    if os.getenv("SD_LOG_CONSOLE", "1") == "1":
        if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler) for h in root.handlers):
            console = logging.StreamHandler()
            console.setLevel(level)
            console.setFormatter(formatter)
            root.addHandler(console)

    # File handler
    log_dir = Path(os.getenv("SD_LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / os.getenv("SD_LOG_FILE", DEFAULT_LOG_FILE)
    try:
        fh = RotatingFileHandler(str(log_file), maxBytes=int(os.getenv("SD_LOG_MAX_BYTES", "2000000")), backupCount=int(os.getenv("SD_LOG_BACKUP_COUNT", "3")), encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
            root.addHandler(fh)
    except Exception as e:
        logger.warning("Failed to create file handler: %s", e)

    _LOGGER_INITIALIZED = True
    logger.debug("Logging initialized level=%s file=%s", level_name, log_file)
    return logger
