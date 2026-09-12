"""
logging_setup.py

Structured logging configuration for the Contract Intelligence Platform.
Every module obtains its logger via get_logger(__name__) instead of using
print statements. Logs are written both to console and to a rotating file
on disk, configured from config/settings.yaml.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from app.utils.config_loader import get_project_root, get_settings

_CONFIGURED = False


def _configure_root_logger() -> None:
    """
    Configure the root 'contract_platform' logger once per process, with a
    console handler and a rotating file handler. Idempotent: safe to call
    multiple times.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    log_cfg = settings.get("logging", {})

    level_name = log_cfg.get("level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)
    log_format = log_cfg.get(
        "format", "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    project_root = get_project_root()
    log_file_rel = log_cfg.get("log_file", "logs/app.log")
    log_file_path: Path = project_root / log_file_rel
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    max_bytes = int(log_cfg.get("max_bytes", 5_242_880))
    backup_count = int(log_cfg.get("backup_count", 3))

    formatter = logging.Formatter(log_format)

    root_logger = logging.getLogger("contract_platform")
    root_logger.setLevel(level)
    root_logger.propagate = False

    # Avoid duplicate handlers if this function somehow runs more than once.
    if not root_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_file_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger namespaced under 'contract_platform'.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        A configured logging.Logger instance.
    """
    _configure_root_logger()
    return logging.getLogger(f"contract_platform.{name}")
