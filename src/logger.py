"""
Centralized Logging Configuration.
Configures both streaming (console) and persistent file-based logging handlers
under the logs/ directory with log rotation and standardized formatting.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Union

# Determine Project Root anchored to repository location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOGS_DIR = PROJECT_ROOT / "logs"

# Default log format with timestamps, logger names, and source line numbers
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_LOGGING_INITIALIZED = False


def setup_logging(
    log_dir: Union[str, Path] = DEFAULT_LOGS_DIR,
    log_filename: str = "assistant.log",
    file_level: int = logging.INFO,
    console_level: int = logging.WARNING,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB per log file
    backup_count: int = 5,
    force_reconfigure: bool = False
) -> logging.Logger:
    """
    Initializes and attaches streaming and file-based handlers to the root
    'doc_assistant' namespace logger.
    
    Args:
        log_dir: Destination directory for log files (default: project_root / 'logs/').
        log_filename: Primary log file name (default: 'assistant.log').
        file_level: Logging level threshold for disk persistence.
        console_level: Logging level threshold for standard error streaming.
        max_bytes: Maximum size per log file before rotation.
        backup_count: Number of rotated backup log files to retain.
        force_reconfigure: If True, clears existing handlers and re-attaches them.
        
    Returns:
        The configured project root logger ('doc_assistant').
    """
    global _LOGGING_INITIALIZED

    logger = logging.getLogger("docgrok")
    legacy_logger = logging.getLogger("doc_assistant")

    # Ensure logs directory is anchored relative to PROJECT_ROOT if not an absolute path
    path_obj = Path(log_dir)
    logs_path = path_obj if path_obj.is_absolute() else (PROJECT_ROOT / path_obj).resolve()
    logs_path.mkdir(parents=True, exist_ok=True)
    log_file = logs_path / log_filename

    current_file_handler = next((h for h in logger.handlers if isinstance(h, RotatingFileHandler)), None)
    if _LOGGING_INITIALIZED and not force_reconfigure:
        if current_file_handler and Path(current_file_handler.baseFilename).resolve() == log_file.resolve():
            return logger
        force_reconfigure = True

    # Set logger to the most verbose of the requested levels
    for l in (logger, legacy_logger):
        l.setLevel(min(file_level, console_level))
        l.propagate = False

    # Clear prior handlers cleanly if reconfiguring
    if force_reconfigure:
        for l in (logger, legacy_logger):
            for h in list(l.handlers):
                try:
                    h.close()
                except Exception:
                    pass
            l.handlers.clear()

    # Avoid adding duplicate handlers if already present
    has_file_handler = any(isinstance(h, RotatingFileHandler) for h in logger.handlers)
    has_stream_handler = any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler) for h in logger.handlers)

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    # 1. File-based handler with automatic rotation and UTF-8 encoding
    if not has_file_handler:
        file_handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        legacy_logger.addHandler(file_handler)

    # 2. Console stream handler (streaming to sys.stderr)
    if not has_stream_handler:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setLevel(console_level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        legacy_logger.addHandler(stream_handler)

    _LOGGING_INITIALIZED = True
    try:
        rel_display = log_file.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        rel_display = str(log_file)
    logger.info(f"DocGrok logging initialized. File logs destination: {rel_display}")
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Returns a child logger scoped within the 'docgrok' namespace.
    Ensures that setup_logging has been called at least once.
    """
    if not _LOGGING_INITIALIZED:
        setup_logging()
    
    if name.startswith("docgrok"):
        return logging.getLogger(name)
    if name.startswith("doc_assistant."):
        name = name[len("doc_assistant."):]
    elif name == "doc_assistant":
        return logging.getLogger("docgrok")
    return logging.getLogger(f"docgrok.{name}")
