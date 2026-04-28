"""
logger.py — Structured Logging for the RAG Pipeline
=====================================================

Provides a factory function that returns module-specific loggers
with consistent formatting and colored console output via Rich.

Usage:
    from app.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Pipeline started")
    logger.warning("No PDFs found in directory")

Design Decision:
    We use Python's built-in `logging` module (not print statements)
    because it provides:
    - Severity levels (DEBUG/INFO/WARNING/ERROR/CRITICAL)
    - Module attribution (which file generated the message)
    - Easy output redirection (console, file, or both)
    - Zero external dependencies for the core logging path

    Rich is used only for the console handler to provide colored,
    readable output during development. If Rich is unavailable,
    we fall back to a plain StreamHandler gracefully.
"""

import logging
import sys
from typing import Optional


# ── Global state ─────────────────────────────────────────────────────────
# Track whether we've already configured the root handler to prevent
# duplicate log lines when multiple modules call get_logger().
_ROOT_CONFIGURED: bool = False

# Default logging level for all pipeline loggers.
DEFAULT_LOG_LEVEL: int = logging.INFO

# Log format for the fallback plain handler (no Rich).
PLAIN_FORMAT: str = (
    "%(asctime)s │ %(levelname)-8s │ %(name)-30s │ %(message)s"
)
PLAIN_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def _configure_root_logger(level: int = DEFAULT_LOG_LEVEL) -> None:
    """
    Configure the root logger with a Rich console handler.

    This function is idempotent — calling it multiple times has no
    additional effect after the first invocation.

    Falls back to a plain StreamHandler if Rich is not installed,
    ensuring the pipeline never crashes due to a missing dev dependency.

    Args:
        level: The minimum severity level to capture.
    """
    global _ROOT_CONFIGURED
    if _ROOT_CONFIGURED:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any pre-existing handlers to avoid duplicate output
    # (e.g., from Jupyter notebooks or interactive interpreters).
    root_logger.handlers.clear()

    try:
        # ── Rich Handler (preferred) ─────────────────────────────────
        from rich.logging import RichHandler

        rich_handler = RichHandler(
            level=level,
            show_time=True,
            show_level=True,
            show_path=False,        # We include the module name manually
            markup=True,            # Allow [bold], [red], etc. in messages
            rich_tracebacks=True,   # Pretty-print exceptions
            tracebacks_show_locals=False,  # Don't leak local vars in logs
        )
        rich_handler.setLevel(level)
        root_logger.addHandler(rich_handler)

    except ImportError:
        # ── Fallback: Plain handler ──────────────────────────────────
        plain_handler = logging.StreamHandler(sys.stdout)
        plain_handler.setLevel(level)
        formatter = logging.Formatter(
            fmt=PLAIN_FORMAT,
            datefmt=PLAIN_DATE_FORMAT,
        )
        plain_handler.setFormatter(formatter)
        root_logger.addHandler(plain_handler)

    _ROOT_CONFIGURED = True


def get_logger(
    name: str,
    level: Optional[int] = None,
) -> logging.Logger:
    """
    Factory function that returns a named logger for a specific module.

    Each module should call this once at the top of the file:
        logger = get_logger(__name__)

    The returned logger inherits the root logger's handler (Rich or plain)
    but can have its own severity level overridden if needed.

    Args:
        name:  The logger name. Convention is to pass __name__ so logs
               show the fully qualified module path (e.g., 'app.core.chunker').
        level: Optional override for this specific logger's level.
               If None, inherits from the root logger.

    Returns:
        A configured logging.Logger instance.
    """
    # Ensure the root logger is set up before creating child loggers.
    _configure_root_logger()

    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)

    return logger
