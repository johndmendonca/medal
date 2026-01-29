"""
Logging configuration for MEDAL framework.
"""
import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    stream: Optional[sys.__class__] = None,
) -> None:
    """
    Configure root logger for the MEDAL package.

    Args:
        level: Logging level (e.g. logging.INFO, logging.DEBUG).
        format_string: Custom format; default includes timestamp, level, name, message.
        stream: Output stream; defaults to sys.stderr.
    """
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
    if stream is None:
        stream = sys.stderr

    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(format_string))

    root = logging.getLogger("medal")
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger for a module.

    Args:
        name: Module name, typically __name__.

    Returns:
        Logger instance.
    """
    if name.startswith("medal."):
        return logging.getLogger(name)
    return logging.getLogger(f"medal.{name}")
