"""Application logging configuration."""

import logging
import sys


def configure_logging(log_level: str = "INFO") -> None:
    """Configure application-wide structured logging.

    Args:
        log_level: Logging level such as DEBUG, INFO, WARNING, or ERROR.
    """
    log_format = "%(asctime)s | %(levelname)s | " "%(name)s | %(message)s"

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
