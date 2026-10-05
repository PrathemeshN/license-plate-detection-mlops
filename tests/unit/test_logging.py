"""Tests for application logging configuration."""

import logging

from app.core.logging import configure_logging


def test_configure_logging_sets_requested_level() -> None:
    """Verify that logging uses the requested log level."""
    configure_logging("DEBUG")

    assert logging.getLogger().level == logging.DEBUG


def test_configure_logging_handles_invalid_level() -> None:
    """Verify that invalid levels fall back to INFO."""
    configure_logging("INVALID")

    assert logging.getLogger().level == logging.INFO
