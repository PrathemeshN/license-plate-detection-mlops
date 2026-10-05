"""Tests for application configuration."""

from app.core.config import Settings


def test_default_confidence_threshold() -> None:
    """Verify the default confidence threshold."""
    settings = Settings()

    assert settings.confidence_threshold == 0.38


def test_default_iou_threshold() -> None:
    """Verify the default IoU threshold."""
    settings = Settings()

    assert settings.iou_threshold == 0.50


def test_upload_limits_are_positive() -> None:
    """Verify upload limits have valid positive defaults."""
    settings = Settings()

    assert settings.max_image_size_mb > 0
    assert settings.max_zip_size_mb > 0
    assert settings.max_bulk_files > 0
