"""Tests for YOLO annotation validation."""

from pathlib import Path

import pytest

from preprocessing.annotation_validator import (
    parse_annotation_line,
    validate_annotation_file,
)


def test_parse_valid_annotation() -> None:
    """Verify that a valid YOLO annotation is parsed."""
    result = parse_annotation_line(
        "0 0.5 0.5 0.2 0.1",
        Path("plate.txt"),
    )

    assert result.class_id == 0
    assert result.x_center == 0.5
    assert result.y_center == 0.5
    assert result.width == 0.2
    assert result.height == 0.1


def test_parse_annotation_rejects_wrong_field_count() -> None:
    """Verify that malformed annotations are rejected."""
    with pytest.raises(ValueError, match="expected 5 values"):
        parse_annotation_line(
            "0 0.5 0.5 0.2",
            Path("plate.txt"),
        )


def test_parse_annotation_rejects_non_numeric_values() -> None:
    """Verify that non-numeric annotation values are rejected."""
    with pytest.raises(ValueError, match="non-numeric"):
        parse_annotation_line(
            "0 abc 0.5 0.2 0.1",
            Path("plate.txt"),
        )


def test_parse_annotation_rejects_wrong_class() -> None:
    """Verify that unsupported class IDs are rejected."""
    with pytest.raises(ValueError, match="unsupported class ID"):
        parse_annotation_line(
            "1 0.5 0.5 0.2 0.1",
            Path("plate.txt"),
        )


@pytest.mark.parametrize(
    "annotation",
    [
        "0 -0.1 0.5 0.2 0.1",
        "0 1.1 0.5 0.2 0.1",
        "0 0.5 -0.1 0.2 0.1",
        "0 0.5 1.1 0.2 0.1",
        "0 0.5 0.5 -0.2 0.1",
        "0 0.5 0.5 0.2 -0.1",
    ],
)
def test_parse_annotation_rejects_invalid_coordinates(
    annotation: str,
) -> None:
    """Verify that invalid normalized coordinates are rejected."""
    with pytest.raises(ValueError):
        parse_annotation_line(annotation, Path("plate.txt"))


def test_parse_annotation_rejects_zero_width() -> None:
    """Verify that zero-width boxes are rejected."""
    with pytest.raises(ValueError, match="width"):
        parse_annotation_line(
            "0 0.5 0.5 0.0 0.1",
            Path("plate.txt"),
        )


def test_parse_annotation_rejects_zero_height() -> None:
    """Verify that zero-height boxes are rejected."""
    with pytest.raises(ValueError, match="height"):
        parse_annotation_line(
            "0 0.5 0.5 0.2 0.0",
            Path("plate.txt"),
        )


def test_validate_annotation_file(tmp_path: Path) -> None:
    """Verify that a valid annotation file is parsed."""
    annotation_file = tmp_path / "plate.txt"
    annotation_file.write_text(
        "0 0.5 0.5 0.2 0.1\n" "0 0.3 0.4 0.1 0.2\n",
        encoding="utf-8",
    )

    result = validate_annotation_file(annotation_file)

    assert len(result) == 2
    assert all(annotation.class_id == 0 for annotation in result)


def test_validate_annotation_file_rejects_empty_file(
    tmp_path: Path,
) -> None:
    """Verify that an empty annotation file is rejected."""
    annotation_file = tmp_path / "empty.txt"
    annotation_file.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        validate_annotation_file(annotation_file)


def test_validate_annotation_file_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """Verify that a missing annotation file is rejected."""
    annotation_file = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        validate_annotation_file(annotation_file)
