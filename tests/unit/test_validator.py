"""Tests for YOLO dataset validation."""

from pathlib import Path

from preprocessing.models import DatasetSplit
from preprocessing.validator import validate_split


def test_validate_split_accepts_matching_files(tmp_path: Path) -> None:
    """Verify that matching images and labels pass validation."""
    image_directory = tmp_path / "images"
    label_directory = tmp_path / "labels"

    image_directory.mkdir()
    label_directory.mkdir()

    (image_directory / "image_001.jpg").touch()
    (label_directory / "image_001.txt").write_text(
        "0 0.5 0.5 0.2 0.1\n",
        encoding="utf-8",
    )

    dataset_split = DatasetSplit(
        name="train",
        image_directory=image_directory,
        label_directory=label_directory,
    )

    result = validate_split(dataset_split)

    assert result.valid is True
    assert result.image_count == 1
    assert result.label_count == 1
    assert result.missing_labels == ()
    assert result.orphan_labels == ()


def test_validate_split_detects_missing_label(tmp_path: Path) -> None:
    """Verify that an image without a label is rejected."""
    image_directory = tmp_path / "images"
    label_directory = tmp_path / "labels"

    image_directory.mkdir()
    label_directory.mkdir()

    (image_directory / "image_001.jpg").touch()

    dataset_split = DatasetSplit(
        name="train",
        image_directory=image_directory,
        label_directory=label_directory,
    )

    result = validate_split(dataset_split)

    assert result.valid is False
    assert result.missing_labels == ("image_001",)


def test_validate_split_detects_orphan_label(tmp_path: Path) -> None:
    """Verify that a label without an image is rejected."""
    image_directory = tmp_path / "images"
    label_directory = tmp_path / "labels"

    image_directory.mkdir()
    label_directory.mkdir()

    (label_directory / "image_001.txt").write_text(
        "0 0.5 0.5 0.2 0.1\n",
        encoding="utf-8",
    )

    dataset_split = DatasetSplit(
        name="train",
        image_directory=image_directory,
        label_directory=label_directory,
    )

    result = validate_split(dataset_split)

    assert result.valid is False
    assert result.orphan_labels == ("image_001",)
