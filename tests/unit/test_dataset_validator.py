"""Tests for dataset-level YOLO validation."""

from pathlib import Path

import pytest

from preprocessing.dataset_validator import (
    build_dataset_splits,
    dataset_is_valid,
    validate_annotations,
    validate_dataset,
)
from preprocessing.models import DatasetSplit, ValidationResult


def _create_dataset_structure(
    root: Path,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> None:
    """Create a minimal YOLO dataset directory structure."""
    for split in splits:
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)


def test_build_dataset_splits(tmp_path: Path) -> None:
    """Verify that all required dataset splits are discovered."""
    _create_dataset_structure(tmp_path)

    splits = build_dataset_splits(tmp_path)

    assert len(splits) == 3
    assert [split.name for split in splits] == [
        "train",
        "val",
        "test",
    ]


def test_build_dataset_splits_rejects_missing_root(
    tmp_path: Path,
) -> None:
    """Verify that a missing dataset root is rejected."""
    missing_root = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="Dataset root"):
        build_dataset_splits(missing_root)


def test_build_dataset_splits_rejects_missing_split(
    tmp_path: Path,
) -> None:
    """Verify that an incomplete dataset is rejected."""
    _create_dataset_structure(
        tmp_path,
        splits=("train", "val"),
    )

    with pytest.raises(FileNotFoundError, match="test"):
        build_dataset_splits(tmp_path)


def test_validate_annotations_accepts_valid_files(
    tmp_path: Path,
) -> None:
    """Verify that valid annotation files pass validation."""
    image_directory = tmp_path / "images"
    label_directory = tmp_path / "labels"

    image_directory.mkdir()
    label_directory.mkdir()

    annotation_file = label_directory / "image_001.txt"
    annotation_file.write_text(
        "0 0.5 0.5 0.2 0.1\n",
        encoding="utf-8",
    )

    dataset_split = DatasetSplit(
        name="train",
        image_directory=image_directory,
        label_directory=label_directory,
    )

    invalid_annotations = validate_annotations(dataset_split)

    assert invalid_annotations == ()


def test_validate_annotations_detects_invalid_files(
    tmp_path: Path,
) -> None:
    """Verify that invalid annotation files are reported."""
    image_directory = tmp_path / "images"
    label_directory = tmp_path / "labels"

    image_directory.mkdir()
    label_directory.mkdir()

    annotation_file = label_directory / "image_001.txt"
    annotation_file.write_text(
        "1 0.5 0.5 0.2 0.1\n",
        encoding="utf-8",
    )

    dataset_split = DatasetSplit(
        name="train",
        image_directory=image_directory,
        label_directory=label_directory,
    )

    invalid_annotations = validate_annotations(dataset_split)

    assert invalid_annotations == (str(annotation_file),)


def test_validate_dataset_passes_valid_dataset(
    tmp_path: Path,
) -> None:
    """Verify that a valid dataset passes validation."""
    _create_dataset_structure(tmp_path)

    for split in ("train", "val", "test"):
        image_path = tmp_path / "images" / split / "image_001.jpg"
        label_path = tmp_path / "labels" / split / "image_001.txt"

        image_path.touch()
        label_path.write_text(
            "0 0.5 0.5 0.2 0.1\n",
            encoding="utf-8",
        )

    results = validate_dataset(tmp_path)

    assert len(results) == 3
    assert all(result.valid for result in results)
    assert dataset_is_valid(results) is True


def test_dataset_is_valid_rejects_invalid_result() -> None:
    """Verify that dataset validity reflects split failures."""
    result = ValidationResult(
        valid=False,
        split_name="train",
        image_count=1,
        label_count=1,
        missing_labels=("image_001",),
        orphan_labels=(),
        invalid_labels=(),
        invalid_images=(),
    )

    assert dataset_is_valid((result,)) is False
