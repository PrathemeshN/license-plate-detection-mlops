"""Validation utilities for YOLO-format datasets."""

from pathlib import Path

from preprocessing.models import DatasetSplit, ValidationResult

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _find_images(directory: Path) -> set[str]:
    """Return image filenames contained in a directory."""
    return {
        path.stem
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def _find_labels(directory: Path) -> set[str]:
    """Return label filenames contained in a directory."""
    return {path.stem for path in directory.glob("*.txt") if path.is_file()}


def validate_split(dataset_split: DatasetSplit) -> ValidationResult:
    """Validate image and label pairing for one dataset split."""
    image_directory = dataset_split.image_directory
    label_directory = dataset_split.label_directory

    if not image_directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_directory}")

    if not label_directory.is_dir():
        raise FileNotFoundError(f"Label directory does not exist: {label_directory}")

    image_stems = _find_images(image_directory)
    label_stems = _find_labels(label_directory)

    missing_labels = tuple(sorted(image_stems - label_stems))
    orphan_labels = tuple(sorted(label_stems - image_stems))

    return ValidationResult(
        valid=not missing_labels and not orphan_labels,
        split_name=dataset_split.name,
        image_count=len(image_stems),
        label_count=len(label_stems),
        missing_labels=missing_labels,
        orphan_labels=orphan_labels,
        invalid_labels=(),
        invalid_images=(),
    )
