"""Data models used by the dataset validation pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BoundingBox:
    """Represent a normalized YOLO bounding box."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class DatasetSplit:
    """Represent one dataset split."""

    name: str
    image_directory: Path
    label_directory: Path


@dataclass(frozen=True)
class DatasetManifest:
    """Describe a versioned license-plate dataset."""

    dataset_name: str
    dataset_version: str
    dataset_root: Path
    class_names: tuple[str, ...]
    random_seed: int
    train_ratio: float
    validation_ratio: float
    test_ratio: float


@dataclass(frozen=True)
class ValidationResult:
    """Represent the result of dataset validation."""

    valid: bool
    split_name: str
    image_count: int
    label_count: int
    missing_labels: tuple[str, ...]
    orphan_labels: tuple[str, ...]
    invalid_labels: tuple[str, ...]
    invalid_images: tuple[str, ...]
