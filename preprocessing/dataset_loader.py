"""Dataset loading utilities for YOLO-format license plate datasets."""

from __future__ import annotations

from pathlib import Path

import yaml

from preprocessing.models import DatasetManifest, DatasetSplit

REQUIRED_SPLITS = ("train", "val", "test")


def load_dataset_manifest(dataset_root: Path) -> DatasetManifest:
    """Load dataset metadata from data.yaml.

    Args:
        dataset_root: Root directory of the YOLO dataset.

    Returns:
        DatasetManifest containing dataset metadata.

    Raises:
        FileNotFoundError: If data.yaml does not exist.
        ValueError: If required dataset configuration is missing or invalid.
    """
    yaml_path = dataset_root / "data.yaml"

    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as file:
        configuration = yaml.safe_load(file)

    if not isinstance(configuration, dict):
        raise ValueError("data.yaml must contain a mapping.")

    names = configuration.get("names")

    if isinstance(names, dict):
        class_names = tuple(str(names[key]) for key in sorted(names))
    elif isinstance(names, list):
        class_names = tuple(str(name) for name in names)
    else:
        raise ValueError("data.yaml must define class names under 'names'.")

    if not class_names:
        raise ValueError("Dataset must define at least one class.")

    return DatasetManifest(
        dataset_name=dataset_root.name,
        dataset_version="unknown",
        dataset_root=dataset_root,
        class_names=class_names,
        random_seed=42,
        train_ratio=0.8,
        validation_ratio=0.1,
        test_ratio=0.1,
    )


def load_dataset_splits(
    dataset_root: Path,
) -> tuple[DatasetSplit, ...]:
    """Load train, validation, and test dataset splits.

    Args:
        dataset_root: Root directory of the YOLO dataset.

    Returns:
        Tuple containing the configured dataset splits.

    Raises:
        FileNotFoundError: If a required split directory is missing.
    """
    splits: list[DatasetSplit] = []

    for split_name in REQUIRED_SPLITS:
        image_directory = dataset_root / "images" / split_name
        label_directory = dataset_root / "labels" / split_name

        if not image_directory.is_dir():
            raise FileNotFoundError(
                f"Image directory not found for {split_name}: " f"{image_directory}"
            )

        if not label_directory.is_dir():
            raise FileNotFoundError(
                f"Label directory not found for {split_name}: " f"{label_directory}"
            )

        splits.append(
            DatasetSplit(
                name=split_name,
                image_directory=image_directory,
                label_directory=label_directory,
            )
        )

    return tuple(splits)


def load_dataset(
    dataset_root: Path,
) -> tuple[DatasetManifest, tuple[DatasetSplit, ...]]:
    """Load dataset metadata and filesystem splits.

    Args:
        dataset_root: Root directory of the YOLO dataset.

    Returns:
        A tuple containing the dataset manifest and dataset splits.
    """
    manifest = load_dataset_manifest(dataset_root)
    splits = load_dataset_splits(dataset_root)

    return manifest, splits
