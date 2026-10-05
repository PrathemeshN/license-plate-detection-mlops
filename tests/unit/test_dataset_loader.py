"""Unit tests for the YOLO dataset loader."""

from pathlib import Path

import pytest
import yaml

from preprocessing.dataset_loader import (
    REQUIRED_SPLITS,
    load_dataset,
    load_dataset_manifest,
    load_dataset_splits,
)


def create_dataset_structure(
    root: Path,
    *,
    create_config: bool = True,
    missing_image_split: str | None = None,
    missing_label_split: str | None = None,
) -> None:
    """Create a minimal YOLO dataset structure for testing."""
    for split in REQUIRED_SPLITS:
        image_directory = root / "images" / split
        label_directory = root / "labels" / split

        if split != missing_image_split:
            image_directory.mkdir(parents=True, exist_ok=True)

        if split != missing_label_split:
            label_directory.mkdir(parents=True, exist_ok=True)

    if create_config:
        configuration = {
            "path": str(root),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {0: "license_plate"},
        }

        with (root / "data.yaml").open(
            "w",
            encoding="utf-8",
        ) as file:
            yaml.safe_dump(configuration, file)


def test_load_dataset_manifest_with_mapping_names(tmp_path: Path) -> None:
    """Load a manifest when names is represented as a mapping."""
    create_dataset_structure(tmp_path)

    manifest = load_dataset_manifest(tmp_path)

    assert manifest.dataset_name == tmp_path.name
    assert manifest.dataset_version == "unknown"
    assert manifest.dataset_root == tmp_path
    assert manifest.class_names == ("license_plate",)
    assert manifest.random_seed == 42
    assert manifest.train_ratio == 0.8
    assert manifest.validation_ratio == 0.1
    assert manifest.test_ratio == 0.1


def test_load_dataset_manifest_with_list_names(tmp_path: Path) -> None:
    """Load a manifest when names is represented as a list."""
    create_dataset_structure(tmp_path)

    configuration = {
        "names": ["license_plate", "vehicle"],
    }

    with (tmp_path / "data.yaml").open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(configuration, file)

    manifest = load_dataset_manifest(tmp_path)

    assert manifest.class_names == ("license_plate", "vehicle")


def test_load_dataset_manifest_sorts_mapping_names(
    tmp_path: Path,
) -> None:
    """Class names from a mapping are ordered by class ID."""
    create_dataset_structure(tmp_path)

    configuration = {
        "names": {
            2: "truck",
            0: "license_plate",
            1: "car",
        },
    }

    with (tmp_path / "data.yaml").open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(configuration, file)

    manifest = load_dataset_manifest(tmp_path)

    assert manifest.class_names == (
        "license_plate",
        "car",
        "truck",
    )


def test_load_dataset_manifest_missing_config(
    tmp_path: Path,
) -> None:
    """Raise an error when data.yaml does not exist."""
    with pytest.raises(
        FileNotFoundError,
        match="Dataset configuration not found",
    ):
        load_dataset_manifest(tmp_path)


@pytest.mark.parametrize(
    "configuration",
    [
        None,
        [],
        "invalid",
        123,
    ],
)
def test_load_dataset_manifest_requires_mapping(
    tmp_path: Path,
    configuration: object,
) -> None:
    """Reject YAML configurations that are not mappings."""
    with (tmp_path / "data.yaml").open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(configuration, file)

    with pytest.raises(
        ValueError,
        match="data.yaml must contain a mapping",
    ):
        load_dataset_manifest(tmp_path)


@pytest.mark.parametrize(
    "names",
    [
        None,
        "license_plate",
        123,
    ],
)
def test_load_dataset_manifest_requires_class_names(
    tmp_path: Path,
    names: object,
) -> None:
    """Reject configurations without valid class names."""
    configuration = {"names": names}

    with (tmp_path / "data.yaml").open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(configuration, file)

    with pytest.raises(
        ValueError,
        match="must define class names",
    ):
        load_dataset_manifest(tmp_path)


@pytest.mark.parametrize(
    "names",
    [
        [],
        {},
    ],
)
def test_load_dataset_manifest_rejects_empty_classes(
    tmp_path: Path,
    names: object,
) -> None:
    """Reject configurations with no classes."""
    configuration = {"names": names}

    with (tmp_path / "data.yaml").open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(configuration, file)

    with pytest.raises(
        ValueError,
        match="at least one class",
    ):
        load_dataset_manifest(tmp_path)


def test_load_dataset_splits(tmp_path: Path) -> None:
    """Load all required dataset splits."""
    create_dataset_structure(tmp_path)

    splits = load_dataset_splits(tmp_path)

    assert len(splits) == 3
    assert tuple(split.name for split in splits) == (
        "train",
        "val",
        "test",
    )

    for split in splits:
        assert split.image_directory == (tmp_path / "images" / split.name)
        assert split.label_directory == (tmp_path / "labels" / split.name)


def test_load_dataset_splits_missing_image_directory(
    tmp_path: Path,
) -> None:
    """Raise an error when a split image directory is missing."""
    create_dataset_structure(
        tmp_path,
        missing_image_split="val",
    )

    with pytest.raises(
        FileNotFoundError,
        match="Image directory not found for val",
    ):
        load_dataset_splits(tmp_path)


def test_load_dataset_splits_missing_label_directory(
    tmp_path: Path,
) -> None:
    """Raise an error when a split label directory is missing."""
    create_dataset_structure(
        tmp_path,
        missing_label_split="test",
    )

    with pytest.raises(
        FileNotFoundError,
        match="Label directory not found for test",
    ):
        load_dataset_splits(tmp_path)


def test_load_dataset_combines_manifest_and_splits(
    tmp_path: Path,
) -> None:
    """Load the dataset manifest and split definitions together."""
    create_dataset_structure(tmp_path)

    manifest, splits = load_dataset(tmp_path)

    assert manifest.dataset_root == tmp_path
    assert manifest.class_names == ("license_plate",)
    assert tuple(split.name for split in splits) == (
        "train",
        "val",
        "test",
    )
