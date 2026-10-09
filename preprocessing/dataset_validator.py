"""Dataset-level validation for YOLO-format datasets."""

from pathlib import Path

from preprocessing.annotation_validator import validate_annotation_file
from preprocessing.models import DatasetSplit, ValidationResult
from preprocessing.validator import validate_split

REQUIRED_SPLITS = ("train", "val", "test")


def build_dataset_splits(dataset_root: Path) -> tuple[DatasetSplit, ...]:
    """Build dataset split definitions from a YOLO dataset root.

    Args:
        dataset_root: Root directory containing images and labels.

    Returns:
        Tuple containing the configured train, validation, and test splits.

    Raises:
        FileNotFoundError: If the required dataset directories do not exist.
    """
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    splits: list[DatasetSplit] = []

    for split_name in REQUIRED_SPLITS:
        image_directory = dataset_root / "images" / split_name
        label_directory = dataset_root / "labels" / split_name

        if not image_directory.is_dir():
            raise FileNotFoundError(
                f"Missing image directory for '{split_name}': " f"{image_directory}"
            )

        if not label_directory.is_dir():
            raise FileNotFoundError(
                f"Missing label directory for '{split_name}': " f"{label_directory}"
            )

        splits.append(
            DatasetSplit(
                name=split_name,
                image_directory=image_directory,
                label_directory=label_directory,
            )
        )

    return tuple(splits)


def validate_annotations(
    dataset_split: DatasetSplit,
) -> tuple[str, ...]:
    """Validate all YOLO annotation files in one dataset split.

    Args:
        dataset_split: Dataset split to validate.

    Returns:
        Tuple containing paths of invalid annotation files.
    """
    invalid_annotations: list[str] = []

    for annotation_path in sorted(dataset_split.label_directory.glob("*.txt")):
        try:
            validate_annotation_file(annotation_path)
        except (ValueError, UnicodeDecodeError):
            invalid_annotations.append(str(annotation_path))

    return tuple(invalid_annotations)


def validate_dataset(
    dataset_root: Path,
) -> tuple[ValidationResult, ...]:
    """Validate all required splits in a YOLO dataset.

    Args:
        dataset_root: Root directory of the YOLO dataset.

    Returns:
        Validation results for train, validation, and test splits.

    Raises:
        FileNotFoundError: If the dataset structure is incomplete.
    """
    splits = build_dataset_splits(dataset_root)
    results: list[ValidationResult] = []

    for dataset_split in splits:
        result = validate_split(dataset_split)
        invalid_annotations = validate_annotations(dataset_split)

        results.append(
            ValidationResult(
                valid=result.valid and not invalid_annotations,
                split_name=result.split_name,
                image_count=result.image_count,
                label_count=result.label_count,
                missing_labels=result.missing_labels,
                orphan_labels=result.orphan_labels,
                invalid_labels=invalid_annotations,
                invalid_images=result.invalid_images,
            )
        )

    return tuple(results)


def dataset_is_valid(
    results: tuple[ValidationResult, ...],
) -> bool:
    """Return whether every dataset split passed validation.

    Args:
        results: Dataset validation results.

    Returns:
        True when every split is valid.
    """
    return all(result.valid for result in results)
