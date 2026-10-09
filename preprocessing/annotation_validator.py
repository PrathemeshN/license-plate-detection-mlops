"""Validation utilities for YOLO annotation files."""

from pathlib import Path

from preprocessing.models import BoundingBox

EXPECTED_CLASS_ID = 0
EXPECTED_FIELD_COUNT = 5
MIN_NORMALIZED_VALUE = 0.0
MAX_NORMALIZED_VALUE = 1.0


def parse_annotation_line(line: str, source: Path) -> BoundingBox:
    """Parse and validate one YOLO annotation line.

    Args:
        line: Raw YOLO annotation line.
        source: Path of the source annotation file.

    Returns:
        A validated normalized bounding box.

    Raises:
        ValueError: If the annotation is malformed or invalid.
    """
    values = line.strip().split()

    if len(values) != EXPECTED_FIELD_COUNT:
        raise ValueError(
            f"{source}: expected {EXPECTED_FIELD_COUNT} values, " f"got {len(values)}."
        )

    try:
        class_id = int(values[0])
        x_center = float(values[1])
        y_center = float(values[2])
        width = float(values[3])
        height = float(values[4])
    except ValueError as exc:
        raise ValueError(f"{source}: annotation contains non-numeric values.") from exc

    if class_id != EXPECTED_CLASS_ID:
        raise ValueError(
            f"{source}: unsupported class ID {class_id}. "
            f"Expected {EXPECTED_CLASS_ID}."
        )

    coordinates = {
        "x_center": x_center,
        "y_center": y_center,
        "width": width,
        "height": height,
    }

    for name, value in coordinates.items():
        if not MIN_NORMALIZED_VALUE <= value <= MAX_NORMALIZED_VALUE:
            raise ValueError(
                f"{source}: {name}={value} is outside " "the normalized range [0, 1]."
            )

    if width <= 0.0:
        raise ValueError(f"{source}: width must be greater than zero.")

    if height <= 0.0:
        raise ValueError(f"{source}: height must be greater than zero.")

    return BoundingBox(
        class_id=class_id,
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
    )


def validate_annotation_file(path: Path) -> tuple[BoundingBox, ...]:
    """Validate every annotation in a YOLO label file.

    Args:
        path: Path to the YOLO annotation file.

    Returns:
        Tuple containing all validated bounding boxes.

    Raises:
        FileNotFoundError: If the annotation file does not exist.
        ValueError: If the annotation file is empty or contains
            an invalid annotation.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file does not exist: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()

    if not lines:
        raise ValueError(f"{path}: annotation file is empty.")

    annotations = []

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"{path}: empty annotation line at line {line_number}.")

        annotations.append(parse_annotation_line(line, path))

    return tuple(annotations)
