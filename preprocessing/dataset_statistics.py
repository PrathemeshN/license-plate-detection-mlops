"""Dataset-level statistics for YOLO-format license plate datasets."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median


@dataclass(frozen=True)
class BoundingBoxStatistics:
    """Statistics describing bounding boxes in a dataset split."""

    count: int
    mean_width: float
    median_width: float
    min_width: float
    max_width: float
    mean_height: float
    median_height: float
    min_height: float
    max_height: float
    mean_area: float
    median_area: float
    min_area: float
    max_area: float


@dataclass(frozen=True)
class SplitStatistics:
    """Statistics describing one YOLO dataset split."""

    name: str
    image_count: int
    annotation_count: int
    empty_annotation_files: int
    multi_plate_images: int
    boxes_per_image_mean: float
    boxes_per_image_median: float
    max_boxes_per_image: int
    bounding_boxes: BoundingBoxStatistics


def _read_yolo_annotations(label_path: Path) -> list[tuple[float, float]]:
    """Read normalized YOLO width and height values from a label file."""
    boxes: list[tuple[float, float]] = []

    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = line.strip()

        if not stripped:
            continue

        parts = stripped.split()

        if len(parts) != 5:
            raise ValueError(
                f"Invalid YOLO annotation at {label_path}:{line_number}. "
                f"Expected 5 values, found {len(parts)}."
            )

        try:
            width = float(parts[3])
            height = float(parts[4])
        except ValueError as exc:
            raise ValueError(
                f"Invalid numeric annotation at " f"{label_path}:{line_number}."
            ) from exc

        if not 0.0 < width <= 1.0:
            raise ValueError(
                f"Invalid normalized width at " f"{label_path}:{line_number}: {width}"
            )

        if not 0.0 < height <= 1.0:
            raise ValueError(
                f"Invalid normalized height at " f"{label_path}:{line_number}: {height}"
            )

        boxes.append((width, height))

    return boxes


def _build_box_statistics(
    boxes: Iterable[tuple[float, float]],
) -> BoundingBoxStatistics:
    """Calculate aggregate statistics for bounding boxes."""
    box_list = list(boxes)

    if not box_list:
        return BoundingBoxStatistics(
            count=0,
            mean_width=0.0,
            median_width=0.0,
            min_width=0.0,
            max_width=0.0,
            mean_height=0.0,
            median_height=0.0,
            min_height=0.0,
            max_height=0.0,
            mean_area=0.0,
            median_area=0.0,
            min_area=0.0,
            max_area=0.0,
        )

    widths = [width for width, _ in box_list]
    heights = [height for _, height in box_list]
    areas = [width * height for width, height in box_list]

    return BoundingBoxStatistics(
        count=len(box_list),
        mean_width=mean(widths),
        median_width=median(widths),
        min_width=min(widths),
        max_width=max(widths),
        mean_height=mean(heights),
        median_height=median(heights),
        min_height=min(heights),
        max_height=max(heights),
        mean_area=mean(areas),
        median_area=median(areas),
        min_area=min(areas),
        max_area=max(areas),
    )


def collect_split_statistics(
    dataset_root: Path,
    split: str,
) -> SplitStatistics:
    """Collect statistics for a YOLO dataset split.

    Args:
        dataset_root: Root directory containing ``images`` and ``labels``.
        split: Dataset split name, such as ``train``, ``val``, or ``test``.

    Returns:
        Statistics for the requested split.

    Raises:
        FileNotFoundError: If the split directory does not exist.
    """
    image_directory = dataset_root / "images" / split
    label_directory = dataset_root / "labels" / split

    if not image_directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_directory}")

    if not label_directory.is_dir():
        raise FileNotFoundError(f"Label directory does not exist: {label_directory}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    image_files = sorted(
        path
        for path in image_directory.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    )

    all_boxes: list[tuple[float, float]] = []
    boxes_per_image: list[int] = []
    empty_annotation_files = 0
    multi_plate_images = 0

    for image_path in image_files:
        label_path = label_directory / f"{image_path.stem}.txt"

        if not label_path.exists():
            raise FileNotFoundError(f"Missing annotation file for image: {image_path}")

        boxes = _read_yolo_annotations(label_path)

        if not boxes:
            empty_annotation_files += 1

        if len(boxes) > 1:
            multi_plate_images += 1

        boxes_per_image.append(len(boxes))
        all_boxes.extend(boxes)

    box_statistics = _build_box_statistics(all_boxes)

    return SplitStatistics(
        name=split,
        image_count=len(image_files),
        annotation_count=len(all_boxes),
        empty_annotation_files=empty_annotation_files,
        multi_plate_images=multi_plate_images,
        boxes_per_image_mean=(mean(boxes_per_image) if boxes_per_image else 0.0),
        boxes_per_image_median=(median(boxes_per_image) if boxes_per_image else 0.0),
        max_boxes_per_image=max(boxes_per_image, default=0),
        bounding_boxes=box_statistics,
    )


def collect_dataset_statistics(
    dataset_root: Path,
) -> tuple[SplitStatistics, ...]:
    """Collect statistics for train, validation, and test splits."""
    return tuple(
        collect_split_statistics(dataset_root, split)
        for split in ("train", "val", "test")
    )
