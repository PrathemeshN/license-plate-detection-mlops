"""Unit tests for YOLO dataset statistics."""

from pathlib import Path

import pytest

from preprocessing.dataset_statistics import (
    _build_box_statistics,
    _read_yolo_annotations,
    collect_dataset_statistics,
    collect_split_statistics,
)


def _create_dataset(
    tmp_path: Path,
    split: str = "train",
) -> Path:
    """Create a minimal valid YOLO dataset structure."""
    image_dir = tmp_path / "images" / split
    label_dir = tmp_path / "labels" / split

    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    return tmp_path


def _create_image_and_label(
    dataset_root: Path,
    split: str,
    filename: str,
    annotations: str,
) -> None:
    """Create a test image placeholder and matching label file."""
    image_path = dataset_root / "images" / split / filename
    label_path = dataset_root / "labels" / split / f"{Path(filename).stem}.txt"

    image_path.write_bytes(b"fake-image")
    label_path.write_text(annotations, encoding="utf-8")


def test_build_box_statistics_with_boxes() -> None:
    """Calculate correct aggregate statistics for bounding boxes."""
    statistics = _build_box_statistics(
        (
            (0.20, 0.10),
            (0.40, 0.20),
            (0.60, 0.30),
        )
    )

    assert statistics.count == 3
    assert statistics.mean_width == pytest.approx(0.4)
    assert statistics.median_width == pytest.approx(0.4)
    assert statistics.min_width == pytest.approx(0.2)
    assert statistics.max_width == pytest.approx(0.6)

    assert statistics.mean_height == pytest.approx(0.2)
    assert statistics.median_height == pytest.approx(0.2)
    assert statistics.min_height == pytest.approx(0.1)
    assert statistics.max_height == pytest.approx(0.3)

    assert statistics.mean_area == pytest.approx((0.02 + 0.08 + 0.18) / 3)
    assert statistics.median_area == pytest.approx(0.08)
    assert statistics.min_area == pytest.approx(0.02)
    assert statistics.max_area == pytest.approx(0.18)


def test_build_box_statistics_empty() -> None:
    """Return zero-valued statistics when no boxes exist."""
    statistics = _build_box_statistics(())

    assert statistics.count == 0
    assert statistics.mean_width == 0.0
    assert statistics.median_width == 0.0
    assert statistics.min_width == 0.0
    assert statistics.max_width == 0.0
    assert statistics.mean_height == 0.0
    assert statistics.median_height == 0.0
    assert statistics.mean_area == 0.0
    assert statistics.median_area == 0.0


def test_read_yolo_annotations(tmp_path: Path) -> None:
    """Read valid YOLO annotations correctly."""
    annotation_file = tmp_path / "plate.txt"

    annotation_file.write_text(
        "0 0.5 0.5 0.2 0.1\n" "0 0.4 0.6 0.3 0.2\n",
        encoding="utf-8",
    )

    boxes = _read_yolo_annotations(annotation_file)

    assert boxes == [
        (pytest.approx(0.2), pytest.approx(0.1)),
        (pytest.approx(0.3), pytest.approx(0.2)),
    ]


def test_read_yolo_annotations_ignores_blank_lines(tmp_path: Path) -> None:
    """Ignore blank lines in annotation files."""
    annotation_file = tmp_path / "plate.txt"

    annotation_file.write_text(
        "\n" "0 0.5 0.5 0.2 0.1\n" "\n",
        encoding="utf-8",
    )

    boxes = _read_yolo_annotations(annotation_file)

    assert len(boxes) == 1
    assert boxes[0] == (pytest.approx(0.2), pytest.approx(0.1))


def test_read_yolo_annotations_rejects_wrong_field_count(
    tmp_path: Path,
) -> None:
    """Reject annotations that do not contain five YOLO fields."""
    annotation_file = tmp_path / "invalid.txt"

    annotation_file.write_text(
        "0 0.5 0.5 0.2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Expected 5 values"):
        _read_yolo_annotations(annotation_file)


def test_read_yolo_annotations_rejects_non_numeric_values(
    tmp_path: Path,
) -> None:
    """Reject non-numeric bounding-box values."""
    annotation_file = tmp_path / "invalid.txt"

    annotation_file.write_text(
        "0 0.5 0.5 invalid 0.1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid numeric annotation"):
        _read_yolo_annotations(annotation_file)


@pytest.mark.parametrize(
    ("width", "height"),
    [
        ("0", "0.1"),
        ("1.1", "0.1"),
        ("0.2", "0"),
        ("0.2", "1.1"),
    ],
)
def test_read_yolo_annotations_rejects_invalid_dimensions(
    tmp_path: Path,
    width: str,
    height: str,
) -> None:
    """Reject bounding boxes outside normalized YOLO dimensions."""
    annotation_file = tmp_path / "invalid.txt"

    annotation_file.write_text(
        f"0 0.5 0.5 {width} {height}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid normalized"):
        _read_yolo_annotations(annotation_file)


def test_collect_split_statistics(tmp_path: Path) -> None:
    """Collect correct statistics for a dataset split."""
    dataset_root = _create_dataset(tmp_path)

    _create_image_and_label(
        dataset_root,
        "train",
        "car_01.jpg",
        "0 0.5 0.5 0.2 0.1\n",
    )

    _create_image_and_label(
        dataset_root,
        "train",
        "car_02.jpg",
        "0 0.5 0.5 0.3 0.2\n" "0 0.4 0.4 0.1 0.1\n",
    )

    statistics = collect_split_statistics(dataset_root, "train")

    assert statistics.name == "train"
    assert statistics.image_count == 2
    assert statistics.annotation_count == 3
    assert statistics.empty_annotation_files == 0
    assert statistics.multi_plate_images == 1
    assert statistics.boxes_per_image_mean == pytest.approx(1.5)
    assert statistics.boxes_per_image_median == pytest.approx(1.5)
    assert statistics.max_boxes_per_image == 2
    assert statistics.bounding_boxes.count == 3


def test_collect_split_statistics_detects_empty_annotation(
    tmp_path: Path,
) -> None:
    """Count images with empty annotation files."""
    dataset_root = _create_dataset(tmp_path)

    _create_image_and_label(
        dataset_root,
        "train",
        "empty.jpg",
        "",
    )

    statistics = collect_split_statistics(dataset_root, "train")

    assert statistics.image_count == 1
    assert statistics.annotation_count == 0
    assert statistics.empty_annotation_files == 1
    assert statistics.multi_plate_images == 0
    assert statistics.max_boxes_per_image == 0


def test_collect_split_statistics_rejects_missing_label(
    tmp_path: Path,
) -> None:
    """Reject an image when its label file is missing."""
    dataset_root = _create_dataset(tmp_path)

    image_path = dataset_root / "images" / "train" / "missing.jpg"
    image_path.write_bytes(b"fake-image")

    with pytest.raises(FileNotFoundError, match="Missing annotation"):
        collect_split_statistics(dataset_root, "train")


def test_collect_split_statistics_rejects_missing_image_directory(
    tmp_path: Path,
) -> None:
    """Reject a split with no image directory."""
    with pytest.raises(FileNotFoundError, match="Image directory"):
        collect_split_statistics(tmp_path, "train")


def test_collect_split_statistics_rejects_missing_label_directory(
    tmp_path: Path,
) -> None:
    """Reject a split with no label directory."""
    image_directory = tmp_path / "images" / "train"
    image_directory.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Label directory"):
        collect_split_statistics(tmp_path, "train")


def test_collect_split_statistics_ignores_non_image_files(
    tmp_path: Path,
) -> None:
    """Ignore files that are not supported image formats."""
    dataset_root = _create_dataset(tmp_path)

    _create_image_and_label(
        dataset_root,
        "train",
        "car.png",
        "0 0.5 0.5 0.2 0.1\n",
    )

    (dataset_root / "images" / "train" / "README.md").write_text(
        "not an image",
        encoding="utf-8",
    )

    statistics = collect_split_statistics(dataset_root, "train")

    assert statistics.image_count == 1
    assert statistics.annotation_count == 1


def test_collect_dataset_statistics(tmp_path: Path) -> None:
    """Collect statistics for all required dataset splits."""
    for split in ("train", "val", "test"):
        _create_dataset(tmp_path, split)

        _create_image_and_label(
            tmp_path,
            split,
            f"{split}_car.jpg",
            "0 0.5 0.5 0.2 0.1\n",
        )

    statistics = collect_dataset_statistics(tmp_path)

    assert len(statistics) == 3
    assert [item.name for item in statistics] == [
        "train",
        "val",
        "test",
    ]

    for item in statistics:
        assert item.image_count == 1
        assert item.annotation_count == 1
        assert item.bounding_boxes.count == 1
