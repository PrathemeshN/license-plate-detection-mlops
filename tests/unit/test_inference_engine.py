from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.inference.engine import InferenceEngine


def test_engine_raises_when_model_is_missing(tmp_path: Path) -> None:
    model_path = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError, match="Model not found"):
        InferenceEngine(model_path, 0.38, 0.50)


@patch("app.inference.engine.YOLO")
def test_engine_initializes_model(mock_yolo: MagicMock, tmp_path: Path) -> None:
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"model")
    engine = InferenceEngine(model_path, 0.38, 0.50)
    mock_yolo.assert_called_once_with(str(model_path))
    assert engine.confidence_threshold == 0.38
    assert engine.iou_threshold == 0.50


@patch("app.inference.engine.YOLO")
def test_predict_returns_detection(mock_yolo: MagicMock, tmp_path: Path) -> None:
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"model")
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"image")
    box = SimpleNamespace(
        xyxy=[MagicMock(tolist=MagicMock(return_value=[10.0, 20.0, 110.0, 120.0]))],
        conf=[MagicMock(item=MagicMock(return_value=0.85))],
        cls=[MagicMock(item=MagicMock(return_value=0))],
    )
    mock_model = mock_yolo.return_value
    mock_model.names = {0: "License_Plate"}
    mock_model.predict.return_value = [SimpleNamespace(boxes=[box])]
    engine = InferenceEngine(model_path, 0.38, 0.50)
    detections = engine.predict(image_path)
    assert len(detections) == 1
    detection = detections[0]
    assert detection.class_id == 0
    assert detection.class_name == "License_Plate"
    assert detection.confidence == 0.85
    assert detection.bbox.x1 == 10.0
    assert detection.bbox.y1 == 20.0
    assert detection.bbox.x2 == 110.0
    assert detection.bbox.y2 == 120.0
    assert detection.review_required is False
    mock_model.predict.assert_called_once_with(
        source=str(image_path), conf=0.38, iou=0.50, verbose=False
    )


@patch("app.inference.engine.YOLO")
def test_predict_flags_low_confidence_detection(
    mock_yolo: MagicMock, tmp_path: Path
) -> None:
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"model")
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"image")
    box = SimpleNamespace(
        xyxy=[MagicMock(tolist=MagicMock(return_value=[1.0, 2.0, 3.0, 4.0]))],
        conf=[MagicMock(item=MagicMock(return_value=0.42))],
        cls=[MagicMock(item=MagicMock(return_value=0))],
    )
    mock_model = mock_yolo.return_value
    mock_model.names = {0: "License_Plate"}
    mock_model.predict.return_value = [SimpleNamespace(boxes=[box])]
    engine = InferenceEngine(model_path, 0.38, 0.50)
    detections = engine.predict(image_path)
    assert len(detections) == 1
    assert detections[0].review_required is True


@patch("app.inference.engine.YOLO")
def test_predict_skips_result_without_boxes(
    mock_yolo: MagicMock, tmp_path: Path
) -> None:
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"model")
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"image")
    mock_model = mock_yolo.return_value
    mock_model.predict.return_value = [SimpleNamespace(boxes=None)]
    engine = InferenceEngine(model_path, 0.38, 0.50)
    assert engine.predict(image_path) == []
