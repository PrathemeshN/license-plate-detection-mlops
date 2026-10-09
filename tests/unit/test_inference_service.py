"""Unit tests for the inference service."""

from pathlib import Path
from unittest.mock import MagicMock

from app.inference.models import BoundingBox, Detection
from app.schemas.inference import PredictionResponse
from app.services.inference_service import InferenceService


def test_predict_converts_detection_to_response(tmp_path: Path) -> None:
    """Verify inference detections are converted to API responses."""
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"image")

    engine = MagicMock()
    engine.predict.return_value = [
        Detection(
            class_id=0,
            class_name="License_Plate",
            confidence=0.91,
            bbox=BoundingBox(
                x1=10.0,
                y1=20.0,
                x2=110.0,
                y2=120.0,
            ),
            review_required=False,
        )
    ]

    service = InferenceService(engine)

    response = service.predict(image_path)

    assert isinstance(response, PredictionResponse)
    assert response.count == 1
    assert len(response.detections) == 1

    detection = response.detections[0]

    assert detection.class_id == 0
    assert detection.class_name == "License_Plate"
    assert detection.confidence == 0.91
    assert detection.bbox.x1 == 10.0
    assert detection.bbox.y1 == 20.0
    assert detection.bbox.x2 == 110.0
    assert detection.bbox.y2 == 120.0
    assert detection.review_required is False

    engine.predict.assert_called_once_with(image_path)


def test_predict_returns_empty_response_when_no_detections(
    tmp_path: Path,
) -> None:
    """Verify an image with no detections returns an empty response."""
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"image")

    engine = MagicMock()
    engine.predict.return_value = []

    service = InferenceService(engine)

    response = service.predict(image_path)

    assert response.count == 0
    assert response.detections == []

    engine.predict.assert_called_once_with(image_path)
