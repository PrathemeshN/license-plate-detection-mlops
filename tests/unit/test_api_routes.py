"""Unit tests for API inference routes."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas.inference import (
    BoundingBoxResponse,
    DetectionResponse,
    PredictionResponse,
)

client = TestClient(app)


def test_predict_success() -> None:
    """Verify a valid image returns prediction results."""
    mock_response = PredictionResponse(
        count=1,
        detections=[
            DetectionResponse(
                class_id=0,
                class_name="License_Plate",
                confidence=0.91,
                bbox=BoundingBoxResponse(
                    x1=10.0,
                    y1=20.0,
                    x2=110.0,
                    y2=120.0,
                ),
                review_required=False,
            )
        ],
    )

    with patch(
        "app.api.routes.inference_service.predict",
        return_value=mock_response,
    ):
        response = client.post(
            "/api/v1/predict",
            files={
                "file": (
                    "plate.jpg",
                    b"fake-image-data",
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 200
    assert response.json() == mock_response.model_dump()


def test_predict_rejects_non_image() -> None:
    """Verify non-image uploads are rejected."""
    response = client.post(
        "/api/v1/predict",
        files={
            "file": (
                "test.txt",
                b"not-an-image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "Only image files are supported."


def test_predict_rejects_empty_file() -> None:
    """Verify empty image uploads are rejected."""
    response = client.post(
        "/api/v1/predict",
        files={
            "file": (
                "empty.jpg",
                b"",
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty."


def test_predict_rejects_oversized_file() -> None:
    """Verify oversized image uploads are rejected."""
    oversized_content = b"x" * (10 * 1024 * 1024 + 1)

    response = client.post(
        "/api/v1/predict",
        files={
            "file": (
                "large.jpg",
                oversized_content,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 413
    assert (
        response.json()["detail"] == "Uploaded image exceeds the maximum allowed size."
    )
