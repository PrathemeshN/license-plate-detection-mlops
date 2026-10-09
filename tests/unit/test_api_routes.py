"""Unit tests for API inference routes."""

from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from app.api.routes import settings as route_settings
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


def _create_test_zip() -> bytes:
    """Create an in-memory ZIP containing test images."""
    buffer = BytesIO()

    with ZipFile(buffer, "w") as archive:
        archive.writestr("Cars001.jpg", b"fake-image-1")
        archive.writestr("Cars002.png", b"fake-image-2")

    return buffer.getvalue()


def test_bulk_predict_success() -> None:
    """Verify ZIP images are processed individually."""
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
            "/api/v1/bulk-predict",
            files={
                "file": (
                    "test.zip",
                    _create_test_zip(),
                    "application/zip",
                )
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["count"] == 2
    assert len(body["results"]) == 2
    assert body["results"][0]["filename"] == "Cars001.jpg"
    assert body["results"][1]["filename"] == "Cars002.png"
    assert body["results"][0]["count"] == 1
    assert body["results"][0]["detections"][0]["confidence"] == 0.91


def test_bulk_predict_rejects_non_zip() -> None:
    """Verify non-ZIP uploads are rejected."""
    response = client.post(
        "/api/v1/bulk-predict",
        files={
            "file": (
                "images.txt",
                b"not-a-zip",
                "text/plain",
            )
        },
    )

    assert response.status_code == 415


def test_bulk_predict_rejects_invalid_zip() -> None:
    """Verify invalid ZIP archives are rejected."""
    response = client.post(
        "/api/v1/bulk-predict",
        files={
            "file": (
                "broken.zip",
                b"not-a-real-zip",
                "application/zip",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == ("Uploaded file is not a valid ZIP archive.")


def test_bulk_predict_rejects_zip_without_images() -> None:
    """Verify ZIP archives without supported images are rejected."""
    buffer = BytesIO()

    with ZipFile(buffer, "w") as archive:
        archive.writestr("readme.txt", b"hello")

    response = client.post(
        "/api/v1/bulk-predict",
        files={
            "file": (
                "empty-images.zip",
                buffer.getvalue(),
                "application/zip",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == ("ZIP archive contains no supported images.")


def test_bulk_predict_rejects_too_many_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(route_settings, "max_bulk_files", 1)
    response = client.post(
        "/api/v1/bulk-predict",
        files={"file": ("many.zip", _create_test_zip(), "application/zip")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "ZIP archive contains too many files."


def test_bulk_predict_rejects_excessive_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(route_settings, "max_zip_expanded_size_mb", 1)
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("large.jpg", b"x" * (1024 * 1024 + 1))

    response = client.post(
        "/api/v1/bulk-predict",
        files={"file": ("expanded.zip", buffer.getvalue(), "application/zip")},
    )

    assert response.status_code == 413
    assert "expands" in response.json()["detail"]


def test_bulk_predict_rejects_unsafe_archive_path() -> None:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("../outside.jpg", b"image")

    response = client.post(
        "/api/v1/bulk-predict",
        files={"file": ("unsafe.zip", buffer.getvalue(), "application/zip")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "ZIP archive contains an unsafe path."


def test_bulk_predict_rejects_oversized_zip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(route_settings, "max_zip_size_mb", 1)
    response = client.post(
        "/api/v1/bulk-predict",
        files={"file": ("oversized.zip", b"x" * (1024 * 1024 + 1), "application/zip")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == (
        "Uploaded ZIP exceeds the maximum allowed size."
    )


def test_predict_rejects_unsupported_image_extension() -> None:
    response = client.post(
        "/api/v1/predict",
        files={"file": ("animation.gif", b"gif", "image/gif")},
    )

    assert response.status_code == 415


def test_predict_rejects_request_over_limit() -> None:
    response = client.post(
        "/api/v1/predict",
        files={"file": ("small.jpg", b"x", "image/jpeg")},
        headers={"content-length": str(26 * 1024 * 1024 + 1)},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == ("Request exceeds the maximum allowed size.")


def test_bulk_frontend_uses_api_response_fields() -> None:
    frontend = Path("frontend/index.html").read_text(encoding="utf-8")

    assert "Processed ${data.count} image(s)" in frontend
    assert "item.count" in frontend
    assert "detection.confidence * 100" in frontend
    assert "escapeHtml(item.filename)" in frontend
    assert "data.total_files" not in frontend
    assert "confidence_percent" not in frontend
