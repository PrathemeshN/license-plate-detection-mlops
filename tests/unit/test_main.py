"""Tests for the FastAPI application."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check() -> None:
    """Verify that the health endpoint returns a healthy status."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_frontend_and_bulk_helper_are_served() -> None:
    frontend_response = client.get("/")
    helper_response = client.get("/static/bulk-results.js")

    assert frontend_response.status_code == 200
    assert "/static/bulk-results.js" in frontend_response.text
    assert helper_response.status_code == 200
    assert "buildBulkCsv" in helper_response.text


def test_readiness_check() -> None:
    """Verify that the readiness endpoint returns a ready status."""
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_readiness_check_reports_uninitialized_model() -> None:
    with patch("app.main.model_is_ready", return_value=False):
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "detail": "Inference model is not loaded.",
    }
