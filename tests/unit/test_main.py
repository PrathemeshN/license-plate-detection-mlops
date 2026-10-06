"""Tests for the FastAPI application."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check() -> None:
    """Verify that the health endpoint returns a healthy status."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_readiness_check() -> None:
    """Verify that the readiness endpoint returns a ready status."""
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
