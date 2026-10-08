"""FastAPI application entry point."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging

settings = get_settings()

configure_logging(settings.log_level)

app = FastAPI(
    title=settings.app_name,
    version=settings.model_version,
    description="API for license plate detection in vehicle images.",
)

app.include_router(router)


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    """Serve the PlateVision frontend."""
    return FileResponse(Path("/app/frontend/index.html"))


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return application health status."""
    return {"status": "healthy"}


@app.get("/ready")
def readiness_check() -> dict[str, str]:
    """Return application readiness status."""
    return {"status": "ready"}
