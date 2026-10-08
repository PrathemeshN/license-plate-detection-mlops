"""FastAPI application entry point."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    """Serve the web frontend."""
    frontend_path = Path("/app/frontend/index.html")
    return FileResponse(frontend_path)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return application health status."""
    return {"status": "healthy"}


@app.get("/ready")
def readiness_check() -> dict[str, str]:
    """Return application readiness status."""
    return {"status": "ready"}
