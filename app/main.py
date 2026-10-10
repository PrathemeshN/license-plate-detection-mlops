"""FastAPI application entry point."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import model_is_ready, router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.middleware.request_size import RequestSizeLimitMiddleware

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
app.add_middleware(
    RequestSizeLimitMiddleware,
    max_bytes=settings.max_request_size_mb * 1024 * 1024,
)

app.include_router(router)
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).resolve().parents[1] / "frontend"),
    name="frontend-static",
)


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    """Serve the web frontend."""
    frontend_path = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    return FileResponse(frontend_path)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return application health status."""
    return {"status": "healthy"}


@app.get("/ready", response_model=None)
def readiness_check() -> dict[str, str] | JSONResponse:
    """Return application readiness status."""
    if not model_is_ready():
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "detail": "Inference model is not loaded.",
            },
        )
    return {"status": "ready"}
