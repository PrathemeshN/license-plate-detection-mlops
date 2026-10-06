"""FastAPI application entry point."""

from fastapi import FastAPI

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


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return application health status.

    Returns:
        A dictionary indicating that the application is healthy.
    """
    return {"status": "healthy"}
