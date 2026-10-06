"""Application configuration management."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = "License Plate Detection API"
    app_environment: str = Field(
        default="development",
        alias="APP_ENV",
    )
    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
    )

    model_name: str = Field(
        default="license-plate-detector",
        alias="MODEL_NAME",
    )
    model_version: str = Field(
        default="1.0.0",
        alias="MODEL_VERSION",
    )
    model_path: str = Field(
        default="models/yolo11n-v3/best.pt",
        alias="MODEL_PATH",
    )

    confidence_threshold: float = Field(
        default=0.38,
        alias="CONFIDENCE_THRESHOLD",
        ge=0.0,
        le=1.0,
    )
    iou_threshold: float = Field(
        default=0.50,
        alias="IOU_THRESHOLD",
        ge=0.0,
        le=1.0,
    )

    max_image_size_mb: int = Field(
        default=10,
        alias="MAX_IMAGE_SIZE_MB",
        gt=0,
    )
    max_zip_size_mb: int = Field(
        default=50,
        alias="MAX_ZIP_SIZE_MB",
        gt=0,
    )
    max_bulk_files: int = Field(
        default=100,
        alias="MAX_BULK_FILES",
        gt=0,
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings.

    Returns:
        The application settings instance.
    """
    return Settings()
