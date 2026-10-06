"""API routes for license plate detection."""

# ruff: noqa: B008

from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.inference import InferenceEngine
from app.schemas.inference import PredictionResponse
from app.services.inference_service import InferenceService

router = APIRouter(prefix="/api/v1", tags=["inference"])

settings = get_settings()

engine = InferenceEngine(
    model_path=Path(settings.model_path),
    confidence_threshold=settings.confidence_threshold,
    iou_threshold=settings.iou_threshold,
)

inference_service = InferenceService(engine)


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Run license plate detection on an uploaded image."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Only image files are supported.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(content) > settings.max_image_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Uploaded image exceeds the maximum allowed size.",
        )

    suffix = Path(file.filename or "image.jpg").suffix or ".jpg"

    with NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
        temporary_file.write(content)
        temporary_path = Path(temporary_file.name)

    try:
        return inference_service.predict(temporary_path)
    finally:
        temporary_path.unlink(missing_ok=True)
