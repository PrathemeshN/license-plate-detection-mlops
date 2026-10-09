"""API routes for license plate detection."""

# ruff: noqa: B008

from io import BytesIO
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile, ZipFile

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.inference import InferenceEngine
from app.schemas.inference import (
    BulkImageResult,
    BulkPredictionResponse,
    PredictionResponse,
)
from app.services.inference_service import InferenceService

router = APIRouter(prefix="/api/v1", tags=["inference"])

settings = get_settings()

engine = InferenceEngine(
    model_path=Path(settings.model_path),
    confidence_threshold=settings.confidence_threshold,
    iou_threshold=settings.iou_threshold,
)

inference_service = InferenceService(engine)

SUPPORTED_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


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


@router.post("/bulk-predict", response_model=BulkPredictionResponse)
async def bulk_predict(file: UploadFile = File(...)) -> BulkPredictionResponse:
    """Run license plate detection for supported images in an uploaded ZIP."""
    supported_zip_types = {"application/zip", "application/x-zip-compressed"}
    filename_is_zip = Path(file.filename or "").suffix.lower() == ".zip"

    if file.content_type not in supported_zip_types and not filename_is_zip:
        raise HTTPException(
            status_code=415,
            detail="Only ZIP files are supported.",
        )

    content = await file.read()

    try:
        with ZipFile(BytesIO(content)) as archive:
            image_members = [
                member
                for member in archive.infolist()
                if (
                    not member.is_dir()
                    and PurePosixPath(member.filename.replace("\\", "/")).suffix.lower()
                    in SUPPORTED_IMAGE_SUFFIXES
                )
            ]

            if not image_members:
                raise HTTPException(
                    status_code=400,
                    detail="ZIP archive contains no supported images.",
                )

            results = []
            for member in image_members:
                image_bytes = archive.read(member)
                image_name = PurePosixPath(member.filename.replace("\\", "/")).name
                suffix = PurePosixPath(image_name).suffix.lower()

                with NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
                    temporary_file.write(image_bytes)
                    temporary_path = Path(temporary_file.name)

                try:
                    prediction = inference_service.predict(temporary_path)
                finally:
                    temporary_path.unlink(missing_ok=True)

                results.append(
                    BulkImageResult(
                        filename=image_name,
                        count=prediction.count,
                        detections=prediction.detections,
                    )
                )
    except BadZipFile as error:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid ZIP archive.",
        ) from error

    return BulkPredictionResponse(count=len(results), results=results)
