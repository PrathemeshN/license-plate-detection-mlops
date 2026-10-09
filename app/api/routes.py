"""API routes for license plate detection."""

# ruff: noqa: B008

import logging
import stat
import zlib
from io import BytesIO
from pathlib import Path, PurePosixPath, PureWindowsPath
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
logger = logging.getLogger(__name__)

settings = get_settings()

inference_service: InferenceService | None = None

SUPPORTED_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
CHUNK_SIZE = 64 * 1024


def initialize_inference_service() -> None:
    """Load the configured model once and publish the inference service."""
    global inference_service
    engine = InferenceEngine(
        model_path=Path(settings.model_path),
        confidence_threshold=settings.confidence_threshold,
        iou_threshold=settings.iou_threshold,
    )
    inference_service = InferenceService(engine)


def model_is_ready() -> bool:
    """Return whether model initialization completed successfully."""
    return inference_service is not None


try:
    initialize_inference_service()
except Exception:
    # Keep health endpoints available so an orchestrator can observe an
    # unready process and restart it; never accept inference without a model.
    logger.exception("Inference model initialization failed")


def _service() -> InferenceService:
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Inference model is not ready.")
    return inference_service


async def _read_limited(file: UploadFile, limit: int, detail: str) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(min(CHUNK_SIZE, limit + 1 - total)):
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail=detail)
        chunks.append(chunk)
    return b"".join(chunks)


def _validate_member_path(name: str) -> str:
    normalized = name.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(name)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or any(part in {"..", ""} for part in normalized.split("/"))
        or (windows_path.drive and not windows_path.root)
    ):
        raise HTTPException(
            status_code=400,
            detail="ZIP archive contains an unsafe path.",
        )
    return normalized


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Run license plate detection on an uploaded image."""
    suffix = Path(file.filename or "image.jpg").suffix.lower() or ".jpg"
    if (
        not file.content_type
        or not file.content_type.startswith("image/")
        or suffix not in SUPPORTED_IMAGE_SUFFIXES
    ):
        raise HTTPException(
            status_code=415,
            detail="Only image files are supported.",
        )

    service = _service()
    image_limit = settings.max_image_size_mb * 1024 * 1024
    content = await _read_limited(
        file,
        image_limit,
        "Uploaded image exceeds the maximum allowed size.",
    )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    with NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
        temporary_file.write(content)
        temporary_path = Path(temporary_file.name)

    try:
        return service.predict(temporary_path)
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

    service = _service()
    zip_limit = settings.max_zip_size_mb * 1024 * 1024
    expanded_limit = settings.max_zip_expanded_size_mb * 1024 * 1024
    image_limit = settings.max_image_size_mb * 1024 * 1024
    content = await _read_limited(
        file,
        zip_limit,
        "Uploaded ZIP exceeds the maximum allowed size.",
    )

    try:
        with ZipFile(BytesIO(content)) as archive:
            members = archive.infolist()
            file_members = [member for member in members if not member.is_dir()]

            if len(members) > settings.max_bulk_files:
                raise HTTPException(
                    status_code=413,
                    detail="ZIP archive contains too many files.",
                )

            declared_expanded_size = sum(member.file_size for member in file_members)
            if declared_expanded_size > expanded_limit:
                raise HTTPException(
                    status_code=413,
                    detail="ZIP archive expands beyond the maximum allowed size.",
                )

            image_members = []
            for member in file_members:
                member_path = _validate_member_path(member.filename)
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise HTTPException(
                        status_code=400,
                        detail="ZIP archive contains an unsafe symbolic link.",
                    )
                if (
                    PurePosixPath(member_path).suffix.lower()
                    in SUPPORTED_IMAGE_SUFFIXES
                ):
                    if member.file_size > image_limit:
                        raise HTTPException(
                            status_code=413,
                            detail="ZIP image exceeds the maximum allowed size.",
                        )
                    image_members.append((member, member_path))

            if not image_members:
                raise HTTPException(
                    status_code=400,
                    detail="ZIP archive contains no supported images.",
                )

            results = []
            extracted_size = 0
            for member, member_path in image_members:
                remaining_expanded = expanded_limit - extracted_size
                with archive.open(member) as image_stream:
                    image_bytes = image_stream.read(
                        min(image_limit, remaining_expanded) + 1
                    )
                extracted_size += len(image_bytes)
                if len(image_bytes) > image_limit:
                    raise HTTPException(
                        status_code=413,
                        detail="ZIP image exceeds the maximum allowed size.",
                    )
                if extracted_size > expanded_limit:
                    raise HTTPException(
                        status_code=413,
                        detail="ZIP archive expands beyond the maximum allowed size.",
                    )
                if not image_bytes:
                    raise HTTPException(
                        status_code=400,
                        detail="ZIP archive contains an empty image.",
                    )
                image_name = PurePosixPath(member_path).name
                suffix = PurePosixPath(image_name).suffix.lower()

                with NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
                    temporary_file.write(image_bytes)
                    temporary_path = Path(temporary_file.name)

                try:
                    prediction = service.predict(temporary_path)
                finally:
                    temporary_path.unlink(missing_ok=True)

                results.append(
                    BulkImageResult(
                        filename=image_name,
                        count=prediction.count,
                        detections=prediction.detections,
                    )
                )
    except (
        BadZipFile,
        EOFError,
        OSError,
        RuntimeError,
        NotImplementedError,
        zlib.error,
    ) as error:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid ZIP archive.",
        ) from error

    return BulkPredictionResponse(count=len(results), results=results)
