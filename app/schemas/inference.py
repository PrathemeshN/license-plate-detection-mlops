"""Schemas for license plate inference API."""

from pydantic import BaseModel, Field


class BoundingBoxResponse(BaseModel):
    """Bounding box coordinates for a detected object."""

    x1: float
    y1: float
    x2: float
    y2: float


class DetectionResponse(BaseModel):
    """Single license plate detection response."""

    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBoxResponse
    review_required: bool


class PredictionResponse(BaseModel):
    """Response returned by the single-image prediction endpoint."""

    count: int = Field(ge=0)
    detections: list[DetectionResponse]


class BulkImageResult(BaseModel):
    """Detection result for one image in a bulk request."""

    filename: str
    count: int = Field(ge=0)
    detections: list[DetectionResponse]


class BulkPredictionResponse(BaseModel):
    """Response returned by the bulk ZIP prediction endpoint."""

    count: int = Field(ge=0)
    results: list[BulkImageResult]
