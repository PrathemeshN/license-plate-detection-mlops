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
    """Response returned by the prediction endpoint."""

    count: int = Field(ge=0)
    detections: list[DetectionResponse]
