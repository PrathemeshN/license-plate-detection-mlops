"""Inference data models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """Normalized bounding box coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Detection:
    """Represent one license plate detection."""

    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    review_required: bool
