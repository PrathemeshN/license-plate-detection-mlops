"""Inference service layer."""

from pathlib import Path

from app.inference import InferenceEngine
from app.schemas.inference import (
    BoundingBoxResponse,
    DetectionResponse,
    PredictionResponse,
)


class InferenceService:
    """Coordinate model inference and API response conversion."""

    def __init__(self, engine: InferenceEngine) -> None:
        """Initialize the inference service."""
        self.engine = engine

    def predict(self, image_path: Path) -> PredictionResponse:
        """Run inference and convert detections into API responses."""
        detections = self.engine.predict(image_path)

        return PredictionResponse(
            count=len(detections),
            detections=[
                DetectionResponse(
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    bbox=BoundingBoxResponse(
                        x1=detection.bbox.x1,
                        y1=detection.bbox.y1,
                        x2=detection.bbox.x2,
                        y2=detection.bbox.y2,
                    ),
                    review_required=detection.review_required,
                )
                for detection in detections
            ],
        )
