"""YOLO-based license plate inference engine."""

from pathlib import Path
from typing import Any

from ultralytics import YOLO  # type: ignore[attr-defined]

from app.inference.models import BoundingBox, Detection


class InferenceEngine:
    """Run license plate detection using a YOLO model."""

    def __init__(
        self, model_path: Path, confidence_threshold: float, iou_threshold: float
    ) -> None:
        """Initialize the inference engine."""
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model: Any = YOLO(str(model_path))
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def predict(self, image_path: Path) -> list[Detection]:
        """Run detection on a single image."""
        results: Any = self.model.predict(
            source=str(image_path),
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections: list[Detection] = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                coordinates = box.xyxy[0].tolist()
                confidence = float(box.conf[0].item())
                class_id = int(box.cls[0].item())
                class_name = self.model.names[class_id]

                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=BoundingBox(
                            x1=float(coordinates[0]),
                            y1=float(coordinates[1]),
                            x2=float(coordinates[2]),
                            y2=float(coordinates[3]),
                        ),
                        review_required=confidence < self.confidence_threshold + 0.10,
                    )
                )

        return detections
