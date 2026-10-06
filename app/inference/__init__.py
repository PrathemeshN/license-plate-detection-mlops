"""Inference package."""

from app.inference.engine import InferenceEngine
from app.inference.models import BoundingBox, Detection

__all__ = ["BoundingBox", "Detection", "InferenceEngine"]
