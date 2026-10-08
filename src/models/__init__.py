"""Deep learning model slot: framework-agnostic interfaces plus a name registry."""

from src.models.base import Detector, ImpactClassifier
from src.models.registry import get_model, get_model_from_config, register

__all__ = ["Detector", "ImpactClassifier", "register", "get_model", "get_model_from_config"]
