"""Abstract interfaces for the pipeline's deep learning models.

These classes describe *what* a model must do, not *how*. Any framework can
sit behind them as long as inputs/outputs follow the shapes documented here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence, TypedDict


class Box(TypedDict):
    """One detection in pixel coordinates (top-left origin)."""

    x: float
    y: float
    width: float
    height: float
    score: float  # confidence in [0, 1]


# A frame is an H x W x C uint8 image array; kept as Any to stay framework-agnostic.
Frame = Any
# One candidate's stack of helmet crops over time, ordered by frame.
CropSequence = Sequence[Any]


class _Model(ABC):
    """Behaviour shared by every model slot."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Registry name of this model (matches configs/models.yaml)."""

    @abstractmethod
    def load(self, weights_path: str) -> None:
        """Load weights from ``weights_path``. Must be called before ``predict``."""


class Detector(_Model):
    """Finds helmets in video frames."""

    @abstractmethod
    def predict(self, frames: Sequence[Frame]) -> list[list[Box]]:
        """Return one list of boxes per input frame, in the same order.

        A frame with no helmets yields an empty list.
        """
        # TODO: implemented by concrete detectors registered via @register.


class ImpactClassifier(_Model):
    """Scores whether a candidate helmet crop sequence contains an impact."""

    @abstractmethod
    def predict(self, crop_sequences: Sequence[CropSequence]) -> list[float]:
        """Return an impact probability in [0, 1] for each candidate, in order."""
        # TODO: implemented by concrete classifiers registered via @register.
