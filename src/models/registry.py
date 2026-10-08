"""Register models by name and look them up per slot.

Usage::

    @register("my_detector")
    class MyDetector(Detector): ...

    model = get_model("detector", "my_detector")
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, TypeVar

import yaml

from src.models.base import Detector, ImpactClassifier

SLOTS: dict[str, type] = {"detector": Detector, "impact_classifier": ImpactClassifier}

_REGISTRY: dict[str, dict[str, type]] = {slot: {} for slot in SLOTS}

T = TypeVar("T", bound=type)


def register(name: str) -> Callable[[T], T]:
    """Class decorator: register a Detector or ImpactClassifier under ``name``.

    The slot is inferred from the base class.
    """

    def decorator(cls: T) -> T:
        for slot, base in SLOTS.items():
            if issubclass(cls, base):
                if name in _REGISTRY[slot]:
                    raise ValueError(f"{slot} model {name!r} is already registered")
                _REGISTRY[slot][name] = cls
                return cls
        raise TypeError(f"{cls.__name__} must subclass one of {[b.__name__ for b in SLOTS.values()]}")

    return decorator


def get_model(slot: str, name: str):
    """Instantiate the model registered as ``name`` in ``slot`` (weights not loaded)."""
    if slot not in _REGISTRY:
        raise KeyError(f"unknown slot {slot!r}; expected one of {list(_REGISTRY)}")
    if name not in _REGISTRY[slot]:
        raise KeyError(f"no {slot} registered as {name!r}; available: {sorted(_REGISTRY[slot])}")
    return _REGISTRY[slot][name]()


def get_model_from_config(slot: str, config_path: str | Path = "configs/models.yaml"):
    """Build and load the model configured for ``slot`` in configs/models.yaml."""
    cfg = yaml.safe_load(Path(config_path).read_text())[slot]
    if cfg["name"] == "TBD":
        raise RuntimeError(f"No model chosen for {slot!r} yet; edit {config_path}")
    model = get_model(slot, cfg["name"])
    model.load(cfg["weights_path"])
    return model
