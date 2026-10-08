"""Route candidates by classifier probability: auto-reject, agent review, or auto-accept."""

from __future__ import annotations

from typing import Literal

Route = Literal["reject", "review", "accept"]


def route(prob: float, low: float, high: float) -> Route:
    """``prob < low`` -> reject, ``prob >= high`` -> accept, otherwise -> review."""
    # TODO: consider routing on extra signals (view agreement, tracking distance).
    if prob < low:
        return "reject"
    if prob >= high:
        return "accept"
    return "review"
