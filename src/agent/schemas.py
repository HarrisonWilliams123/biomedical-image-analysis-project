"""Data contracts between the detection pipeline and the review agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

# Columns that data/processed/candidates.parquet must contain.
CANDIDATE_COLUMNS = ["candidate_id", "game_play", "view", "step", "player_1", "player_2", "prob"]

Decision = Literal["impact", "no_impact", "uncertain"]


@dataclass
class Candidate:
    """One possible helmet impact emitted by the classifier."""

    candidate_id: str
    game_play: str
    view: str
    step: int
    player_1: str
    player_2: str
    prob: float


@dataclass
class Verdict:
    """Final decision for a candidate, from the router or the agent."""

    candidate_id: str
    decision: Decision
    confidence: float
    rationale: str
    source: Literal["router", "agent"]
    turns: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
