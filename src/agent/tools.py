"""Tools the review agent may call. Placeholder: tracking lookups are not wired up yet."""

from __future__ import annotations

from typing import Any

SUBMIT_VERDICT = "submit_verdict"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_tracking_context",
        "description": (
            "Get player-tracking data (positions, speeds, distance between the two players) "
            "for a game_play around a given step. Use it to judge whether contact is physically plausible."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "game_play": {"type": "string"},
                "step": {"type": "integer"},
                "player_1": {"type": "string"},
                "player_2": {"type": "string"},
                "window": {"type": "integer", "description": "Steps before/after to include."},
            },
            "required": ["game_play", "step", "player_1", "player_2", "window"],
            "additionalProperties": False,
        },
    },
    {
        "name": SUBMIT_VERDICT,
        "description": "Submit the final verdict for this candidate. Call exactly once, when done.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": ["impact", "no_impact", "uncertain"]},
                "confidence": {"type": "number"},
                "rationale": {"type": "string"},
            },
            "required": ["decision", "confidence", "rationale"],
            "additionalProperties": False,
        },
    },
]


def run_tool(name: str, tool_input: dict[str, Any]) -> Any:
    """Execute a non-terminal tool and return a JSON-serialisable result."""
    if name == "get_tracking_context":
        # TODO: load tracking data and return per-step positions/speeds/distances.
        return {"error": "tracking data not wired up yet", "request": tool_input}
    raise ValueError(f"unknown tool {name!r}")
