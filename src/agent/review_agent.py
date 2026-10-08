"""Tool-using LLM agent that reviews one borderline candidate at a time.

Placeholder loop over the Anthropic Messages API. ``client`` only needs a
``messages.create(**kwargs)`` method, so tests can pass a fake.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.agent.schemas import Candidate, Verdict
from src.agent.tools import SUBMIT_VERDICT, TOOLS, run_tool

# TODO: refine the prompt and add examples once the tracking tool is real.
SYSTEM_PROMPT = (
    "You review borderline helmet-impact detections from NFL video. "
    "Use get_tracking_context to check whether the two players were close enough and moving "
    "in a way consistent with helmet contact, then call submit_verdict once with your decision."
)


class ReviewAgent:
    def __init__(self, client: Any, model: str, max_turns: int = 6, log_path: str | Path | None = None):
        self.client = client
        self.model = model
        self.max_turns = max_turns
        self.log_path = Path(log_path) if log_path else None

    def review(self, candidate: Candidate) -> Verdict:
        """Run the tool loop until submit_verdict is called or max_turns is reached."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": f"Review this candidate:\n{json.dumps(asdict(candidate))}"}
        ]
        turn = 0
        for turn in range(1, self.max_turns + 1):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
            # Append the full content (including thinking blocks) so the history stays valid.
            messages.append({"role": "assistant", "content": response.content})
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            self._log(candidate.candidate_id, turn, response.stop_reason, tool_uses)

            for block in tool_uses:
                if block.name == SUBMIT_VERDICT:
                    return Verdict(
                        candidate_id=candidate.candidate_id,
                        decision=block.input["decision"],
                        confidence=float(block.input["confidence"]),
                        rationale=block.input["rationale"],
                        source="agent",
                        turns=turn,
                    )
            if response.stop_reason != "tool_use":
                break

            results = []
            for block in tool_uses:
                try:
                    content, is_error = json.dumps(run_tool(block.name, block.input)), False
                except Exception as exc:  # report tool failures back to the model
                    content, is_error = str(exc), True
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": content, "is_error": is_error}
                )
            messages.append({"role": "user", "content": results})

        return Verdict(
            candidate_id=candidate.candidate_id,
            decision="uncertain",
            confidence=0.0,
            rationale="agent did not submit a verdict",
            source="agent",
            turns=turn,
        )

    def _log(self, candidate_id: str, turn: int, stop_reason: str, tool_uses: list) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "candidate_id": candidate_id,
            "turn": turn,
            "stop_reason": stop_reason,
            "tool_calls": [{"name": b.name, "input": b.input} for b in tool_uses],
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
