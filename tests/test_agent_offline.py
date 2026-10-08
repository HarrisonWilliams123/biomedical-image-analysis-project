"""Offline tests for the review agent: a fake LLM client stands in for the API, no key needed."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_agent_review import run  # noqa: E402
from src.agent.evaluate import evaluate  # noqa: E402
from src.agent.review_agent import ReviewAgent  # noqa: E402
from src.agent.router import route  # noqa: E402
from src.agent.schemas import Candidate  # noqa: E402

CFG = {"route_low": 0.30, "route_high": 0.70, "max_turns": 6, "model": "fake-model"}


def tool_use(id_, name, input_):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=input_)


class FakeClient:
    """Replays scripted responses and records each request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        content = self.responses.pop(0)
        stop = "tool_use" if any(b.type == "tool_use" for b in content) else "end_turn"
        return SimpleNamespace(content=content, stop_reason=stop)


def candidate(cid="c1", prob=0.5):
    return Candidate(cid, "58168_003392", "Endzone", 40, "37084", "38590", prob)


def test_route_thresholds():
    assert route(0.10, 0.30, 0.70) == "reject"
    assert route(0.30, 0.30, 0.70) == "review"
    assert route(0.69, 0.30, 0.70) == "review"
    assert route(0.70, 0.30, 0.70) == "accept"


def test_agent_calls_tool_then_submits(tmp_path):
    client = FakeClient(
        [
            [tool_use("t1", "get_tracking_context",
                      {"game_play": "58168_003392", "step": 40, "player_1": "37084", "player_2": "38590", "window": 5})],
            [tool_use("t2", "submit_verdict", {"decision": "impact", "confidence": 0.8, "rationale": "close"})],
        ]
    )
    log = tmp_path / "agent_log.jsonl"
    verdict = ReviewAgent(client, "fake-model", max_turns=6, log_path=log).review(candidate())

    assert verdict.decision == "impact" and verdict.source == "agent" and verdict.turns == 2
    # History is [user, assistant(tool_use), user(tool_result), ...]; check the result is wired back.
    tool_result_msg = client.requests[1]["messages"][2]
    assert tool_result_msg["role"] == "user"
    assert tool_result_msg["content"][0]["tool_use_id"] == "t1"
    assert len(log.read_text().splitlines()) == 2


def test_agent_gives_up_after_max_turns():
    looping = [[tool_use(f"t{i}", "get_tracking_context",
                         {"game_play": "g", "step": 1, "player_1": "a", "player_2": "b", "window": 1})]
               for i in range(3)]
    verdict = ReviewAgent(FakeClient(looping), "fake-model", max_turns=3).review(candidate())
    assert verdict.decision == "uncertain" and verdict.turns == 3


def test_run_only_reviews_borderline():
    fake = FakeClient([[tool_use("t1", "submit_verdict",
                                 {"decision": "no_impact", "confidence": 0.6, "rationale": "far apart"})]])
    cands = [candidate("low", 0.1), candidate("mid", 0.5), candidate("high", 0.9)]
    verdicts = {v.candidate_id: v for v in run(cands, CFG, lambda: fake)}

    assert len(fake.requests) == 1
    assert verdicts["low"].decision == "no_impact" and verdicts["low"].source == "router"
    assert verdicts["mid"].decision == "no_impact" and verdicts["mid"].source == "agent"
    assert verdicts["high"].decision == "impact" and verdicts["high"].source == "router"
    json.dumps([v.to_dict() for v in verdicts.values()])  # serialisable for verdicts.jsonl

    scores = evaluate(verdicts.values(), {"low": 0, "mid": 1, "high": 1})
    assert scores["n"] == 3 and scores["precision"] == 1.0 and scores["recall"] == 0.5
