"""Route candidates and send borderline ones to the review agent.

data/processed/candidates.parquet -> router -> ReviewAgent -> outputs/verdicts.jsonl

Usage:
    python scripts/run_agent_review.py [--candidates PATH] [--out PATH] [--config PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.review_agent import ReviewAgent  # noqa: E402
from src.agent.router import route  # noqa: E402
from src.agent.schemas import CANDIDATE_COLUMNS, Candidate, Verdict  # noqa: E402


def load_candidates(path: Path) -> list[Candidate]:
    df = pd.read_parquet(path)
    missing = set(CANDIDATE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return [Candidate(**row) for row in df[CANDIDATE_COLUMNS].to_dict("records")]


def run(candidates: list[Candidate], cfg: dict, client) -> list[Verdict]:
    """Route each candidate; only "review" candidates reach the agent (and the client)."""
    agent = None
    verdicts = []
    for c in candidates:
        r = route(c.prob, cfg["route_low"], cfg["route_high"])
        if r == "review":
            if agent is None:
                model = os.environ.get("REVIEW_AGENT_MODEL") or cfg["model"]
                agent = ReviewAgent(client(), model, cfg["max_turns"], cfg.get("log_path"))
            verdicts.append(agent.review(c))
        else:
            verdicts.append(
                Verdict(
                    candidate_id=c.candidate_id,
                    decision="impact" if r == "accept" else "no_impact",
                    confidence=c.prob if r == "accept" else 1 - c.prob,
                    rationale=f"auto-{r}ed by router (prob={c.prob:.3f})",
                    source="router",
                )
            )
    return verdicts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", type=Path, default=Path("data/processed/candidates.parquet"))
    parser.add_argument("--out", type=Path, default=Path("outputs/verdicts.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/agent.yaml"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())

    def make_client():
        # Imported lazily so routing-only runs don't need the SDK or a key.
        # Reads ANTHROPIC_API_KEY from the environment (export it or source .env first).
        import anthropic

        return anthropic.Anthropic()

    verdicts = run(load_candidates(args.candidates), cfg, make_client)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for v in verdicts:
            f.write(json.dumps(v.to_dict()) + "\n")
    print(f"Wrote {len(verdicts)} verdicts to {args.out}")


if __name__ == "__main__":
    main()
