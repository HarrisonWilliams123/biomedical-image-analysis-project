# src/agent: AI review agent

Re-examines borderline impact detections using player-tracking data.
The modules here are placeholders. Drop in your own implementation and keep these names:

| Module | Purpose |
|---|---|
| `schemas.py` | `Candidate`, `Verdict`, `CANDIDATE_COLUMNS` |
| `router.py` | `route(prob, low, high)` → `reject` / `review` / `accept` |
| `tools.py` | tool definitions (`TOOLS`) and `run_tool` dispatch |
| `review_agent.py` | `ReviewAgent(client, model, max_turns, log_path).review(candidate)` |
| `evaluate.py` | precision/recall/F1 against labels |

## Input: `data/processed/candidates.parquet`

| Column | Type | Meaning |
|---|---|---|
| `candidate_id` | str | unique id for the candidate |
| `game_play` | str | game/play key, e.g. `58168_003392` |
| `view` | str | camera view (e.g. `Endzone`, `Sideline`) |
| `step` | int | tracking step of the candidate |
| `player_1` | str | first player id |
| `player_2` | str | second player id (or `G` for ground) |
| `prob` | float | impact probability from the classifier, in [0, 1] |

## Routing

Settings live in `configs/agent.yaml`: `prob < route_low` is rejected, `prob >= route_high` is
accepted, and everything in between goes to the agent (up to `max_turns` model calls each).

## Running

```bash
pip install -r requirements-agent.txt
python scripts/run_agent_review.py            # writes outputs/verdicts.jsonl
python -m pytest tests/test_agent_offline.py  # no API key needed
```
