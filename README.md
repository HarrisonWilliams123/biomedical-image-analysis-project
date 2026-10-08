# biomedical-image-analysis-project
Detects helmet impacts in NFL video. Borderline detections are re-examined by an AI
review agent that uses player-tracking data.

## Layout

| Path | What |
|---|---|
| `configs/models.yaml` | which model fills each slot (not chosen yet: `TBD`) |
| `configs/agent.yaml` | review-agent routing thresholds, model, turn limit, log path |
| `src/models/` | framework-agnostic `Detector` / `ImpactClassifier` interfaces + registry ([README](src/models/README.md)) |
| `src/agent/` | review agent: router, tools, agent loop, evaluation ([README](src/agent/README.md)) |
| `scripts/run_agent_review.py` | `data/processed/candidates.parquet` -> route -> agent -> `outputs/verdicts.jsonl` |
| `tests/` | offline tests (fake LLM client, no API key) |
| `data/`, `outputs/` | local only, gitignored |

## Quick start

```bash
pip install -r requirements-agent.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY, export it before running
python -m pytest tests/
python scripts/run_agent_review.py
```
