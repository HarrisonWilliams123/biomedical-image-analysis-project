"""Score verdicts against ground-truth impact labels."""

from __future__ import annotations

from typing import Iterable

from sklearn.metrics import f1_score, precision_score, recall_score

from src.agent.schemas import Verdict


def evaluate(verdicts: Iterable[Verdict], labels: dict[str, int]) -> dict[str, float]:
    """Precision/recall/F1 treating ``decision == "impact"`` as positive.

    ``labels`` maps candidate_id -> 1 (impact) / 0 (no impact); unlabeled verdicts are skipped.
    """
    # TODO: decide how "uncertain" should count, and report metrics per route/source.
    pairs = [(labels[v.candidate_id], int(v.decision == "impact")) for v in verdicts if v.candidate_id in labels]
    y_true = [t for t, _ in pairs]
    y_pred = [p for _, p in pairs]
    return {
        "n": len(pairs),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
