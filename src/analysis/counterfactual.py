from __future__ import annotations

from collections import Counter

import numpy as np


def counterfactual_event_shift(
    x: np.ndarray,
    score_fn,
    event_names: list[str],
    max_changes: int = 5,
) -> dict[str, object]:
    """Greedy counterfactual on bag-of-events vector.

    Removes high-contributing events first while minimizing total perturbation.
    """
    original_score = float(score_fn(x[None, :])[0])
    current = x.copy()
    changes: list[dict[str, object]] = []

    ranked = np.argsort(-current)
    for idx in ranked[: max_changes * 2]:
        if current[idx] <= 0:
            continue
        test = current.copy()
        test[idx] = max(0.0, test[idx] - 1.0)
        new_score = float(score_fn(test[None, :])[0])
        if new_score < original_score:
            changes.append(
                {
                    "event": event_names[idx],
                    "delta": float(test[idx] - current[idx]),
                    "score_before": original_score,
                    "score_after": new_score,
                }
            )
            current = test
            original_score = new_score
        if len(changes) >= max_changes:
            break

    return {
        "num_changes": len(changes),
        "changes": changes,
        "final_score": original_score,
    }


def dominant_event_drift(normal_sequences: list[list[str]], abnormal_sequences: list[list[str]]) -> list[str]:
    n_counts = Counter(e for s in normal_sequences for e in s)
    a_counts = Counter(e for s in abnormal_sequences for e in s)
    scores = []
    for e, ac in a_counts.items():
        nc = n_counts.get(e, 0)
        scores.append((e, (ac + 1) / (nc + 1)))
    return [e for e, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:15]]
