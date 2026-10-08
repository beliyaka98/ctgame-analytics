"""Next-task recommendation from predicted success."""

from __future__ import annotations

import pandas as pd

#: Default target chance of success. About 0.7 keeps a task in the "zone of proximal
#: development": hard enough to train the skill, but still likely to be solved.
DEFAULT_TARGET = 0.7


def recommend_next(
    prediction: pd.DataFrame,
    observed: pd.DataFrame,
    student_id: str,
    n: int = 3,
    target: float = DEFAULT_TARGET,
) -> pd.DataFrame:
    """Rank tasks for one student by how close their predicted success is to ``target``.

    Tasks the student has not tried come first (``reason = "new"``). If fewer than ``n``
    new tasks are left, tried tasks with an observed success below ``target`` are added
    for review (``reason = "review"``).

    Parameters
    ----------
    prediction:
        Student x task predicted success, e.g. ``NMFResult.predict()``.
    observed:
        Student x task observed success with ``NaN`` for untried tasks
        (``features.success_matrix``).
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if not 0 < target < 1:
        raise ValueError("target must be between 0 and 1")
    if student_id not in prediction.index:
        raise KeyError(f"unknown student '{student_id}'")

    predicted = prediction.loc[student_id]
    seen = (
        observed.loc[student_id].reindex(predicted.index)
        if student_id in observed.index
        else pd.Series(float("nan"), index=predicted.index)
    )
    table = pd.DataFrame(
        {
            "predicted_success": predicted,
            "observed_success": seen,
            "distance": (predicted - target).abs(),
        }
    )
    table.index.name = "task_id"
    new = table[table["observed_success"].isna()].assign(reason="new")
    review = table[table["observed_success"] < target].assign(reason="review")
    ranked = pd.concat(
        [
            new.sort_values(["distance", "task_id"]),
            review.sort_values(["distance", "task_id"]),
        ]
    )
    return ranked.head(n).reset_index()[
        ["task_id", "reason", "predicted_success", "observed_success", "distance"]
    ]


def recommend_all(
    prediction: pd.DataFrame,
    observed: pd.DataFrame,
    n: int = 3,
    target: float = DEFAULT_TARGET,
) -> pd.DataFrame:
    """Recommendations for every student as one long table with a ``student_id`` column."""
    parts = [
        recommend_next(prediction, observed, sid, n=n, target=target).assign(student_id=sid)
        for sid in prediction.index
    ]
    out = pd.concat(parts, ignore_index=True)
    return out[["student_id", *[c for c in out.columns if c != "student_id"]]]
