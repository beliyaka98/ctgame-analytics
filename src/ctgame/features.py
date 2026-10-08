"""Turn raw attempt logs into the matrices used by the models."""

from __future__ import annotations

import pandas as pd

from .schema import check_tasks_known, validate_logs, validate_tasks


def success_matrix(logs: pd.DataFrame, min_attempts: int = 1) -> pd.DataFrame:
    """Student x task table with the share of solved attempts.

    Cells are ``NaN`` where the student made fewer than ``min_attempts`` attempts,
    so "not tried" is never confused with "failed".
    """
    if min_attempts < 1:
        raise ValueError("min_attempts must be at least 1")
    logs = validate_logs(logs)
    cells = {"index": "student_id", "columns": "task_id", "values": "correct"}
    rate = logs.pivot_table(**cells, aggfunc="mean")
    count = logs.pivot_table(**cells, aggfunc="count")
    matrix = rate.where(count >= min_attempts).astype(float)
    matrix.columns.name = "task_id"
    return matrix.sort_index().sort_index(axis=1)


def skill_features(
    logs: pd.DataFrame, tasks: pd.DataFrame, include_behaviour: bool = False
) -> pd.DataFrame:
    """Student x feature table for profiling.

    Columns: ``success_<skill>`` (share of solved attempts per skill) and, with
    ``include_behaviour=True``, ``mean_time_sec`` and ``hints_per_attempt``. Time and
    hints mostly follow task difficulty, so they are off by default: on the synthetic
    data they add noise and lower the agreement with the true profiles.
    A student who never tried a skill gets the group mean for it, so every student
    keeps a complete profile.
    """
    logs = validate_logs(logs)
    tasks = validate_tasks(tasks)
    check_tasks_known(logs, tasks)

    merged = logs.merge(tasks[["task_id", "skill"]], on="task_id", how="left")
    rates = merged.pivot_table(
        index="student_id", columns="skill", values="correct", aggfunc="mean"
    )
    rates.columns = [f"success_{skill}" for skill in rates.columns]
    features = rates.astype(float)
    if include_behaviour:
        behaviour = merged.groupby("student_id").agg(
            mean_time_sec=("time_sec", "mean"),
            hints_per_attempt=("hints_used", "mean"),
        )
        features = features.join(behaviour)
    features.index.name = "student_id"
    return features.fillna(features.mean())
