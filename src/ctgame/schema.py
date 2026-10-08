"""Schema and validation of game-task attempt logs and the task catalogue."""

from __future__ import annotations

import pandas as pd

#: Required columns of an attempt log (one row per attempt) and their meaning.
LOG_COLUMNS: dict[str, str] = {
    "student_id": "pseudonymous student code",
    "task_id": "game task code",
    "correct": "1 if the attempt was solved, else 0",
    "time_sec": "time spent on the attempt, in seconds (> 0)",
    "hints_used": "number of hints opened during the attempt (integer >= 0)",
}

#: Required columns of the task catalogue (one row per task) and their meaning.
TASK_COLUMNS: dict[str, str] = {
    "task_id": "game task code",
    "skill": "critical-thinking skill the task trains",
    "difficulty": "difficulty on a logit-like scale (about -2 easy ... +2 hard)",
}


class SchemaError(ValueError):
    """Raised when a table does not follow the expected schema.

    All problems are collected and reported together, so a researcher can fix a
    data file in one pass instead of re-running the tool after every error.
    """

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


def _missing_columns(table: pd.DataFrame, required: dict[str, str]) -> list[str]:
    return [
        f"missing column '{col}' ({meaning})"
        for col, meaning in required.items()
        if col not in table.columns
    ]


def _numeric(table: pd.DataFrame, columns: list[str], problems: list[str]) -> None:
    for col in columns:
        table[col] = pd.to_numeric(table[col], errors="coerce")
        n_bad = int(table[col].isna().sum())
        if n_bad:
            problems.append(f"{col}: {n_bad} empty or non-numeric value(s)")


def validate_logs(logs: pd.DataFrame) -> pd.DataFrame:
    """Check an attempt log and return a clean copy with normalised types.

    Extra columns are kept. Raises :class:`SchemaError` listing every problem found.
    """
    problems = _missing_columns(logs, LOG_COLUMNS)
    if problems:
        raise SchemaError(problems)
    if logs.empty:
        raise SchemaError(["the log has no rows"])

    out = logs.copy()
    out["student_id"] = out["student_id"].astype(str)
    out["task_id"] = out["task_id"].astype(str)
    _numeric(out, ["correct", "time_sec", "hints_used"], problems)
    if not problems:
        if not out["correct"].isin([0, 1]).all():
            problems.append("correct: values must be 0 or 1")
        if (out["time_sec"] <= 0).any():
            problems.append("time_sec: values must be positive")
        if ((out["hints_used"] < 0) | (out["hints_used"] % 1 != 0)).any():
            problems.append("hints_used: values must be whole numbers >= 0")
    if problems:
        raise SchemaError(problems)

    out["correct"] = out["correct"].astype(int)
    out["hints_used"] = out["hints_used"].astype(int)
    out["time_sec"] = out["time_sec"].astype(float)
    return out


def validate_tasks(tasks: pd.DataFrame) -> pd.DataFrame:
    """Check the task catalogue and return a clean copy with normalised types."""
    problems = _missing_columns(tasks, TASK_COLUMNS)
    if problems:
        raise SchemaError(problems)

    out = tasks.copy()
    out["task_id"] = out["task_id"].astype(str)
    out["skill"] = out["skill"].astype(str)
    _numeric(out, ["difficulty"], problems)
    duplicated = sorted(out.loc[out["task_id"].duplicated(), "task_id"].unique())
    if duplicated:
        problems.append(f"task_id: duplicated id(s) {', '.join(duplicated[:5])}")
    if problems:
        raise SchemaError(problems)
    return out


def check_tasks_known(logs: pd.DataFrame, tasks: pd.DataFrame) -> None:
    """Raise :class:`SchemaError` if the log mentions tasks absent from the catalogue."""
    unknown = sorted(set(logs["task_id"].astype(str)) - set(tasks["task_id"].astype(str)))
    if unknown:
        shown = ", ".join(unknown[:5]) + (" ..." if len(unknown) > 5 else "")
        raise SchemaError(
            [f"{len(unknown)} task id(s) in the log are not in the catalogue: {shown}"]
        )
