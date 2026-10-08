"""Shared fixtures: one synthetic study and one pilot, generated once per test session."""

import pandas as pd
import pytest

from ctgame import SimulatedStudy, simulate_logs, simulate_pilot


@pytest.fixture(scope="session")
def study() -> SimulatedStudy:
    return simulate_logs(n_students=120, n_tasks=40, n_profiles=3, seed=42)


@pytest.fixture(scope="session")
def pilot() -> pd.DataFrame:
    return simulate_pilot(n_per_group=30, effect=5.0, seed=7)


@pytest.fixture
def tiny_logs() -> pd.DataFrame:
    """Hand-made log whose success matrix is easy to check by eye."""
    return pd.DataFrame(
        {
            "student_id": ["A", "A", "A", "B", "B"],
            "task_id": ["T1", "T1", "T2", "T1", "T3"],
            "correct": [0, 1, 1, 1, 0],
            "time_sec": [30.0, 25.0, 40.0, 20.0, 55.0],
            "hints_used": [1, 0, 0, 0, 2],
        }
    )


@pytest.fixture
def tiny_tasks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "task_id": ["T1", "T2", "T3"],
            "skill": ["analysis", "inference", "analysis"],
            "difficulty": [0.0, 0.5, 1.0],
        }
    )
