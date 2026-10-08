import pandas as pd
import pytest

from ctgame import simulate_logs, simulate_pilot, validate_logs, validate_tasks
from ctgame.simulate import SKILLS


def test_study_follows_the_schema(study):
    validate_logs(study.logs)
    validate_tasks(study.tasks)
    assert set(study.tasks["skill"]) == set(SKILLS)
    assert study.students["profile"].nunique() == 3


def test_every_student_plays(study):
    assert study.logs["student_id"].nunique() == len(study.students)


def test_every_student_plays_even_with_a_low_attempt_rate():
    study = simulate_logs(n_students=30, n_tasks=4, attempt_rate=0.01, seed=0)
    assert study.logs["student_id"].nunique() == 30


def test_same_seed_gives_same_data():
    a, b = simulate_logs(n_students=20, seed=5), simulate_logs(n_students=20, seed=5)
    pd.testing.assert_frame_equal(a.logs, b.logs)


def test_different_seed_gives_different_data():
    a, b = simulate_logs(n_students=20, seed=5), simulate_logs(n_students=20, seed=6)
    assert not a.logs.equals(b.logs)


def test_stronger_students_solve_more(study):
    """Success rate must follow the simulated mastery of the task's skill."""
    merged = study.logs.merge(study.tasks, on="task_id").merge(study.students, on="student_id")
    merged["mastery"] = [row[f"mastery_{row['skill']}"] for _, row in merged.iterrows()]
    high = merged.loc[merged["mastery"] > merged["difficulty"], "correct"].mean()
    low = merged.loc[merged["mastery"] < merged["difficulty"], "correct"].mean()
    assert high > low + 0.3


@pytest.mark.parametrize(
    "kwargs",
    [{"n_students": 1}, {"n_tasks": 2}, {"n_profiles": 0}, {"attempt_rate": 0.0}],
)
def test_invalid_simulation_arguments(kwargs):
    with pytest.raises(ValueError):
        simulate_logs(**kwargs)


def test_pilot_shape_and_range():
    pilot = simulate_pilot(n_per_group=10, seed=1)
    assert len(pilot) == 20
    assert pilot["group"].value_counts().to_dict() == {"control": 10, "treatment": 10}
    assert ((pilot[["pre", "post"]] >= 0) & (pilot[["pre", "post"]] <= 100)).all().all()


def test_pilot_needs_enough_students():
    with pytest.raises(ValueError):
        simulate_pilot(n_per_group=2)
