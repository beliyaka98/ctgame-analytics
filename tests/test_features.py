import numpy as np
import pytest

from ctgame import SchemaError, skill_features, success_matrix


def test_success_matrix_by_hand(tiny_logs):
    m = success_matrix(tiny_logs)
    assert list(m.index) == ["A", "B"]
    assert list(m.columns) == ["T1", "T2", "T3"]
    assert m.loc["A", "T1"] == 0.5  # one of two attempts solved
    assert m.loc["B", "T3"] == 0.0  # tried and failed ...
    assert np.isnan(m.loc["A", "T3"])  # ... is different from never tried


def test_min_attempts_hides_thin_cells(tiny_logs):
    m = success_matrix(tiny_logs, min_attempts=2)
    assert m.notna().sum().sum() == 1
    assert m.loc["A", "T1"] == 0.5


def test_min_attempts_must_be_positive(tiny_logs):
    with pytest.raises(ValueError):
        success_matrix(tiny_logs, min_attempts=0)


def test_skill_features_by_hand(tiny_logs, tiny_tasks):
    f = skill_features(tiny_logs, tiny_tasks)
    assert list(f.columns) == ["success_analysis", "success_inference"]
    assert f.loc["A", "success_analysis"] == 0.5
    assert f.loc["B", "success_analysis"] == 0.5  # T1 solved, T3 failed
    # B never tried an inference task, so B gets the mean of the students who did
    assert f.loc["B", "success_inference"] == f.loc["A", "success_inference"] == 1.0


def test_behaviour_features_are_optional(tiny_logs, tiny_tasks):
    f = skill_features(tiny_logs, tiny_tasks, include_behaviour=True)
    assert f.loc["B", "hints_per_attempt"] == 1.0
    assert f.loc["A", "mean_time_sec"] == pytest.approx(95 / 3)


def test_skill_features_need_known_tasks(tiny_logs, tiny_tasks):
    with pytest.raises(SchemaError):
        skill_features(tiny_logs, tiny_tasks.iloc[:2])


def test_matrix_of_simulated_study(study, observed):
    assert observed.shape == (120, 40)
    assert 0.5 < observed.notna().to_numpy().mean() < 0.7  # attempt_rate = 0.6
    values = observed.to_numpy()
    assert np.nanmin(values) >= 0
    assert np.nanmax(values) <= 1
