import numpy as np
import pandas as pd
import pytest

from ctgame import fit_nmf, recommend_all, recommend_next


@pytest.fixture
def toy():
    """Two students, four tasks; A tried T1 and T2, B tried everything."""
    prediction = pd.DataFrame(
        [[0.9, 0.4, 0.72, 0.5], [0.95, 0.6, 0.8, 0.3]],
        index=["A", "B"],
        columns=["T1", "T2", "T3", "T4"],
    )
    observed = pd.DataFrame(
        [[1.0, 0.0, np.nan, np.nan], [1.0, 0.5, 1.0, 0.0]],
        index=["A", "B"],
        columns=["T1", "T2", "T3", "T4"],
    )
    return prediction, observed


def test_new_tasks_come_first_ranked_by_distance(toy):
    table = recommend_next(*toy, "A", n=3, target=0.7)
    assert list(table["task_id"]) == ["T3", "T4", "T2"]
    assert list(table["reason"]) == ["new", "new", "review"]
    assert table["distance"].iloc[0] == pytest.approx(0.02)


def test_review_only_when_everything_was_tried(toy):
    table = recommend_next(*toy, "B", n=5, target=0.7)
    assert set(table["reason"]) == {"review"}
    assert list(table["task_id"]) == ["T2", "T4"]  # solved tasks are never repeated


def test_student_without_history_gets_new_tasks(toy):
    prediction, observed = toy
    table = recommend_next(prediction, observed.drop(index="A"), "A", n=4)
    assert set(table["reason"]) == {"new"}
    assert len(table) == 4


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [({"n": 0}, ValueError), ({"target": 1.0}, ValueError), ({"student_id": "Z"}, KeyError)],
)
def test_bad_arguments(toy, kwargs, error):
    args = {"student_id": "A", **kwargs}
    with pytest.raises(error):
        recommend_next(*toy, **args)


def test_recommend_all_on_the_study(observed):
    prediction = fit_nmf(observed, n_components=3).predict()
    table = recommend_all(prediction, observed, n=3)
    assert len(table) == 3 * len(observed)
    assert table.columns[0] == "student_id"
    assert (table["reason"] == "new").all()  # every student has untried tasks
    for sid, rows in table.groupby("student_id"):
        assert observed.loc[sid, rows["task_id"]].isna().all()
