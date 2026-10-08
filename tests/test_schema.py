import pandas as pd
import pytest

from ctgame.schema import SchemaError, check_tasks_known, validate_logs, validate_tasks


def test_valid_log_is_returned_with_clean_types(tiny_logs):
    out = validate_logs(tiny_logs.astype({"correct": str}))
    assert out["correct"].dtype.kind == "i"
    assert out["time_sec"].dtype.kind == "f"
    assert len(out) == len(tiny_logs)


def test_validation_does_not_modify_the_input(tiny_logs):
    before = tiny_logs.copy()
    validate_logs(tiny_logs)
    pd.testing.assert_frame_equal(tiny_logs, before)


def test_extra_columns_are_kept(tiny_logs):
    assert "session" in validate_logs(tiny_logs.assign(session=1)).columns


def test_missing_columns_are_all_reported(tiny_logs):
    with pytest.raises(SchemaError) as err:
        validate_logs(tiny_logs.drop(columns=["correct", "time_sec"]))
    assert len(err.value.problems) == 2


def test_empty_log_is_rejected(tiny_logs):
    with pytest.raises(SchemaError, match="no rows"):
        validate_logs(tiny_logs.iloc[0:0])


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("correct", 2, "0 or 1"),
        ("correct", "yes", "non-numeric"),
        ("time_sec", 0, "positive"),
        ("hints_used", -1, "whole numbers"),
        ("hints_used", 1.5, "whole numbers"),
    ],
)
def test_bad_values_are_rejected(tiny_logs, column, value, message):
    bad = tiny_logs.astype({column: object})
    bad.loc[0, column] = value
    with pytest.raises(SchemaError, match=message):
        validate_logs(bad)


def test_task_catalogue_rejects_duplicates_and_bad_difficulty(tiny_tasks):
    bad = pd.concat([tiny_tasks, tiny_tasks.iloc[[0]]], ignore_index=True)
    bad["difficulty"] = bad["difficulty"].astype(object)
    bad.loc[1, "difficulty"] = "hard"
    with pytest.raises(SchemaError) as err:
        validate_tasks(bad)
    assert len(err.value.problems) == 2


def test_task_catalogue_requires_columns(tiny_tasks):
    with pytest.raises(SchemaError, match="skill"):
        validate_tasks(tiny_tasks.drop(columns=["skill"]))


def test_unknown_tasks_are_reported(tiny_logs, tiny_tasks):
    check_tasks_known(tiny_logs, tiny_tasks)
    with pytest.raises(SchemaError, match="not in the catalogue: T3"):
        check_tasks_known(tiny_logs, tiny_tasks.iloc[:2])
