import numpy as np
import pandas as pd
import pytest

from ctgame import compare_components, evaluate_holdout


def test_nmf_beats_the_task_mean_baseline(observed):
    result = evaluate_holdout(observed, n_components=3)
    assert result.rmse_nmf < result.rmse_baseline
    assert result.improvement > 0.1
    expected = observed.notna().to_numpy().sum() * 0.2
    assert result.n_test == pytest.approx(expected, rel=0.1)


def test_the_true_number_of_profiles_predicts_best(observed):
    """The simulated study has 3 latent profiles; hold-out error should point to k = 3."""
    table = compare_components(observed, components=range(1, 6))
    assert table.loc[table["rmse_nmf"].idxmin(), "n_components"] == 3
    assert (table["rmse_baseline"] == table["rmse_baseline"].iloc[0]).all()  # same split


def test_evaluation_is_reproducible_and_leaves_the_input_alone(observed):
    before = observed.copy()
    a = evaluate_holdout(observed, seed=4)
    b = evaluate_holdout(observed, seed=4)
    assert a == b
    pd.testing.assert_frame_equal(observed, before)


@pytest.mark.parametrize("share", [0.0, 1.0])
def test_test_share_must_be_a_proper_fraction(observed, share):
    with pytest.raises(ValueError, match="test_share"):
        evaluate_holdout(observed, test_share=share)


def test_too_little_data_is_rejected():
    tiny = pd.DataFrame([[1.0, np.nan], [np.nan, 0.0]])
    with pytest.raises(ValueError, match="no cells"):
        evaluate_holdout(tiny, n_components=1, test_share=0.01)
