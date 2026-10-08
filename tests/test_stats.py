import numpy as np
import pytest

from ctgame import ancova, simulate_pilot


def test_detects_a_true_effect():
    result = ancova(simulate_pilot(n_per_group=100, effect=8.0, seed=1))
    assert result.p_value < 0.001
    assert result.ci_low < 8.0 < result.ci_high
    assert result.adjusted_means["treatment"] > result.adjusted_means["control"]
    assert 0 < result.partial_eta_sq < 1


def test_no_effect_is_inside_the_interval():
    result = ancova(simulate_pilot(n_per_group=100, effect=0.0, seed=1))
    assert result.ci_low < 0 < result.ci_high


def test_effect_equals_the_least_squares_coefficient(pilot):
    """Cross-check statsmodels against a plain numpy least-squares fit."""
    design = np.column_stack(
        [
            np.ones(len(pilot)),
            (pilot["group"] == "treatment").astype(float),
            pilot["pre"],
        ]
    )
    coef, *_ = np.linalg.lstsq(design, pilot["post"].to_numpy(), rcond=None)
    result = ancova(pilot)
    assert result.effect == pytest.approx(coef[1])
    diff = result.adjusted_means["treatment"] - result.adjusted_means["control"]
    assert diff == pytest.approx(coef[1])


def test_summary_and_degrees_of_freedom(pilot):
    result = ancova(pilot)
    assert (result.df_effect, result.df_resid, result.n) == (1, 57, 60)
    assert result.summary().startswith("F(1, 57) = ")
    assert result.assumptions_met


def test_small_p_values_are_reported_as_bound():
    result = ancova(simulate_pilot(n_per_group=100, effect=8.0, seed=1))
    assert "p < .001" in result.summary()


def test_reference_group_flips_the_sign(pilot):
    a = ancova(pilot)
    b = ancova(pilot, reference="treatment")
    assert b.effect == pytest.approx(-a.effect)
    assert b.p_value == pytest.approx(a.p_value)


def test_rows_with_missing_values_are_dropped(pilot):
    holes = pilot.copy()
    holes.loc[[0, 1], "pre"] = np.nan
    assert ancova(holes).n == 58


def test_custom_column_names(pilot):
    renamed = pilot.rename(columns={"post": "score", "pre": "baseline", "group": "arm"})
    result = ancova(renamed, outcome="score", group="arm", covariate="baseline")
    assert result.effect == pytest.approx(ancova(pilot).effect)


@pytest.mark.parametrize(
    ("change", "kwargs", "error"),
    [
        (lambda d: d.drop(columns="pre"), {}, KeyError),
        (lambda d: d.assign(group=["a", "b", "c"] * 20), {}, ValueError),
        (lambda d: d, {"reference": "placebo"}, ValueError),
        (lambda d: d.iloc[:32], {}, ValueError),
    ],
)
def test_bad_input(pilot, change, kwargs, error):
    with pytest.raises(error):
        ancova(change(pilot), **kwargs)
