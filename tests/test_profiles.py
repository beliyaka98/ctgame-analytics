import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score

from ctgame import cluster_profiles, fit_nmf, skill_features


@pytest.fixture
def low_rank() -> pd.DataFrame:
    """Exact non-negative rank-2 matrix, 30 students x 20 tasks."""
    rng = np.random.default_rng(0)
    values = rng.uniform(0, 1, (30, 2)) @ rng.uniform(0, 1, (2, 20)) / 2
    return pd.DataFrame(
        values, index=[f"S{i}" for i in range(30)], columns=[f"T{j}" for j in range(20)]
    )


def test_nmf_recovers_a_low_rank_matrix(low_rank):
    result = fit_nmf(low_rank, n_components=2, max_iter=5000, tol=1e-9)
    assert result.rmse < 0.01


def test_nmf_fills_missing_cells(low_rank):
    rng = np.random.default_rng(1)
    hidden = rng.random(low_rank.shape) < 0.3
    result = fit_nmf(low_rank.mask(hidden), n_components=2, max_iter=5000, tol=1e-9)
    error = (result.predict() - low_rank).to_numpy()[hidden]
    assert np.sqrt(np.mean(error**2)) < 0.05


def test_nmf_loss_never_increases(observed):
    history = np.array(fit_nmf(observed, n_components=3).rmse_history)
    assert np.all(np.diff(history) <= 1e-12)


def test_nmf_output_shape_and_scaling(observed):
    result = fit_nmf(observed, n_components=3)
    assert result.converged
    assert result.student_factors.shape == (120, 3)
    assert result.task_factors.shape == (3, 40)
    assert (result.student_factors.to_numpy() >= 0).all()
    np.testing.assert_allclose(result.task_factors.max(axis=1), 1.0, atol=1e-6)
    prediction = result.predict()
    assert prediction.index.equals(observed.index)
    assert ((prediction.to_numpy() >= 0) & (prediction.to_numpy() <= 1)).all()


def test_nmf_reports_when_it_did_not_converge(observed):
    result = fit_nmf(observed, max_iter=5)
    assert not result.converged
    assert len(result.rmse_history) == 5


def test_nmf_is_reproducible(observed):
    a, b = fit_nmf(observed, seed=3), fit_nmf(observed, seed=3)
    pd.testing.assert_frame_equal(a.predict(), b.predict())


def test_nmf_matches_scikit_learn_on_a_complete_matrix(observed):
    """Without missing cells the masked version must fit as well as sklearn's NMF."""
    complete = observed.fillna(observed.mean())
    ours = fit_nmf(complete, n_components=3, max_iter=5000, tol=1e-8).rmse
    model = NMF(n_components=3, init="nndsvda", max_iter=5000, tol=1e-6, random_state=0)
    W = model.fit_transform(complete.to_numpy())
    theirs = np.sqrt(np.mean((complete.to_numpy() - W @ model.components_) ** 2))
    assert ours <= theirs * 1.02


@pytest.mark.parametrize(
    ("change", "components", "message"),
    [
        (lambda m: m - 1, 2, "non-negative"),
        (lambda m: m * np.nan, 2, "no observed"),
        (lambda m: m, 0, "n_components"),
        (lambda m: m, 25, "n_components"),
    ],
)
def test_nmf_rejects_bad_input(low_rank, change, components, message):
    with pytest.raises(ValueError, match=message):
        fit_nmf(change(low_rank), n_components=components)


def test_clusters_recover_the_true_profiles(study):
    features = skill_features(study.logs, study.tasks)
    result = cluster_profiles(features, seed=0)
    truth = study.students.set_index("student_id").loc[features.index, "profile"]
    assert result.n_clusters == 3
    assert adjusted_rand_score(truth, result.labels) > 0.75


def test_cluster_output_is_tidy(study):
    features = skill_features(study.logs, study.tasks)
    result = cluster_profiles(features, k_range=range(2, 5), seed=0)
    sizes = result.labels.value_counts().sort_index().to_numpy()
    assert np.all(np.diff(sizes) <= 0)  # cluster 0 is the largest
    assert set(result.silhouette) == {2, 3, 4}
    assert list(result.coords.columns) == ["PC1", "PC2"]
    assert sum(result.explained_variance) >= 0.9
    assert result.centres.shape == (result.n_clusters, features.shape[1])


def test_variance_one_keeps_all_components(study):
    features = skill_features(study.logs, study.tasks)
    result = cluster_profiles(features, variance=1.0, k_range=range(3, 4))
    assert len(result.explained_variance) == features.shape[1]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"variance": 0}, "variance"), ({"k_range": range(200, 201)}, "k_range")],
)
def test_cluster_rejects_bad_arguments(study, kwargs, message):
    features = skill_features(study.logs, study.tasks)
    with pytest.raises(ValueError, match=message):
        cluster_profiles(features, **kwargs)
