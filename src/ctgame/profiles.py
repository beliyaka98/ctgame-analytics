"""Learner profiling: masked NMF of the success matrix and PCA + K-means clustering."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

_EPS = 1e-10


@dataclass(frozen=True)
class NMFResult:
    """Factorisation ``X ~ W @ H`` of a student x task matrix."""

    student_factors: pd.DataFrame  #: W, students x factors
    task_factors: pd.DataFrame  #: H, factors x tasks (each row scaled to max 1)
    rmse_history: list[float] = field(repr=False)  #: RMSE on observed cells per iteration
    converged: bool

    @property
    def rmse(self) -> float:
        """Final root-mean-square error on the observed cells."""
        return self.rmse_history[-1]

    def predict(self) -> pd.DataFrame:
        """Predicted success for every student-task pair, clipped to [0, 1]."""
        values = self.student_factors.to_numpy() @ self.task_factors.to_numpy()
        return pd.DataFrame(
            np.clip(values, 0.0, 1.0),
            index=self.student_factors.index,
            columns=self.task_factors.columns,
        )


def fit_nmf(
    matrix: pd.DataFrame,
    n_components: int = 3,
    max_iter: int = 2000,
    tol: float = 1e-6,
    seed: int = 0,
) -> NMFResult:
    """Non-negative matrix factorisation that ignores missing cells.

    Uses the multiplicative updates of Lee and Seung (2001) with a mask, so only the
    observed cells are fitted. ``sklearn.decomposition.NMF`` needs a complete matrix,
    and filling the gaps with zeros would treat "not tried" as "failed".
    Stops when the relative RMSE improvement falls below ``tol``.
    """
    X = matrix.to_numpy(dtype=float)
    mask = ~np.isnan(X)
    if X.ndim != 2 or not mask.any():
        raise ValueError("the matrix has no observed values")
    if np.nanmin(X) < 0:
        raise ValueError("NMF needs non-negative values")
    if not 1 <= n_components <= min(X.shape):
        raise ValueError(f"n_components must be between 1 and {min(X.shape)}")

    observed = np.where(mask, X, 0.0)
    rng = np.random.default_rng(seed)
    scale = np.sqrt(max(observed[mask].mean(), _EPS) / n_components)
    W = rng.uniform(0.1, 1.0, (X.shape[0], n_components)) * scale
    H = rng.uniform(0.1, 1.0, (n_components, X.shape[1])) * scale

    history: list[float] = []
    converged = False
    for _ in range(max_iter):
        H *= (W.T @ observed) / (W.T @ (mask * (W @ H)) + _EPS)
        W *= (observed @ H.T) / ((mask * (W @ H)) @ H.T + _EPS)
        rmse = float(np.sqrt(np.mean((X - W @ H)[mask] ** 2)))
        if history and history[-1] - rmse <= tol * history[-1]:
            history.append(rmse)
            converged = True
            break
        history.append(rmse)

    # Scale every task-factor row to max 1 so loadings are comparable; W absorbs the scale.
    peak = H.max(axis=1, keepdims=True) + _EPS
    H, W = H / peak, W * peak.T
    names = [f"F{k + 1}" for k in range(n_components)]
    return NMFResult(
        student_factors=pd.DataFrame(W, index=matrix.index, columns=names),
        task_factors=pd.DataFrame(H, index=names, columns=matrix.columns),
        rmse_history=history,
        converged=converged,
    )


@dataclass(frozen=True)
class ClusterResult:
    """Output of :func:`cluster_profiles`."""

    labels: pd.Series  #: cluster of every student; cluster 0 is the largest
    n_clusters: int
    silhouette: dict[int, float]  #: silhouette score for every k tried
    coords: pd.DataFrame  #: first two principal components (for plots)
    explained_variance: list[float]  #: variance ratio of the components kept
    centres: pd.DataFrame  #: mean of the original features in every cluster


def cluster_profiles(
    features: pd.DataFrame,
    k_range: range = range(2, 7),
    variance: float = 0.9,
    seed: int = 0,
) -> ClusterResult:
    """Group students into profiles: standardise, PCA, then K-means.

    PCA keeps the components that explain ``variance`` of the total variance, which
    removes noise and correlated features before clustering. The number of clusters
    is the ``k`` in ``k_range`` with the best silhouette score.
    """
    if not 0 < variance <= 1:
        raise ValueError("variance must be in (0, 1]")
    ks = [k for k in k_range if 2 <= k < len(features)]
    if not ks:
        raise ValueError("k_range has no k with 2 <= k < number of students")

    scaled = StandardScaler().fit_transform(features.to_numpy(dtype=float))
    n_max = min(scaled.shape)
    pca = PCA(
        n_components=variance if variance < 1 else n_max, svd_solver="full", random_state=seed
    ).fit(scaled)
    reduced = pca.transform(scaled)

    silhouette: dict[int, float] = {}
    labels_by_k: dict[int, np.ndarray] = {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(reduced)
        labels_by_k[k] = km.labels_
        silhouette[k] = float(silhouette_score(reduced, km.labels_))
    best = max(silhouette, key=lambda k: silhouette[k])

    # Renumber clusters by size so the output does not depend on K-means' arbitrary order.
    raw = labels_by_k[best]
    order = np.argsort(-np.bincount(raw), kind="stable")
    labels = pd.Series(np.argsort(order)[raw], index=features.index, name="cluster")

    coords = PCA(n_components=min(2, n_max), random_state=seed).fit_transform(scaled)
    coords_df = pd.DataFrame(
        coords, index=features.index, columns=[f"PC{i + 1}" for i in range(coords.shape[1])]
    )
    return ClusterResult(
        labels=labels,
        n_clusters=best,
        silhouette=silhouette,
        coords=coords_df,
        explained_variance=[float(v) for v in pca.explained_variance_ratio_],
        centres=features.groupby(labels).mean(),
    )
