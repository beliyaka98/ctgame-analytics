"""Figures for the analysis results.

Functions build :class:`matplotlib.figure.Figure` objects directly instead of using
``pyplot``, so they need no display and behave the same on CI runners.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .profiles import ClusterResult, NMFResult
from .stats import AncovaResult

_COLORS = ["#2a6fb0", "#e07b39", "#3a9a5b", "#b04a8f", "#7d6bb3", "#8c8c8c"]


def plot_task_factors(nmf: NMFResult, tasks: pd.DataFrame | None = None) -> Figure:
    """Heatmap of the NMF task factors; tasks are grouped by skill when ``tasks`` is given."""
    H = nmf.task_factors
    labels = list(H.columns)
    if tasks is not None:
        skill = tasks.set_index("task_id")["skill"].reindex(labels).fillna("?")
        labels = sorted(labels, key=lambda t: (skill[t], t))
        H = H[labels]
    fig = Figure(figsize=(max(6.0, 0.18 * len(labels)), 0.5 * len(H) + 1.6), layout="constrained")
    ax = fig.subplots()
    image = ax.imshow(H.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(len(H)), H.index)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    if tasks is not None:
        groups = [skill[t] for t in labels]
        starts = [i for i in range(len(groups)) if i == 0 or groups[i] != groups[i - 1]]
        for i in starts[1:]:
            ax.axvline(i - 0.5, color="black", linewidth=0.8)
        ends = [*starts[1:], len(groups)]
        top = ax.secondary_xaxis("top")
        top.set_xticks(
            [(a + b - 1) / 2 for a, b in zip(starts, ends, strict=True)],
            [groups[a] for a in starts],
        )
        top.tick_params(length=0)
    ax.set_title("NMF task factors (loading 0-1)")
    fig.colorbar(image, ax=ax, shrink=0.8)
    return fig


def plot_clusters(result: ClusterResult) -> Figure:
    """Students on the first two principal components, coloured by cluster."""
    fig = Figure(figsize=(5.5, 4.2), layout="constrained")
    ax = fig.subplots()
    y = result.coords.iloc[:, 1] if result.coords.shape[1] > 1 else np.zeros(len(result.coords))
    for k in range(result.n_clusters):
        members = (result.labels == k).to_numpy()
        ax.scatter(
            result.coords.iloc[members, 0],
            np.asarray(y)[members],
            s=18,
            color=_COLORS[k % len(_COLORS)],
            label=f"cluster {k} (n={members.sum()})",
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(
        f"PCA + K-means, k = {result.n_clusters}, "
        f"silhouette = {result.silhouette[result.n_clusters]:.2f}"
    )
    ax.legend(fontsize=8)
    return fig


def plot_adjusted_means(result: AncovaResult) -> Figure:
    """Adjusted post-test means with 95% confidence intervals.

    Drawn as points with intervals, not bars: the y-axis does not start at zero, and
    truncated bars would exaggerate the difference.
    """
    names = list(result.adjusted_means)
    means = np.array([result.adjusted_means[g] for g in names])
    ci = np.array([result.adjusted_ci[g] for g in names])
    fig = Figure(figsize=(4.2, 4.2), layout="constrained")
    ax = fig.subplots()
    for i in range(len(names)):
        ax.errorbar(
            [i],
            [means[i]],
            yerr=[[means[i] - ci[i, 0]], [ci[i, 1] - means[i]]],
            fmt="o",
            markersize=9,
            capsize=8,
            color=_COLORS[i % len(_COLORS)],
        )
        ax.annotate(
            f"{means[i]:.1f}",
            (i, means[i]),
            xytext=(12, 0),
            textcoords="offset points",
            va="center",
        )
    ax.set_xticks(range(len(names)), names)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylim(min(ci[:, 0]) - 3, max(ci[:, 1]) + 3)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylabel("adjusted post-test mean")
    ax.set_title(f"ANCOVA: p = {result.p_value:.3f}, partial eta^2 = {result.partial_eta_sq:.2f}")
    return fig
