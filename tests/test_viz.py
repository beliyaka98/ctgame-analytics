import io

import pytest
from matplotlib.figure import Figure

from ctgame import ancova, cluster_profiles, fit_nmf, skill_features
from ctgame.viz import plot_adjusted_means, plot_clusters, plot_task_factors


def _renders(fig: Figure) -> bool:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    return buffer.getvalue().startswith(b"\x89PNG")


@pytest.mark.parametrize("with_tasks", [True, False])
def test_task_factor_heatmap(study, observed, with_tasks):
    fig = plot_task_factors(fit_nmf(observed), study.tasks if with_tasks else None)
    assert _renders(fig)
    assert len(fig.axes[0].get_xticklabels()) == 40
    if with_tasks:  # skill names are written above the task groups
        top = fig.axes[0].child_axes[0]
        assert [t.get_text() for t in top.get_xticklabels()] == sorted(set(study.tasks["skill"]))
    else:
        assert not fig.axes[0].child_axes


def test_cluster_scatter(study):
    result = cluster_profiles(skill_features(study.logs, study.tasks))
    fig = plot_clusters(result)
    assert _renders(fig)
    assert len(fig.axes[0].collections) == result.n_clusters


def test_adjusted_means_points(pilot):
    result = ancova(pilot)
    fig = plot_adjusted_means(result)
    assert _renders(fig)
    plotted = [c.lines[0].get_ydata()[0] for c in fig.axes[0].containers]
    assert plotted == list(result.adjusted_means.values())
