"""One-call analysis pipeline that writes figures, tables and a static HTML report."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .features import skill_features, success_matrix
from .profiles import cluster_profiles, fit_nmf
from .recommend import DEFAULT_TARGET, recommend_all
from .stats import ancova
from .viz import plot_adjusted_means, plot_clusters, plot_task_factors

#: File names that :func:`load_study` expects in a data folder.
STUDY_FILES = {"logs": "logs.csv", "tasks": "tasks.csv", "pilot": "pilot.csv"}


def load_study(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Read ``logs.csv``, ``tasks.csv`` and ``pilot.csv`` from ``data_dir``."""
    folder = Path(data_dir)
    missing = [name for name in STUDY_FILES.values() if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{folder}: missing {', '.join(missing)}")
    return {key: pd.read_csv(folder / name) for key, name in STUDY_FILES.items()}


def build_report(
    logs: pd.DataFrame,
    tasks: pd.DataFrame,
    pilot: pd.DataFrame,
    out_dir: str | Path,
    n_components: int = 3,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the whole analysis and write the results to ``out_dir``.

    Writes ``index.html``, three PNG figures, ``recommendations.csv``, ``clusters.csv``
    and ``summary.json``, and returns the summary as a dict.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    observed = success_matrix(logs)
    nmf = fit_nmf(observed, n_components=n_components, seed=seed)
    clusters = cluster_profiles(skill_features(logs, tasks), seed=seed)
    recs = recommend_all(nmf.predict(), observed, n=3, target=DEFAULT_TARGET)
    pilot_test = ancova(pilot)

    figures = {
        "task_factors.png": plot_task_factors(nmf, tasks),
        "clusters.png": plot_clusters(clusters),
        "adjusted_means.png": plot_adjusted_means(pilot_test),
    }
    for name, fig in figures.items():
        fig.savefig(out / name, dpi=150)
    recs.to_csv(out / "recommendations.csv", index=False)
    clusters.labels.to_csv(out / "clusters.csv")

    summary: dict[str, Any] = {
        "ctgame_version": __version__,
        "data": {
            "students": int(observed.shape[0]),
            "tasks": int(observed.shape[1]),
            "attempts": len(logs),
            "observed_share": round(float(observed.notna().to_numpy().mean()), 3),
        },
        "nmf": {
            "components": n_components,
            "rmse": round(nmf.rmse, 4),
            "iterations": len(nmf.rmse_history),
            "converged": nmf.converged,
        },
        "clusters": {
            "k": clusters.n_clusters,
            "silhouette": {str(k): round(v, 3) for k, v in clusters.silhouette.items()},
            "explained_variance": [round(v, 3) for v in clusters.explained_variance],
            "sizes": clusters.labels.value_counts().sort_index().tolist(),
        },
        "ancova": {
            "summary": pilot_test.summary(),
            "p_value": round(pilot_test.p_value, 4),
            "partial_eta_sq": round(pilot_test.partial_eta_sq, 3),
            "adjusted_means": {g: round(m, 2) for g, m in pilot_test.adjusted_means.items()},
            "assumptions_met": pilot_test.assumptions_met,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "index.html").write_text(_html(summary, clusters.centres, recs), encoding="utf-8")
    return summary


def _html(summary: dict[str, Any], centres: pd.DataFrame, recs: pd.DataFrame) -> str:
    data, nmf, cl, an = summary["data"], summary["nmf"], summary["clusters"], summary["ancova"]
    table = {"float_format": lambda v: f"{v:.2f}", "border": 0, "classes": "t"}
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ctgame-analytics report</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;
color:#1d1d1d;background:#fff;line-height:1.5}}
h1{{font-size:1.6rem}} h2{{font-size:1.2rem;margin-top:2rem;border-bottom:1px solid #ddd}}
img{{max-width:100%;height:auto}} .t{{border-collapse:collapse;font-size:.85rem}}
.t td,.t th{{padding:.25rem .6rem;border-bottom:1px solid #eee;text-align:right}}
.note{{color:#666;font-size:.9rem}}
</style></head><body>
<h1>ctgame-analytics — demo analysis report</h1>
<p class="note">Built automatically by the CI/CD pipeline with ctgame
{html.escape(summary["ctgame_version"])} on <b>synthetic</b> data (no real student data). Source:
<a href="https://github.com/beliyaka98/ctgame-analytics">github.com/beliyaka98/ctgame-analytics</a></p>
<h2>1. Data</h2>
<p>{data["students"]} students, {data["tasks"]} tasks, {data["attempts"]} attempts;
{data["observed_share"]:.0%} of student-task pairs were tried.</p>
<h2>2. Latent skill factors (masked NMF)</h2>
<p>{nmf["components"]} factors, RMSE on observed cells = {nmf["rmse"]},
{nmf["iterations"]} iterations, converged: {nmf["converged"]}.</p>
<img src="task_factors.png" alt="Heatmap of NMF task factors">
<h2>3. Learner profiles (PCA + K-means)</h2>
<p>Best k = {cl["k"]} by silhouette; cluster sizes {cl["sizes"]}.</p>
<img src="clusters.png" alt="Students on two principal components coloured by cluster">
{centres.to_html(**table)}
<h2>4. Pilot evaluation (ANCOVA)</h2>
<p>{html.escape(an["summary"])}. Assumptions met: {an["assumptions_met"]}.</p>
<img src="adjusted_means.png" alt="Adjusted post-test means by group">
<h2>5. Next-task recommendations (first 5 students)</h2>
{recs[recs["student_id"].isin(recs["student_id"].unique()[:5])].to_html(index=False, **table)}
<p class="note">Full tables: <a href="recommendations.csv">recommendations.csv</a>,
<a href="clusters.csv">clusters.csv</a>, <a href="summary.json">summary.json</a>.</p>
</body></html>
"""
