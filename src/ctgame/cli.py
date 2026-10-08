"""Command-line interface: ``ctgame <command> [options]``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from . import __version__
from .evaluate import compare_components
from .features import skill_features, success_matrix
from .profiles import cluster_profiles, fit_nmf
from .recommend import DEFAULT_TARGET, recommend_next
from .report import build_report, load_study
from .schema import SchemaError
from .simulate import simulate_logs, simulate_pilot
from .stats import ancova
from .viz import plot_clusters, plot_task_factors


def _simulate(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    study = simulate_logs(
        n_students=args.students, n_tasks=args.tasks, n_profiles=args.profiles, seed=args.seed
    )
    study.logs.to_csv(out / "logs.csv", index=False)
    study.tasks.to_csv(out / "tasks.csv", index=False)
    study.students.to_csv(out / "students_truth.csv", index=False)
    simulate_pilot(n_per_group=args.pilot_n, effect=args.effect, seed=args.seed).to_csv(
        out / "pilot.csv", index=False
    )
    print(f"wrote {len(study.logs)} attempts of {args.students} students to {out}")
    return 0


def _profile(args: argparse.Namespace) -> int:
    logs, tasks = pd.read_csv(args.logs), pd.read_csv(args.tasks)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nmf = fit_nmf(success_matrix(logs), n_components=args.components, seed=args.seed)
    features = skill_features(logs, tasks, include_behaviour=args.with_behaviour)
    clusters = cluster_profiles(features, seed=args.seed)
    nmf.student_factors.to_csv(out / "student_factors.csv")
    nmf.task_factors.to_csv(out / "task_factors.csv")
    clusters.labels.to_csv(out / "clusters.csv")
    plot_task_factors(nmf, tasks).savefig(out / "task_factors.png", dpi=150)
    plot_clusters(clusters).savefig(out / "clusters.png", dpi=150)
    print(f"NMF: {args.components} factors, RMSE {nmf.rmse:.3f}")
    print(
        f"clusters: k = {clusters.n_clusters}, "
        f"silhouette {clusters.silhouette[clusters.n_clusters]:.2f}"
    )
    print(clusters.centres.round(2).to_string())
    return 0


def _recommend(args: argparse.Namespace) -> int:
    observed = success_matrix(pd.read_csv(args.logs))
    nmf = fit_nmf(observed, n_components=args.components, seed=args.seed)
    table = recommend_next(nmf.predict(), observed, args.student, n=args.n, target=args.target)
    print(table.round(3).to_string(index=False))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    observed = success_matrix(pd.read_csv(args.logs))
    table = compare_components(
        observed, components=args.components, test_share=args.test_share, seed=args.seed
    )
    print(table.round(4).to_string(index=False))
    best = table.loc[table["rmse_nmf"].idxmin()]
    print(
        f"best: {int(best['n_components'])} factors, hold-out RMSE {best['rmse_nmf']:.4f} "
        f"vs baseline {best['rmse_baseline']:.4f} ({best['improvement']:.1%} lower)"
    )
    return 0


def _ancova(args: argparse.Namespace) -> int:
    result = ancova(
        pd.read_csv(args.data), outcome=args.outcome, group=args.group, covariate=args.covariate
    )
    print(result.summary())
    for name, mean in result.adjusted_means.items():
        print(f"  adjusted mean {name}: {mean:.2f}")
    if not result.assumptions_met:
        print("  warning: an ANCOVA assumption check is significant (p < .05)")
    return 0


def _report(args: argparse.Namespace) -> int:
    summary = build_report(
        **load_study(args.data), out_dir=args.out, n_components=args.components, seed=args.seed
    )
    print(f"report written to {Path(args.out) / 'index.html'}")
    print(summary["ancova"]["summary"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser (exposed for documentation and tests)."""
    parser = argparse.ArgumentParser(prog="ctgame", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("simulate", help="write a synthetic study (logs, tasks, pilot)")
    p.add_argument("--out", default="data")
    p.add_argument("--students", type=int, default=120)
    p.add_argument("--tasks", type=int, default=40)
    p.add_argument("--profiles", type=int, default=3)
    p.add_argument("--pilot-n", type=int, default=30, help="students per pilot group")
    p.add_argument("--effect", type=float, default=5.0, help="true treatment effect, points")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=_simulate)

    p = sub.add_parser("profile", help="NMF factors and PCA + K-means learner profiles")
    p.add_argument("--logs", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--out", default="results")
    p.add_argument("--components", type=int, default=3)
    p.add_argument(
        "--with-behaviour",
        action="store_true",
        help="also cluster on mean time and hints per attempt",
    )
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_profile)

    p = sub.add_parser("recommend", help="next tasks for one student")
    p.add_argument("--logs", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--target", type=float, default=DEFAULT_TARGET)
    p.add_argument("--components", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_recommend)

    p = sub.add_parser("evaluate", help="hold-out RMSE of NMF for several numbers of factors")
    p.add_argument("--logs", required=True)
    p.add_argument("--components", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--test-share", type=float, default=0.2, help="share of cells to hide")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_evaluate)

    p = sub.add_parser("ancova", help="compare pilot groups, adjusting for the pre-test")
    p.add_argument("--data", required=True)
    p.add_argument("--outcome", default="post")
    p.add_argument("--group", default="group")
    p.add_argument("--covariate", default="pre")
    p.set_defaults(func=_ancova)

    p = sub.add_parser("report", help="run everything and write an HTML report")
    p.add_argument("--data", default="data", help="folder with logs.csv, tasks.csv, pilot.csv")
    p.add_argument("--out", default="site")
    p.add_argument("--components", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns 0 on success and 2 on invalid input data."""
    args = build_parser().parse_args(argv)
    try:
        code: int = args.func(args)
    except SchemaError as err:
        print("invalid input data:", *err.problems, sep="\n  - ", file=sys.stderr)
        return 2
    except (FileNotFoundError, KeyError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    sys.exit(main())
