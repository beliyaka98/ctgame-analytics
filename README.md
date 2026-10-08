# ctgame-analytics

[![CI/CD](https://github.com/beliyaka98/ctgame-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/beliyaka98/ctgame-analytics/actions/workflows/ci.yml)
[![Release](https://github.com/beliyaka98/ctgame-analytics/actions/workflows/release.yml/badge.svg)](https://github.com/beliyaka98/ctgame-analytics/releases)
[![Demo report](https://img.shields.io/badge/demo-report-2a6fb0)](https://beliyaka98.github.io/ctgame-analytics/)
[![Python 3.11–3.14](https://img.shields.io/badge/python-3.11%E2%80%933.14-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Learner profiling, next-task recommendation and pilot evaluation for game-based
critical-thinking tasks.**

This package is the analysis module of the master's thesis *"Development of an algorithm for
the personalised selection of game tasks to develop schoolchildren's critical thinking based on
artificial intelligence methods"* (Astana IT University, 2025–2027). It turns the attempt
logs of educational game tasks into:

1. **latent skill factors**: non-negative matrix factorisation (NMF) of the student × task
   success matrix that ignores tasks a student never tried;
2. **learner profiles**: PCA + K-means clustering, with the number of clusters chosen by the
   silhouette score;
3. **next-task recommendations**: for each student, the untried tasks whose predicted success is
   closest to a target of 0.7;
4. **pilot evaluation**: one-way ANCOVA of post-test scores with the pre-test as covariate,
   plus checks of the ANCOVA assumptions.

Real pilot data hold personal data of minors and are never stored in this repository. The package
ships a **simulator with a known ground truth**, so every result can be reproduced and the tests
can check that the methods recover the true structure.

**Live demo report** (rebuilt by CI on every push to `main`):
<https://beliyaka98.github.io/ctgame-analytics/>

## Installation

```bash
pip install git+https://github.com/beliyaka98/ctgame-analytics.git
# or, for development
git clone https://github.com/beliyaka98/ctgame-analytics.git
cd ctgame-analytics
python -m pip install -e ".[dev]"
```

Python 3.11 or newer. Dependencies: NumPy, pandas, scikit-learn, SciPy, statsmodels, Matplotlib.
Each [GitHub release](https://github.com/beliyaka98/ctgame-analytics/releases) also has a
ready-built wheel.

## Quick start (command line)

```bash
ctgame simulate --out data                     # synthetic logs.csv, tasks.csv, pilot.csv
ctgame profile  --logs data/logs.csv --tasks data/tasks.csv --out results
ctgame recommend --logs data/logs.csv --student S001 --n 3
ctgame evaluate --logs data/logs.csv --components 1 2 3 4 5   # hold-out error per k
ctgame ancova   --data data/pilot.csv
ctgame report   --data data --out site         # everything + site/index.html
```

Example output of `ctgame ancova` on the default synthetic pilot (true effect = 5 points):

```text
F(1, 57) = 11.16, p = .001, partial eta^2 = 0.16; adjusted difference = 4.04 [95% CI 1.62, 6.45]
  adjusted mean control: 52.42
  adjusted mean treatment: 56.45
```

## Quick start (Python)

```python
import ctgame as cg

study = cg.simulate_logs(n_students=120, n_tasks=40, seed=42)
observed = cg.success_matrix(study.logs)  # NaN = task not tried

nmf = cg.fit_nmf(observed, n_components=3)  # masked NMF
print(cg.recommend_next(nmf.predict(), observed, "S001", n=3))

profiles = cg.cluster_profiles(cg.skill_features(study.logs, study.tasks))
print(profiles.n_clusters, profiles.silhouette)

result = cg.ancova(cg.simulate_pilot(effect=5.0))
print(result.summary(), result.assumptions_met)
```

## Input data

| File | One row per | Required columns |
|---|---|---|
| `logs.csv` | attempt | `student_id`, `task_id`, `correct` (0/1), `time_sec` (> 0), `hints_used` (integer ≥ 0) |
| `tasks.csv` | game task | `task_id`, `skill`, `difficulty` |
| `pilot.csv` | pilot student | `group` (two groups), `pre`, `post` (column names can be changed) |

Extra columns are allowed. Invalid files are rejected with **all** problems listed at once (exit
code 2). See [docs/data-format.md](docs/data-format.md) for details and privacy rules.

## Methods

| Step | Method | Why this choice |
|---|---|---|
| Latent factors | NMF with multiplicative updates and a mask (Lee & Seung, 2001) | Success rates are non-negative, and non-negative factors are easy to read as skill groups. The mask keeps "not tried" apart from "failed"; `sklearn.decomposition.NMF` needs a complete matrix. |
| Profiles | Standardise → PCA (90% of variance) → K-means, k by silhouette | PCA removes correlated noise. K-means is simple and fast, and the silhouette score chooses k without manual tuning. |
| Recommendation | Untried task with predicted success closest to 0.7 | Keeps tasks in the zone of proximal development: hard enough to learn from, but usually solvable. |
| Validation | Hide 20% of the observed cells, fit on the rest, compare the RMSE with a task-mean baseline | Shows that personalisation helps, and chooses the number of factors from data instead of by eye. |
| Evaluation | ANCOVA `post ~ group + pre`, type II sums of squares | Adjusts for different starting levels of the groups. It also checks equal regression slopes and normal residuals (Shapiro–Wilk). |

On the default simulated study (3 true profiles), the hold-out check picks **3 factors**. Their
prediction error is **14.8% lower** than the task-mean baseline (RMSE 0.373 vs 0.438):

```text
 n_components  rmse_nmf  rmse_baseline  improvement
            1    0.4371         0.4376       0.0010
            2    0.3901         0.4376       0.1085
            3    0.3727         0.4376       0.1482
            4    0.3918         0.4376       0.1045
            5    0.4069         0.4376       0.0701
```

## Project layout

```text
src/ctgame/         package: schema, simulate, features, profiles, recommend, evaluate, stats, viz, report, cli
tests/              pytest suite (unit, cross-checks against scikit-learn/numpy, end-to-end CLI)
requirements/       oldest supported dependency versions (tested in CI)
docs/               data format and privacy rules
.github/workflows/  ci.yml (CI + Pages deployment), release.yml (releases on tags)
```

## Development

```bash
ruff check . && ruff format --check .   # style and lint
mypy                                    # static types
pytest --cov                            # tests; fails if coverage < 90%
pre-commit install                      # optional (pip install pre-commit): ruff before each commit
```

Branches follow **GitHub Flow**: every change goes through a short-lived branch and a pull request
that must pass CI. Commit messages use [Conventional Commits](https://www.conventionalcommits.org/).
See [CONTRIBUTING.md](CONTRIBUTING.md).

## CI/CD

| Workflow | Trigger | What it does |
|---|---|---|
| [`ci.yml`](.github/workflows/ci.yml) | push to `main`, pull request | ruff + mypy → pytest on Ubuntu (Python 3.11–3.14), Windows and macOS → pytest with the **oldest** supported dependencies → build and `twine check` the wheel → install that wheel and build the demo report → **deploy to GitHub Pages** (only for `main`) |
| [`release.yml`](.github/workflows/release.yml) | tag `v*` | checks that the tag equals the package version → tests → builds the wheel and sdist → **creates a GitHub release** with notes from `CHANGELOG.md` |
| [`dependabot.yml`](.github/dependabot.yml) | monthly | opens a pull request when a GitHub Action has a new version |

## Limitations

- Results on synthetic data show that the code works. They are not findings about real students.
- The recommender uses only predicted success; it does not yet plan the order of skills.
- ANCOVA here compares exactly two groups.

## Citation

If you use this software, please cite it as described in [CITATION.cff](CITATION.cff).

## License

[MIT](LICENSE) © 2026 Akniyet Sarsenbek
