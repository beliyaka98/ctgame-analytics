# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `Dockerfile`: a multi-stage image of the `ctgame` CLI with the library versions locked in
  `requirements/runtime.txt`, running as a non-root user.
- `Jenkinsfile`: a declarative pipeline for every commit on every branch (static checks, unit
  tests with JUnit and coverage reports, image build, the analysis run inside the image) and, on
  `main`, image push and deployment to a Kubernetes test cluster with a rollback on failure.
- `ci/jenkins/`: a local Jenkins, an image registry and a k3s cluster in Docker Compose, with
  Jenkins configured as code.
- `deploy/k8s/`: Kubernetes manifests of the test environment that serves the HTML report.

## [0.1.0] - 2026-10-08

### Added
- Validation of attempt logs and the task catalogue that lists every problem at once (`ctgame.schema`).
- Simulator of game-task logs and pre/post pilots with a known ground truth (`ctgame.simulate`).
- Student × task success matrix and per-skill features (`ctgame.features`).
- Masked NMF that ignores untried tasks, and PCA + K-means learner profiles with k chosen by
  silhouette (`ctgame.profiles`).
- Next-task recommendation by distance to a target success of 0.7 (`ctgame.recommend`).
- Hold-out evaluation of the success predictions against a task-mean baseline, and the
  `ctgame evaluate` command that compares numbers of NMF factors (`ctgame.evaluate`, #1).
- One-way ANCOVA with adjusted means, effect size and assumption checks (`ctgame.stats`).
- Figures, an HTML report and the `ctgame` command-line tool.
- CI/CD: lint, type check, tests on Linux/Windows/macOS with Python 3.11–3.14 and with the oldest
  dependencies, wheel build, demo report deployed to GitHub Pages, releases on tags.

[Unreleased]: https://github.com/beliyaka98/ctgame-analytics/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/beliyaka98/ctgame-analytics/releases/tag/v0.1.0
