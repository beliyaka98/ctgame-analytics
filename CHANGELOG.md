# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Validation of attempt logs and the task catalogue that lists every problem at once (`ctgame.schema`).
- Simulator of game-task logs and pre/post pilots with a known ground truth (`ctgame.simulate`).
- Student × task success matrix and per-skill features (`ctgame.features`).
- Masked NMF that ignores untried tasks, and PCA + K-means learner profiles with k chosen by
  silhouette (`ctgame.profiles`).
- Next-task recommendation by distance to a target success of 0.7 (`ctgame.recommend`).
- One-way ANCOVA with adjusted means, effect size and assumption checks (`ctgame.stats`).
- Figures, an HTML report and the `ctgame` command-line tool.
- CI/CD: lint, type check, tests on Linux/Windows/macOS with Python 3.11–3.14 and with the oldest
  dependencies, wheel build, demo report deployed to GitHub Pages, releases on tags.
