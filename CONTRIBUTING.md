# Contributing

Thank you for helping. The project is a single-author research tool, so the process is small but strict:
nothing reaches `main` without passing CI.

## Workflow (GitHub Flow)

1. Open or pick an **issue** that describes the change.
2. Create a short-lived branch from `main`: `feat/<topic>`, `fix/<topic>` or `docs/<topic>`.
3. Commit in small steps using [Conventional Commits](https://www.conventionalcommits.org/):
   `feat: ...`, `fix: ...`, `test: ...`, `docs: ...`, `ci: ...`, `refactor: ...`, `chore: ...`.
4. Push and open a **pull request** that says `Closes #<issue>`. Fill in the checklist.
5. Merge only when every CI job is green.

## Local checks

```bash
python -m pip install -e ".[dev]"
ruff check . && ruff format --check .
mypy
pytest --cov            # coverage must stay >= 90%
```

## Rules for scientific code

- Every random step takes a `seed` argument, so every result can be reproduced.
- A new method needs a test that recovers a **known** answer: simulated ground truth, a hand-made
  example or a cross-check against a reference library.
- Never commit real student data. Use pseudonymous IDs and the simulator.
- Public functions have type hints and a docstring that explains *why*, not only *what*.

## Releasing

1. Update `version` in `pyproject.toml` and add a section to `CHANGELOG.md`.
2. Merge to `main`, then tag: `git tag -a vX.Y.Z -m "vX.Y.Z" && git push origin vX.Y.Z`.
3. The `Release` workflow tests, builds and publishes the GitHub release.
