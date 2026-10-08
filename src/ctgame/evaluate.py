"""Hold-out evaluation of the success predictions behind the recommender."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .profiles import fit_nmf


@dataclass(frozen=True)
class HoldoutResult:
    """Prediction error on hidden cells for NMF and for the task-mean baseline."""

    n_components: int
    rmse_nmf: float
    rmse_baseline: float
    n_test: int  #: number of hidden student-task cells

    @property
    def improvement(self) -> float:
        """Relative RMSE reduction of NMF over the baseline (0.1 = 10% lower error)."""
        return 1.0 - self.rmse_nmf / self.rmse_baseline


def evaluate_holdout(
    observed: pd.DataFrame,
    n_components: int = 3,
    test_share: float = 0.2,
    seed: int = 0,
) -> HoldoutResult:
    """Hide a random share of the observed cells, fit NMF on the rest, score the hidden cells.

    The baseline predicts the mean success of each task in the training cells, which is
    what a non-personalised system would use. NMF is useful only if it beats it.
    """
    if not 0 < test_share < 1:
        raise ValueError("test_share must be between 0 and 1")
    values = observed.to_numpy(dtype=float, copy=True)
    cells = np.argwhere(~np.isnan(values))
    rng = np.random.default_rng(seed)
    test = cells[rng.random(len(cells)) < test_share]
    if len(test) == 0:
        raise ValueError("no cells were held out; use more data or a larger test_share")
    rows, cols = test[:, 0], test[:, 1]
    truth = values[rows, cols].copy()
    values[rows, cols] = np.nan
    train = pd.DataFrame(values, index=observed.index, columns=observed.columns)

    predicted = fit_nmf(train, n_components=n_components, seed=seed).predict().to_numpy()
    task_mean = train.mean(axis=0).fillna(float(np.nanmean(values))).to_numpy()

    def rmse(guess: np.ndarray) -> float:
        return float(np.sqrt(np.mean((guess - truth) ** 2)))

    return HoldoutResult(
        n_components=n_components,
        rmse_nmf=rmse(predicted[rows, cols]),
        rmse_baseline=rmse(task_mean[cols]),
        n_test=len(test),
    )


def compare_components(
    observed: pd.DataFrame,
    components: Iterable[int] = range(1, 6),
    test_share: float = 0.2,
    seed: int = 0,
) -> pd.DataFrame:
    """Hold-out RMSE for several numbers of NMF factors (same split for all of them)."""
    rows = []
    for k in components:
        result = evaluate_holdout(observed, n_components=k, test_share=test_share, seed=seed)
        rows.append(
            {
                "n_components": k,
                "rmse_nmf": result.rmse_nmf,
                "rmse_baseline": result.rmse_baseline,
                "improvement": result.improvement,
            }
        )
    return pd.DataFrame(rows)
