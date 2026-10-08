"""Pilot evaluation: one-way ANCOVA of the post-test with the pre-test as covariate."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as sps


@dataclass(frozen=True)
class AncovaResult:
    """Output of :func:`ancova`. ``effect`` is the adjusted difference treatment - reference."""

    f_value: float
    p_value: float
    df_effect: int
    df_resid: int
    partial_eta_sq: float
    effect: float
    ci_low: float
    ci_high: float
    adjusted_means: dict[str, float]
    adjusted_ci: dict[str, tuple[float, float]]
    slopes_p_value: float  #: test of homogeneous regression slopes (group x covariate)
    normality_p_value: float  #: Shapiro-Wilk test of the residuals
    n: int

    def summary(self) -> str:
        """APA-style report line, e.g. ``F(1, 57) = 9.10, p = .004, partial eta^2 = .14``."""
        p = "< .001" if self.p_value < 0.001 else f"= {self.p_value:.3f}".replace("0.", ".", 1)
        return (
            f"F({self.df_effect}, {self.df_resid}) = {self.f_value:.2f}, p {p}, "
            f"partial eta^2 = {self.partial_eta_sq:.2f}; adjusted difference = "
            f"{self.effect:.2f} [95% CI {self.ci_low:.2f}, {self.ci_high:.2f}]"
        )

    @property
    def assumptions_met(self) -> bool:
        """True when neither assumption check is significant at the 0.05 level."""
        return self.slopes_p_value >= 0.05 and self.normality_p_value >= 0.05


def ancova(
    data: pd.DataFrame,
    outcome: str = "post",
    group: str = "group",
    covariate: str = "pre",
    reference: str | None = None,
) -> AncovaResult:
    """Compare two groups on ``outcome`` while adjusting for ``covariate``.

    Fits ``outcome ~ treatment + covariate`` by OLS (type II sums of squares). The
    reference group defaults to the first group name in alphabetical order.
    """
    missing = [c for c in (outcome, group, covariate) if c not in data.columns]
    if missing:
        raise KeyError(f"missing column(s): {', '.join(missing)}")
    df = data[[outcome, group, covariate]].dropna()
    df.columns = ["y", "g", "x"]
    df = df.assign(g=df["g"].astype(str), y=df["y"].astype(float), x=df["x"].astype(float))
    levels = sorted(df["g"].unique())
    if len(levels) != 2:
        raise ValueError(f"ANCOVA here compares exactly two groups, got {len(levels)}")
    reference = levels[0] if reference is None else reference
    if reference not in levels:
        raise ValueError(f"reference group '{reference}' not found in {levels}")
    other = levels[1] if reference == levels[0] else levels[0]
    df["t"] = (df["g"] == other).astype(int)
    if df.groupby("t").size().min() < 3:
        raise ValueError("each group needs at least 3 complete rows")

    model = smf.ols("y ~ t + x", data=df).fit()
    table = sm.stats.anova_lm(model, typ=2)
    ss_effect, ss_resid = table.loc["t", "sum_sq"], table.loc["Residual", "sum_sq"]
    ci = model.conf_int().loc["t"]

    at_mean = pd.DataFrame({"t": [0, 1], "x": [df["x"].mean()] * 2})
    frame = model.get_prediction(at_mean).summary_frame(alpha=0.05)
    names = [reference, other]
    slopes = smf.ols("y ~ t * x", data=df).fit()

    return AncovaResult(
        f_value=float(table.loc["t", "F"]),
        p_value=float(table.loc["t", "PR(>F)"]),
        df_effect=int(table.loc["t", "df"]),
        df_resid=int(table.loc["Residual", "df"]),
        partial_eta_sq=float(ss_effect / (ss_effect + ss_resid)),
        effect=float(model.params["t"]),
        ci_low=float(ci.iloc[0]),
        ci_high=float(ci.iloc[1]),
        adjusted_means={name: float(m) for name, m in zip(names, frame["mean"], strict=True)},
        adjusted_ci={
            name: (float(lo), float(hi))
            for name, lo, hi in zip(
                names, frame["mean_ci_lower"], frame["mean_ci_upper"], strict=True
            )
        },
        slopes_p_value=float(slopes.pvalues["t:x"]),
        normality_p_value=float(sps.shapiro(model.resid).pvalue),
        n=len(df),
    )
