"""Synthetic study data with a known ground truth.

Real pilot logs hold personal data of minors and cannot be published, so the
package ships a generator that reproduces their structure. Because the true
learner profiles and the true treatment effect are known, the tests can check
that the analysis methods recover them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Core critical-thinking skills (Facione, 1990) used to label the game tasks.
SKILLS: tuple[str, ...] = ("analysis", "inference", "evaluation", "explanation")


@dataclass(frozen=True)
class SimulatedStudy:
    """Output of :func:`simulate_logs`."""

    logs: pd.DataFrame  #: one row per attempt (see ``schema.LOG_COLUMNS``)
    tasks: pd.DataFrame  #: task catalogue (see ``schema.TASK_COLUMNS``)
    students: pd.DataFrame  #: true profile and skill mastery of every student


def _profile_centres(n_profiles: int, n_skills: int) -> np.ndarray:
    """Every profile is strong in one skill, fair in the next one and weak in the rest."""
    centres = np.full((n_profiles, n_skills), -0.8)
    for p in range(n_profiles):
        centres[p, p % n_skills] = 1.4
        centres[p, (p + 1) % n_skills] = 0.5
    return centres


def simulate_logs(
    n_students: int = 120,
    n_tasks: int = 40,
    n_profiles: int = 3,
    attempt_rate: float = 0.6,
    seed: int = 42,
) -> SimulatedStudy:
    """Simulate attempt logs of students who belong to latent skill profiles.

    The chance to solve a task follows a logistic model,
    ``P(correct) = 1 / (1 + exp(-1.7 * (mastery[skill] - difficulty)))``,
    and a second attempt at the same task is slightly easier.
    """
    if n_students < 2 or n_tasks < len(SKILLS):
        raise ValueError(f"need at least 2 students and {len(SKILLS)} tasks")
    if not 1 <= n_profiles <= n_students:
        raise ValueError("n_profiles must be between 1 and n_students")
    if not 0 < attempt_rate <= 1:
        raise ValueError("attempt_rate must be in (0, 1]")

    rng = np.random.default_rng(seed)
    n_skills = len(SKILLS)

    task_skill = np.arange(n_tasks) % n_skills
    difficulty = np.round(rng.normal(0.0, 0.8, n_tasks), 2)
    tasks = pd.DataFrame(
        {
            "task_id": [f"T{t + 1:03d}" for t in range(n_tasks)],
            "skill": [SKILLS[s] for s in task_skill],
            "difficulty": difficulty,
        }
    )

    profile = rng.integers(0, n_profiles, n_students)
    ability = rng.normal(0.0, 0.3, (n_students, 1))  # general level shared by all skills
    mastery = _profile_centres(n_profiles, n_skills)[profile] + ability
    mastery += rng.normal(0.0, 0.35, mastery.shape)
    student_ids = [f"S{s + 1:03d}" for s in range(n_students)]
    students = pd.DataFrame({"student_id": student_ids, "profile": profile})
    for k, skill in enumerate(SKILLS):
        students[f"mastery_{skill}"] = mastery[:, k].round(3)

    rows = []
    for s in range(n_students):
        tried = np.flatnonzero(rng.random(n_tasks) < attempt_rate)
        if tried.size == 0:  # every student plays at least one task
            tried = rng.integers(0, n_tasks, 1)
        for t in tried:
            logit = 1.7 * (mastery[s, task_skill[t]] - difficulty[t])
            n_attempts = 1 + int(rng.random() < 0.3)
            for attempt in range(1, n_attempts + 1):
                p = 1.0 / (1.0 + np.exp(-(logit + 0.4 * (attempt - 1))))
                rows.append(
                    (
                        student_ids[s],
                        tasks["task_id"].iloc[t],
                        attempt,
                        int(rng.random() < p),
                        round(float(rng.lognormal(np.log(60) + 0.3 * difficulty[t], 0.4)), 1),
                        int(rng.poisson(2.0 * (1.0 - p))),
                    )
                )
    logs = pd.DataFrame(
        rows, columns=["student_id", "task_id", "attempt", "correct", "time_sec", "hints_used"]
    )
    return SimulatedStudy(logs=logs, tasks=tasks, students=students)


def simulate_pilot(n_per_group: int = 30, effect: float = 5.0, seed: int = 7) -> pd.DataFrame:
    """Simulate a pre-test / post-test pilot with a control and a treatment group.

    ``post = 10 + 0.85 * pre + effect * treatment + noise``; scores are on a 0-100 scale.
    """
    if n_per_group < 3:
        raise ValueError("need at least 3 students per group")
    rng = np.random.default_rng(seed)
    n = 2 * n_per_group
    group = np.repeat(["control", "treatment"], n_per_group)
    pre = np.clip(rng.normal(50.0, 10.0, n), 0, 100)
    post = 10.0 + 0.85 * pre + effect * (group == "treatment") + rng.normal(0.0, 6.0, n)
    return pd.DataFrame(
        {
            "student_id": [f"P{i + 1:03d}" for i in range(n)],
            "group": group,
            "pre": pre.round(1),
            "post": np.clip(post, 0, 100).round(1),
        }
    )
