# Data format and privacy rules

## `logs.csv`: one row per attempt

| Column | Type | Rule |
|---|---|---|
| `student_id` | text | pseudonymous code, e.g. `S001`; never a name or an e-mail address |
| `task_id` | text | must exist in `tasks.csv` when skill features are computed |
| `correct` | 0 / 1 | 1 if the attempt solved the task |
| `time_sec` | number | > 0 |
| `hints_used` | integer | ≥ 0 |

Several attempts at the same task are allowed. The success matrix stores the **share** of solved
attempts, and `NaN` means "not tried", which is different from "failed" (0.0).

## `tasks.csv`: one row per game task

| Column | Type | Rule |
|---|---|---|
| `task_id` | text | unique |
| `skill` | text | the critical-thinking skill the task trains, e.g. `analysis`, `inference`, `evaluation`, `explanation` |
| `difficulty` | number | about −2 (easy) … +2 (hard) |

## `pilot.csv`: one row per pilot student

`group` (exactly two groups, e.g. `control` and `treatment`), `pre` and `post` (test scores).
Other column names can be passed with `--outcome`, `--group` and `--covariate`.

## Privacy

- Real pilot data are personal data of minors. They stay on the university server and are **never**
  committed. `.gitignore` blocks `data/`, `private/` and spreadsheets.
- Use pseudonymous codes; the key that links codes to names is kept separately by the school.
- Published results use aggregated or synthetic data only.
