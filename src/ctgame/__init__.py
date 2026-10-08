"""ctgame: analytics for game-based critical-thinking tasks.

Learner profiling (masked NMF, PCA + K-means), next-task recommendation and
pilot evaluation (ANCOVA) for the logs of educational game tasks.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ctgame-analytics")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"

from .features import skill_features, success_matrix
from .schema import SchemaError, validate_logs, validate_tasks
from .simulate import SimulatedStudy, simulate_logs, simulate_pilot

__all__ = [
    "SchemaError",
    "SimulatedStudy",
    "__version__",
    "simulate_logs",
    "simulate_pilot",
    "skill_features",
    "success_matrix",
    "validate_logs",
    "validate_tasks",
]
