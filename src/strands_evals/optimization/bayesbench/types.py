"""Core data types for bayesbench.

All dataclasses are frozen. Scores live on the [0, 1] pass-rate scale.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

# A configuration maps every factor name to one level name.
Config = dict[str, str]

# A cell key is the tuple of levels in `space.factor_order`.
# It is hashable, so it is the canonical dict key (Config is not hashable).
CellKey = tuple[str, ...]


@dataclass(frozen=True)
class Observation:
    """One benchmark result for one configuration."""

    config: Config
    value: float  # mean score in [0, 1]
    variance: float | None = None  # sampling variance; None -> compute from m
    m: int | None = None  # task count; used when variance is None
    cost: float | None = None  # same unit as the cost table
    # Per-task 0/1 outcomes, index-aligned across cells sharing a benchmark.
    # Enables the paired bootstrap in compare(); None -> posterior path only.
    task_outcomes: Sequence[float] | None = None


@dataclass(frozen=True)
class Prediction:
    """Posterior prediction for one configuration."""

    config: Config
    mean: float
    std: float  # sqrt(a' Sigma a), latent (no obs noise)
    ci90: tuple[float, float]  # mean +/- 1.645 * std


@dataclass(frozen=True)
class Comparison:
    """Posterior comparison of two configurations, f(a) vs f(b)."""

    mean_difference: float  # E[f(a) - f(b)]
    std_difference: float
    probability_a_better: float  # via posterior samples
    credible_interval: tuple[float, float]  # 90% on the difference
    shared_benchmark: bool  # True -> correlated-error caveat applies
    caveat: str = ""  # human-readable warning when shared_benchmark
    method: str = "posterior"  # "posterior" | "paired_bootstrap"


@dataclass(frozen=True)
class ContrastResult:
    """Posterior summary of one scientific contrast z = L theta."""

    name: str
    mean: float
    std: float
    ci90: tuple[float, float]
    p_positive: float
    # Conditional versions of the same contrast, e.g. per model. Empty is ok.
    conditional: dict[str, "ContrastResult"] = field(default_factory=dict)


@dataclass(frozen=True)
class HypothesisReport:
    """Exploratory candidate-generation output.

    NOT inference. A ranked, human-reviewable list of where the two-way model
    fits worst — candidates for the next hypothesis, each with the numbers that
    earned it the rank. Three signals:

    residual_cells: cells with the largest |standardized residual|, each a
        (config, resid_z, y_observed, y_predicted, n_obs) dict.
    pair_structure: factor pairs ranked by residual structure localized to the
        pair (eta^2 of grouped residuals), each a
        (pair, eta_sq, n_groups, n_obs) dict — high => structure the fitted
        two-ways miss (three-way, or an interaction shrunk too hard).
    factor_dispersion: factors ranked by how much per-level residual variance
        differs across levels, each a (factor, dispersion_ratio, per_level) dict
        — high => a level whose outcomes the model explains worse than others.
    messages: honesty notes (support warnings, the "exploratory not inference"
        reminder).
    """

    residual_cells: list[dict]
    pair_structure: list[dict]
    factor_dispersion: list[dict]
    messages: list[str]


@dataclass(frozen=True)
class Diagnostics:
    """Model-health snapshot from held-out z-scores."""

    n_holdout: int  # observations scored before being absorbed
    z_mean: float
    z_sq_sum: float
    flagged: bool
    boundary_cells: list[CellKey]  # cells with p_hat outside [0.05, 0.95]
    messages: list[str]
