"""InformationGainSampler: the public loop.

sampler = InformationGainSampler(space, tau_main=..., tau_int=...)
batch = sampler.suggest(n=8)             # seed first, then EIG
sampler.observe(config, value, variance) # or m=task_count
sampler.predict(configs); sampler.compare(a, b); sampler.contrast(...)
sampler.should_stop(L, delta)
"""

from collections.abc import Callable

import numpy as np
from scipy.stats import norm

from . import noise
from .acquisition import eig as global_eig
from .acquisition import greedy_batch, rank_one_update
from .contrasts import interaction_contrast_L, level_contrast_L, targeted_eig
from .diagnostics import DiagnosticsTracker
from .encoding import Encoder
from .hypotheses import suggest_hypotheses as _suggest_hypotheses
from .paired import paired_bootstrap
from .posterior import Posterior
from .seed import next_balanced, seed_design
from .space import ExperimentSpace
from .types import (
    CellKey,
    Comparison,
    Config,
    ContrastResult,
    Diagnostics,
    HypothesisReport,
    Observation,
    Prediction,
)

# Prior defaults.
TAU_INTERCEPT = 0.25
MU_INTERCEPT = 0.45
TAU_INT_DEFAULT = 0.03
DELTA_DEFAULT = 0.02
N_SAMPLES = 10_000


class InformationGainSampler:
    """Sequential EIG sampler over a categorical experiment space.

    tau_main: per-factor prior sd, dict[str, float] or float (broadcasts).
    tau_int:  per-pair prior sd, dict[frozenset[str], float] or float.
              Pairs missing from the dict get TAU_INT_DEFAULT.
    cost_table: dict[CellKey, float]; missing cells cost 1.0. Unit is
              caller-defined (token count is a reasonable default).
    strategy: "eig" (default), "balanced" (greedy coverage, no posterior in
              selection), or "auto" (EIG while it has an edge, balanced once
              the edge is gone or diagnostics distrust the model). All
              strategies share the same seed design and the same posterior
              for scoring; they differ only in selection.
    seed:     int | None. Pins the internal RNG so the stochastic methods
              (sample_posterior, compare, paired bootstrap) return
              byte-identical results across machines/accounts for the same
              data. Selection (suggest) is already RNG-free. An explicit
              rng= argument to a method still overrides.
    """

    # auto handover: switch to balanced when the best EIG pick teaches at
    # most this fraction more than the balanced pick would.
    AUTO_EDGE = 0.05

    def __init__(
        self,
        space: ExperimentSpace,
        interactions: int = 2,
        tau_main: dict[str, float] | float = 0.15,
        tau_int: dict[frozenset, float] | float = TAU_INT_DEFAULT,
        cost_table: dict[CellKey, float] | None = None,
        alpha: float = 1.0,
        n_seed: int | None = None,
        m_default: int = noise.M_DEFAULT,
        strategy: str = "eig",
        seed: int | None = None,
        eligible_configs: list[Config] | None = None,
        candidate_variance: Callable[[Config, float], float] | None = None,
    ):
        if strategy not in ("eig", "balanced", "auto"):
            raise ValueError(f"unknown strategy {strategy!r}")
        self.strategy = strategy
        self.space = space
        all_keys = space.cells()
        allowed = set(space.key(c) for c in eligible_configs) if eligible_configs is not None else set(all_keys)
        self.eligible_keys = [key for key in all_keys if key in allowed]
        if not self.eligible_keys:
            raise ValueError("the sampler needs at least one eligible cell")
        self._eligible = allowed
        self._masked = eligible_configs is not None
        self._candidate_variance = candidate_variance
        self.encoder = Encoder(space, interactions=interactions)
        self.cost_table = cost_table or {}
        self.alpha = alpha
        # seed pins every stochastic method (sample_posterior/compare/paired)
        # for byte-identical replication across machines/accounts. Selection
        # (suggest) is already RNG-free. An explicit rng= arg still overrides.
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        # n_seed=None derives a grid-appropriate size that keeps the >= 2
        # appearances-per-level guarantee (seed.default_n_seed); 12 on 5x4x5.
        self._seed_plan = seed_design(space, n_seed, self.eligible_keys if self._masked else None)
        self.n_seed = len(self._seed_plan)
        self.m_default = m_default

        # Build the diagonal prior from per-factor / per-pair scales.
        prior_var = np.empty(self.encoder.p)
        prior_var[0] = TAU_INTERCEPT**2
        for factor, cols in self.encoder.main_idx.items():
            tau = tau_main[factor] if isinstance(tau_main, dict) else tau_main
            prior_var[cols] = tau**2
        for pair, cols in self.encoder.interaction_idx.items():
            if isinstance(tau_int, dict):
                tau = tau_int.get(pair, TAU_INT_DEFAULT)
            else:
                tau = tau_int
            prior_var[cols] = tau**2
        prior_mean = np.zeros(self.encoder.p)
        prior_mean[0] = MU_INTERCEPT

        self.posterior = Posterior(prior_mean, prior_var)
        self.observations: list[Observation] = []
        self._tracker = DiagnosticsTracker()

    # -- core verbs --------------------------------------------------------

    def suggest(self, n: int = 8, target: np.ndarray | None = None, allow_repeats: bool = False) -> list[Config]:
        """The next n configurations worth running.

        Seed cells come first (deterministic coverage design); after the seed
        is exhausted, greedy EIG. `target` is an optional L matrix: rows of
        contrasts to focus the acquisition on. `allow_repeats=True` lets the
        sampler propose re-running observed cells — necessary when a
        precision threshold sits below the single-run noise floor, since
        replication is then the only way down. Greedy batching prices
        repeats correctly (each hypothetical observation shrinks the next
        one's gain).
        """
        if n < 0:
            raise ValueError("n must be nonnegative")
        observed = {self.space.key(o.config) for o in self.observations}
        picks: list[CellKey] = []

        for key in self._seed_plan:  # finish the seed before any selection
            if len(picks) == n:
                break
            if key not in observed:
                picks.append(key)

        if allow_repeats:
            remaining = [k for k in self.eligible_keys if k not in set(picks)]
        else:
            remaining = [k for k in self.eligible_keys if k not in observed and k not in set(picks)]
        n_more = min(n - len(picks), len(remaining))
        if n_more > 0 and remaining:
            if self.strategy == "balanced":
                picks.extend(self._next_balanced_keys(picks, observed, n_more))
            else:
                picks.extend(self._suggest_eig(remaining, target, n_more, picks, observed))
        return [self.space.config(k) for k in picks]

    def _next_balanced_keys(self, picks, observed, k) -> list[CellKey]:
        chosen = list(observed) + list(picks)
        out: list[CellKey] = []
        for _ in range(k):
            try:
                nxt = next_balanced(self.space, chosen, self.eligible_keys)
            except ValueError:  # coverage is exhausted; replication uses EIG
                break
            out.append(nxt)
            chosen.append(nxt)
        return out

    def _suggest_eig(self, remaining, target, k, picks, observed) -> list[CellKey]:
        cands = self.encoder.design_matrix(remaining)
        mean, _ = self.posterior.predictive(cands)
        obs_var = np.array(
            [
                self.observation_variance(self.space.config(key), float(mu))
                for key, mu in zip(remaining, mean, strict=True)
            ]
        )
        cost = np.array([self.cost_table.get(key, 1.0) for key in remaining])

        if self.strategy == "auto" and self._should_hand_over(remaining, cands, obs_var, target):
            balanced = self._next_balanced_keys(picks, observed, k)
            if len(balanced) == k:
                return balanced

        if target is None:
            idx = greedy_batch(self.posterior.cov, cands, obs_var, k, cost=cost, alpha=self.alpha)
        else:
            idx = self._greedy_targeted(target, cands, obs_var, cost, k)
        return [remaining[i] for i in idx]

    def _should_hand_over(self, remaining, cands, obs_var, target=None) -> bool:
        """auto strategy: is EIG's edge over the balanced pick gone?

        The posterior prices the handover itself: when the best candidate
        teaches at most AUTO_EDGE more than what balanced would pick anyway,
        selection no longer matters and coverage is the safer default.
        """
        observed = [self.space.key(o.config) for o in self.observations]
        try:
            bal_key = next_balanced(self.space, observed, self.eligible_keys)
        except ValueError:
            return False
        scores = (
            global_eig(self.posterior.cov, cands, obs_var)
            if target is None
            else targeted_eig(self.posterior.cov, target, cands, obs_var)
        )
        bal_score = scores[remaining.index(bal_key)] if bal_key in remaining else float(np.min(scores))
        return float(np.max(scores)) <= (1.0 + self.AUTO_EDGE) * bal_score

    def observation_variance(self, config: Config, predicted_mean: float) -> float:
        """Noise of the proposed observation, supplied by the executor when known.

        The optional callback can account for benchmark metric, task count and
        a planned replica group. It must be deterministic. No callback retains
        the original binomial approximation with m_default.
        """
        value = (
            noise.candidate_variance(predicted_mean, self.m_default)
            if self._candidate_variance is None
            else self._candidate_variance(config, predicted_mean)
        )
        if not np.isfinite(value) or value <= 0:
            raise ValueError("candidate observation variance must be finite and positive")
        # A callback may describe several independent replicas as one proposed
        # group. Its variance can legitimately be below the single-run floor.
        return float(value)

    def _greedy_targeted(self, L, cands, obs_var, cost, k) -> list[int]:
        """Greedy batch under targeted EIG (same pattern as greedy_batch)."""
        cov = self.posterior.cov
        picked: list[int] = []
        available = np.ones(len(cands), dtype=bool)
        weights = cost**self.alpha
        for _ in range(k):
            scores = targeted_eig(cov, L, cands, obs_var) / weights
            scores[~available] = -np.inf
            best = int(np.argmax(scores))
            picked.append(best)
            available[best] = False
            cov = rank_one_update(cov, cands[best], obs_var[best])
        return picked

    def observe(
        self,
        config: Config,
        value: float,
        variance: float | None = None,
        m: int | None = None,
        cost: float | None = None,
        task_outcomes=None,
    ) -> None:
        """Ingest one benchmark result and refit the posterior.

        The result is scored against the pre-update posterior first (held-out
        z-score) so diagnostics measure prediction, not fit.

        task_outcomes (optional): per-task 0/1 outcome vector. Stored only;
        the posterior update is unchanged. When two cells that share a
        benchmark both carry aligned task_outcomes, compare() switches to the
        paired bootstrap automatically.

        Raises ValueError on a non-finite value or variance, an ineligible
        configuration, or a failed refit — and in every error case the sampler
        is left exactly as it was: nothing is appended and no diagnostics are
        recorded, so one bad result cannot poison all later updates.
        """
        if not np.isfinite(value):
            raise ValueError("observation value must be finite")
        if variance is None:
            variance = noise.plugin_variance(value, m if m is not None else self.m_default)
        variance = max(variance, noise.SIGMA_MIN_SQ)
        if not np.isfinite(variance):
            raise ValueError("observation variance must be finite")

        key = self.space.key(config)
        if key not in self._eligible:
            raise ValueError("cannot observe an ineligible configuration")
        a = self.encoder.phi(key)
        pred_mean, pred_var = self.posterior.predictive(a)

        observation = Observation(
            config=dict(config), value=value, variance=variance, m=m, cost=cost, task_outcomes=task_outcomes
        )
        # Refit from the candidate stacked data BEFORE committing anything, so
        # a rejected observation leaves observations, tracker, and posterior
        # untouched. (Posterior.update only mutates after its checks and the
        # Cholesky factorization succeed.)
        pending = self.observations + [observation]
        keys = [self.space.key(o.config) for o in pending]
        self.posterior.update(
            self.encoder.design_matrix(keys),
            np.array([o.value for o in pending]),
            np.array([o.variance for o in pending]),
        )
        self._tracker.record(value, float(pred_mean[0]), float(pred_var[0]), variance, key)
        self.observations.append(observation)

    # -- posterior queries --------------------------------------------------

    def predict(self, configs: list[Config]) -> list[Prediction]:
        """Posterior mean, sd, and 90% interval for each configuration."""
        keys = [self.space.key(c) for c in configs]
        a = self.encoder.design_matrix(keys)
        mean, var = self.posterior.predictive(a)
        std = np.sqrt(var)
        return [
            Prediction(
                config=dict(c), mean=float(mu), std=float(s), ci90=(float(mu - 1.645 * s), float(mu + 1.645 * s))
            )
            for c, mu, s in zip(configs, mean, std, strict=True)
        ]

    def sample_posterior(
        self,
        configs: list[Config],
        n: int = N_SAMPLES,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Posterior samples of f at each config, shape (n, len(configs))."""
        rng = rng or self._rng
        keys = [self.space.key(c) for c in configs]
        a = self.encoder.design_matrix(keys)
        return self.posterior.sample(n, rng) @ a.T

    def _task_outcomes_for(self, config: Config):
        """Latest stored per-task outcome vector for a cell, or None."""
        key = self.space.key(config)
        found = None
        for o in self.observations:
            if self.space.key(o.config) == key and o.task_outcomes is not None:
                found = o.task_outcomes
        return found

    def compare(self, a: Config, b: Config, rng: np.random.Generator | None = None) -> Comparison:
        """P(f(a) > f(b) | D), with the shared-benchmark caveat flag.

        When a and b share a benchmark AND both cells carry index-aligned
        per-task outcomes (see observe(task_outcomes=...)), a paired bootstrap
        over tasks is used instead of the independent posterior — it keeps the
        cross-cell correlation and is calibrated where the posterior path is
        conservative.
        """
        rng = rng or self._rng
        shared = "benchmark" in self.space.factors and a.get("benchmark") == b.get("benchmark")

        if shared:
            oa, ob = self._task_outcomes_for(a), self._task_outcomes_for(b)
            if oa is not None and ob is not None and len(oa) == len(ob):
                r = paired_bootstrap(oa, ob, rng=rng)
                return Comparison(
                    mean_difference=r["mean_difference"],
                    std_difference=r["std_difference"],
                    probability_a_better=r["probability_a_better"],
                    credible_interval=r["credible_interval"],
                    shared_benchmark=True,
                    caveat="Paired bootstrap over shared tasks: keeps the "
                    "cross-cell correlation the posterior ignores.",
                    method="paired_bootstrap",
                )

        samples = self.sample_posterior([a, b], N_SAMPLES, rng)
        diff = samples[:, 0] - samples[:, 1]
        lo, hi = np.quantile(diff, [0.05, 0.95])
        caveat = (
            (
                "Both configurations share a benchmark, so their errors are "
                "positively correlated; the model treats them as independent, "
                "which usually makes this probability conservative. A paired "
                "task-level comparison is more powerful when task outcomes are "
                "available (pass observe(task_outcomes=...))."
            )
            if shared
            else ""
        )
        return Comparison(
            mean_difference=float(diff.mean()),
            std_difference=float(diff.std(ddof=1)),
            probability_a_better=float((diff > 0).mean()),
            credible_interval=(float(lo), float(hi)),
            shared_benchmark=shared,
            caveat=caveat,
            method="posterior",
        )

    def contrast(
        self,
        factor: str,
        baseline: str,
        treatment: str,
        conditional_factor: str | None = None,
    ) -> ContrastResult:
        """Posterior summary of a level contrast, with conditional children."""
        L, names = level_contrast_L(
            self.space,
            self.encoder,
            factor,
            baseline,
            treatment,
            conditional_factor,
            self.eligible_keys if self._masked else None,
        )
        mean = L @ self.posterior.mean
        var = np.einsum("ij,jk,ik->i", L, self.posterior.cov, L)
        std = np.sqrt(var)

        def result(i: int, conditional: dict) -> ContrastResult:
            return ContrastResult(
                name=names[i],
                mean=float(mean[i]),
                std=float(std[i]),
                ci90=(float(mean[i] - 1.645 * std[i]), float(mean[i] + 1.645 * std[i])),
                p_positive=float(norm.sf(0.0, loc=mean[i], scale=std[i])),
                conditional=conditional,
            )

        children = {names[i]: result(i, {}) for i in range(1, len(names))}
        return result(0, children)

    def contrast_L(self, factor, baseline, treatment, conditional_factor=None) -> np.ndarray:
        """The raw L matrix for suggest(target=...)."""
        L, _ = level_contrast_L(
            self.space,
            self.encoder,
            factor,
            baseline,
            treatment,
            conditional_factor,
            self.eligible_keys if self._masked else None,
        )
        return L

    def interaction_L(
        self,
        factor,
        baseline,
        treatment,
        conditional_factor,
        conditional_baseline,
        conditional_treatment,
    ) -> np.ndarray:
        """Linear contrast for an effect's change between two conditions."""
        return interaction_contrast_L(
            self.space,
            self.encoder,
            factor,
            baseline,
            treatment,
            conditional_factor,
            conditional_baseline,
            conditional_treatment,
            self.eligible_keys if self._masked else None,
        )

    def linear_contrast(self, L: np.ndarray, name: str = "contrast") -> ContrastResult:
        """Summarize a single linear quantity with its full joint covariance."""
        row = np.asarray(L, dtype=float).reshape(-1)
        if row.shape != (self.encoder.p,) or not np.all(np.isfinite(row)):
            raise ValueError("L must contain one finite row with encoder.p columns")
        mean = float(row @ self.posterior.mean)
        std = float(np.sqrt(max(0.0, row @ self.posterior.cov @ row)))
        p_positive = float(norm.sf(0, loc=mean, scale=std)) if std else float(mean > 0)
        return ContrastResult(name, mean, std, (mean - 1.645 * std, mean + 1.645 * std), p_positive)

    def diagnostics(self) -> Diagnostics:
        """Model-health snapshot: held-out z window, flag state, warnings."""
        return self._tracker.snapshot()

    def suggest_hypotheses(self, top_k: int = 5) -> HypothesisReport:
        """Rank exploratory hypothesis candidates from residual structure.

        Makes diagnostics() actionable: where the two-way model fits worst —
        high-residual cells, factor pairs with leftover structure, factors
        with uneven per-level fit. Exploratory candidate generation, NOT
        inference. See `hypotheses.py`.
        """
        return _suggest_hypotheses(self, top_k=top_k)

    def should_stop(self, target: np.ndarray | None = None, delta: float = DELTA_DEFAULT) -> bool:
        """True when every targeted quantity has posterior sd < delta.

        target=None means response-surface mode: all cells.
        """
        if target is None:
            target = self.encoder.design_matrix(self.eligible_keys)
        var = np.einsum("ij,jk,ik->i", target, self.posterior.cov, target)
        return bool(np.max(var) < delta**2)
