# bayesbench

Sequential Bayesian experimental design for agent benchmarking. The sampler keeps a posterior over a categorical effect model (mains + two-way interactions), selects the next benchmark runs by expected information gain, and stops when the quantities you care about are known to a chosen precision. NumPy and SciPy only (install the `optimization` extra). Execution stays outside the boundary: `suggest()` returns configurations, `observe()` ingests results, and your executor (for example a strands-evals `Experiment`) runs them.

```python
from strands_evals.optimization.bayesbench import ExperimentSpace
from strands_evals.optimization.bayesbench.sampler import InformationGainSampler

space = ExperimentSpace(factors={
    "model": ["model_a", "model_b", "model_c", "model_d", "model_e"],
    "harness": ["harness_a", "harness_b", "harness_c", "harness_d"],
    "benchmark": ["bench_a", "bench_b", "bench_c", "bench_d", "bench_e"],
})
sampler = InformationGainSampler(
    space,
    tau_main={"model": 0.08, "harness": 0.10, "benchmark": 0.15},
    tau_int={frozenset({"model", "harness"}): 0.06},
    strategy="auto",   # "balanced" (coverage), "eig", or "auto"
)

batch = sampler.suggest(n=8)                  # run these with your executor
for config, (score, m) in zip(batch, results):
    sampler.observe(config, score, m=m)       # variance from task count

sampler.predict(configs)                      # mean, std, 90% interval
sampler.compare(a, b)                         # P(f(a) > f(b) | D)
sampler.contrast("harness", "harness_a", "harness_c",
                 conditional_factor="model")  # ablation, overall + per model
sampler.diagnostics()                         # held-out z, misspec warnings
sampler.should_stop(target=L, delta=0.02)
```

## Strategies

All three share the same seed design and the same posterior for scoring; they differ only in selection. `balanced` is a deterministic greedy coverage design (no statistics in selection). `eig` maximizes expected information gain, cost-aware, greedy-batched. `auto` starts with EIG and hands over to coverage when the posterior prices EIG's edge below 5%, or is the safe fallback when diagnostics flag. Measured crossover: EIG beats balanced while the run count stays below the parameter count p, and the crossover grows with the grid.

## Caveat on `compare()`

Cells sharing a benchmark share tasks, so their errors are positively correlated; the model treats them as independent. Consequence: `compare(A, B)` probabilities across models *on the same benchmark* are miscalibrated. Counterintuitively, they are usually *conservative*, because paired comparisons on shared tasks are more powerful than independent ones.

The paired bootstrap handles this when task-level outcomes are available. Pass per-task outcomes to `observe(config, value, m=m, task_outcomes=[0,1,1,...])`; `compare(a, b)` then uses a paired bootstrap over shared tasks automatically when `a` and `b` share a benchmark, and reports `Comparison.method`. On synthetic shared-task data the independent path over-covers (90% CI coverage 1.00, conservative) while the paired path is calibrated (0.93) with a 2.7x narrower interval. The full task-level hierarchical model stays deferred.

## Layout

- `space.py` / `encoding.py` — categorical grid and dummy-coded features
- `noise.py` — binomial plug-in observation noise
- `posterior.py` — conjugate Gaussian posterior (Cholesky, no explicit inverses)
- `acquisition.py` / `contrasts.py` — global and targeted expected information gain
- `sampler.py` — `InformationGainSampler`, the public loop
- `diagnostics.py` / `hypotheses.py` — held-out misspecification checks and residual-based hypothesis candidates
- `paired.py` — paired bootstrap for within-benchmark comparisons
- `recovery.py` — interaction-recovery metric (recall/precision of significant effects)
- `ledger.py` — JSON-lines session log; replaying it rebuilds the sampler

The claims above (strategy crossover, calibration, paired-bootstrap coverage) were measured on synthetic surfaces and a real leaderboard fit before this module was brought into strands-evals. The milestone gate tests under `tests/strands_evals/optimization/bayesbench/` carry the executable parts of that record.
