"""Session ledger: one experiment = one JSON-lines file.

Line 1 defines the space and the sampler:
    {"type": "space", "factors": {...}, "sampler_kwargs": {...}}
Then one line per observation:
    {"type": "observation", "config": {...}, "value": ..., "variance": ...,
     "m": ..., "cost": ..., "ts": ...}
Optionally, for audit, one line per suggestion batch:
    {"type": "suggestion", "configs": [...], "ts": ...}

`load_sampler` replays the observations through `observe()`: the rebuilt
posterior is identical to the live one.

JSON has no frozenset or tuple dict keys. When you write the space record,
join such keys with "|" (for example, tau_int {"model|harness": 0.06},
cost_table {"model_a|harness_c|bench_a": 2.0}). `load_sampler` converts
them back.
"""

import json

from .sampler import InformationGainSampler
from .space import ExperimentSpace


def append(path, record: dict) -> None:
    """Append one record to the ledger as a single JSON line."""
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def space_record(sampler: InformationGainSampler, sampler_kwargs: dict | None = None) -> dict:
    """Build the line-1 'space' record for a sampler, recording the seed.

    Seed and n_seed are pulled from the live sampler so every run's ledger
    carries the exact values needed to replicate its stochastic decisions
    byte-for-byte (the whole point of the constructor seed). tau_int and
    cost_table keys are JSON-encoded ('|'-joined) if present in the passed
    sampler_kwargs; load_sampler reverses this.
    """
    kwargs = dict(sampler_kwargs or {})
    if isinstance(kwargs.get("tau_int"), dict):
        kwargs["tau_int"] = {"|".join(sorted(k)): v for k, v in kwargs["tau_int"].items()}
    if isinstance(kwargs.get("cost_table"), dict):
        kwargs["cost_table"] = {"|".join(k): v for k, v in kwargs["cost_table"].items()}
    kwargs.setdefault("seed", sampler.seed)  # always record the seed
    kwargs.setdefault("n_seed", sampler.n_seed)  # and the concrete seed size
    return {"type": "space", "factors": sampler.space.factors, "sampler_kwargs": kwargs}


def load_sampler(path) -> InformationGainSampler:
    """Rebuild the sampler from a ledger by replaying its observations."""
    records = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: malformed ledger line {lineno}: {exc}") from exc
    if not records or records[0].get("type") != "space":
        raise ValueError(f"{path}: the first ledger line must be a 'space' record")

    head = records[0]
    kwargs = dict(head.get("sampler_kwargs") or {})
    if isinstance(kwargs.get("tau_int"), dict):
        kwargs["tau_int"] = {frozenset(k.split("|")): v for k, v in kwargs["tau_int"].items()}
    if isinstance(kwargs.get("cost_table"), dict):
        kwargs["cost_table"] = {tuple(k.split("|")): v for k, v in kwargs["cost_table"].items()}
    sampler = InformationGainSampler(ExperimentSpace(head["factors"]), **kwargs)

    for rec in records[1:]:
        if rec.get("type") != "observation":  # suggestions are audit-only
            continue
        sampler.observe(
            rec["config"],
            rec["value"],
            variance=rec.get("variance"),
            m=rec.get("m"),
            cost=rec.get("cost"),
            task_outcomes=rec.get("task_outcomes"),
        )
    return sampler
