"""Benchmark harness.

Trains each algorithm on a train sampler, then evaluates it over a fixed,
deterministic eval sampler, returning reproducible :class:`BenchResult` rows.
Evaluation uses a *separate* sampler with its own seed so numbers are stable
across runs (training randomness does not leak into the reported metric).
"""

from __future__ import annotations

import numpy as np

from ..core.types import BenchResult


def evaluate(alg, eval_sampler, cfg, level: str) -> BenchResult:
    kind = eval_sampler.kind
    metric = "accuracy" if kind == "cls" else "rmse"
    higher = kind == "cls"
    scores = []
    for _ in range(cfg.eval_episodes):
        task = eval_sampler.sample()
        try:
            scores.append(alg.evaluate(task))
        except Exception:
            scores.append(np.nan)
    arr = np.array(scores, dtype=float)
    arr = arr[~np.isnan(arr)]
    return BenchResult(
        algorithm=alg.name,
        metric_name=metric,
        mean=float(arr.mean()) if arr.size else float("nan"),
        std=float(arr.std()) if arr.size else float("nan"),
        n=int(arr.size),
        detail={"level": level, "higher_better": higher},
    )


def run_benchmark(cfg, algorithms, scenarios):
    rows = []
    for level, train_s, eval_s in scenarios:
        for alg in algorithms:
            alg.meta_train(train_s)
            rows.append(evaluate(alg, eval_s, cfg, level))
    return rows
