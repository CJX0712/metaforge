"""Core dataclasses shared across the MetaForge pipeline.

A ``Task`` is the atomic unit of few-shot learning: a support set the learner
adapts on and a query set it is evaluated on. ``kind`` discriminates
classification (discrete labels, ``n_way``/``k_shot`` defined) from regression
(continuous targets, those fields unused).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Task:
    x_support: np.ndarray  # (n_way*k_shot, dim) or (n_support, dim)
    y_support: np.ndarray  # (n_support,) int labels or float targets
    x_query: np.ndarray  # (n_way*q_shot, dim) or (n_query, dim)
    y_query: np.ndarray  # (n_query,) int labels or float targets
    kind: str = "cls"  # "cls" | "reg"
    n_way: int = 0
    k_shot: int = 0
    q_shot: int = 0

    def classes(self) -> list:
        if self.kind != "cls":
            return []
        return sorted(set(int(y) for y in self.y_support.tolist()))

    def check(self) -> None:
        from .errors import DataError

        xs, ys = self.x_support, self.y_support
        xq, yq = self.x_query, self.y_query
        if xs.ndim != 2 or xq.ndim != 2:
            raise DataError("support/query features must be 2-D")
        if xs.shape[1] != xq.shape[1]:
            raise DataError("support/query feature dim mismatch")
        if ys.shape[0] != xs.shape[0] or yq.shape[0] != xq.shape[0]:
            raise DataError("label/support count mismatch")
        if self.kind == "cls":
            if len(self.classes()) != self.n_way:
                raise DataError(f"task has {len(self.classes())} classes but n_way={self.n_way}")
            # every query label must occur in support
            sup = set(self.classes())
            for y in self.y_query.tolist():
                if int(y) not in sup:
                    raise DataError("query contains a class unseen in support")


@dataclass
class BenchResult:
    algorithm: str
    metric_name: str
    mean: float
    std: float
    n: int
    detail: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "metric": self.metric_name,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "n": self.n,
        }
