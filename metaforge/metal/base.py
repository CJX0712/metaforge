"""Common machinery for the meta-learners."""

from __future__ import annotations

from ..core.types import Task
from ..core.util import accuracy_cls, rmse, seeded_rng
from .nn import MLP
from .predictor import Predictor


class BaseMetaLearner:
    name = "base"

    def __init__(self, cfg, seed: int | None = None):
        self.cfg = cfg
        self.seed = int(cfg.seed if seed is None else seed)
        self._rng = seeded_rng(self.seed)
        self._backbone: MLP | None = None

    def _build(self, dim: int, out_dim: int, out_activation: str, seed=None):
        hidden = list(self.cfg.hidden)
        sizes = [dim] + hidden + [out_dim]
        s = self.seed if seed is None else seed
        return MLP(sizes, out_activation=out_activation, seed=s)

    # ---- interface --------------------------------------------------------
    def meta_train(self, sampler):  # pragma: no cover - overridden
        raise NotImplementedError

    def adapt(self, task: Task) -> Predictor:  # pragma: no cover
        raise NotImplementedError

    def evaluate(self, task: Task) -> float:
        pred = self.adapt(task).predict(task.x_query)
        if task.kind == "cls":
            return accuracy_cls(pred, task.y_query)
        return rmse(pred, task.y_query)

    def metric_name(self, kind: str) -> str:
        return "accuracy" if kind == "cls" else "rmse"
