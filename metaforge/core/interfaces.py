"""Protocol contracts.

Modules depend only on these abstract interfaces, never on concrete classes.
This is what lets the pipeline swap in a pure-numpy learner, a scikit-learn
baseline, or an optional torch-backed SOTA learner without touching call sites.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from .types import Task


@runtime_checkable
class TaskSampler(Protocol):
    """Yields tasks from a task distribution."""

    def sample(self) -> Task: ...

    @property
    def kind(self) -> str: ...


@runtime_checkable
class Predictor(Protocol):
    """A fitted adapter that predicts for a single task at query time."""

    def predict(self, x: np.ndarray) -> np.ndarray: ...

    def predict_proba(self, x: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class MetaLearner(Protocol):
    """A meta-learning algorithm."""

    name: str

    def meta_train(self, sampler: Any) -> None: ...

    def adapt(self, task: Task) -> Predictor: ...

    def evaluate(self, task: Task) -> float: ...
