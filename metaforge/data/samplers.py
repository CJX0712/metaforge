"""Few-shot task distributions: the data the meta-learners meta-train on.

Two generators are provided, both with continuous difficulty knobs and
deterministic RNGs:

* :class:`GaussianBlobSampler` -- N-way K-shot *classification*. Difficulty is
  controlled by ``class_sep`` (how far apart the class centers are drawn) and
  ``sigma`` (within-class noise).
* :class:`SinusoidSampler` -- *regression* few-shot sine curves, the canonical
  MAML demo (amplitude / phase vary per task).
"""

from __future__ import annotations

import numpy as np

from ..core.errors import DataError
from ..core.types import Task
from ..core.util import seeded_rng


class GaussianBlobSampler:
    """N-way K-shot classification with controllable separability."""

    kind = "cls"

    def __init__(
        self,
        n_way: int = 5,
        k_shot: int = 5,
        q_shot: int = 15,
        dim: int = 4,
        class_sep: float = 3.0,
        sigma: float = 1.0,
        seed: int = 0,
    ):
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_shot = q_shot
        self.dim = dim
        self.class_sep = class_sep
        self.sigma = sigma
        self._rng = seeded_rng(seed)

    def sample(self) -> Task:
        rng = self._rng
        centers = rng.normal(0.0, self.class_sep, size=(self.n_way, self.dim))
        xs_s, ys_s, xs_q, ys_q = [], [], [], []
        for c in range(self.n_way):
            pts = centers[c] + rng.normal(
                0.0, self.sigma, size=(self.k_shot + self.q_shot, self.dim)
            )
            xs_s.append(pts[: self.k_shot])
            ys_s.extend([c] * self.k_shot)
            xs_q.append(pts[self.k_shot :])
            ys_q.extend([c] * self.q_shot)
        x_s = np.vstack(xs_s).astype(float)
        y_s = np.asarray(ys_s, dtype=int)
        x_q = np.vstack(xs_q).astype(float)
        y_q = np.asarray(ys_q, dtype=int)
        # shuffle within support / query so order is not class-contiguous
        p_s = rng.permutation(x_s.shape[0])
        p_q = rng.permutation(x_q.shape[0])
        task = Task(
            x_support=x_s[p_s],
            y_support=y_s[p_s],
            x_query=x_q[p_q],
            y_query=y_q[p_q],
            kind="cls",
            n_way=self.n_way,
            k_shot=self.k_shot,
            q_shot=self.q_shot,
        )
        task.check()
        return task

    def sample_many(self, n: int) -> list[Task]:
        return [self.sample() for _ in range(n)]


class SinusoidSampler:
    """Regression few-shot tasks: y = A * sin(x + phi) + noise."""

    kind = "reg"

    def __init__(
        self,
        n_support: int = 10,
        n_query: int = 10,
        amp_lo: float = 0.1,
        amp_hi: float = 5.0,
        phase_lo: float = 0.0,
        phase_hi: float = np.pi,
        noise: float = 0.1,
        seed: int = 0,
    ):
        self.n_support = n_support
        self.n_query = n_query
        self.amp_lo = amp_lo
        self.amp_hi = amp_hi
        self.phase_lo = phase_lo
        self.phase_hi = phase_hi
        self.noise = noise
        self._rng = seeded_rng(seed)

    def _task(self, n_points: int) -> tuple[np.ndarray, np.ndarray]:
        rng = self._rng
        amp = rng.uniform(self.amp_lo, self.amp_hi)
        phase = rng.uniform(self.phase_lo, self.phase_hi)
        x = rng.uniform(-5.0, 5.0, size=(n_points, 1))
        y = amp * np.sin(x + phase) + rng.normal(0.0, self.noise, size=x.shape)
        return x.astype(float), y.astype(float)

    def sample(self) -> Task:
        xs, ys = self._task(self.n_support)
        xq, yq = self._task(self.n_query)
        return Task(x_support=xs, y_support=ys, x_query=xq, y_query=yq, kind="reg")

    def sample_many(self, n: int) -> list[Task]:
        return [self.sample() for _ in range(n)]


# ---- presets used by the demo / benchmark --------------------------------
def blob_preset(level: str, **kw) -> GaussianBlobSampler:
    table = {
        "easy": dict(class_sep=6.0, sigma=0.6),
        "medium": dict(class_sep=3.0, sigma=1.0),
        "hard": dict(class_sep=1.5, sigma=1.4),
    }
    if level not in table:
        raise DataError(f"unknown blob level {level!r}")
    return GaussianBlobSampler(**{**table[level], **kw})
