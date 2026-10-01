"""Reptile (Nichol, Achiam & Schulman, 2018).

First-order meta-optimizer: for each task run a few SGD steps from the current
parameters ``theta`` to ``phi``, then move ``theta`` toward ``phi``. No
second-order derivatives, so it is cheap and stable.

Verifiable invariant
--------------------
After meta-training, few-shot accuracy on a *tuned-difficulty* task is strictly
greater than a no-meta (train-on-support) baseline and above random chance.
"""

from __future__ import annotations

import numpy as np

from ..core.util import to_onehot
from .base import BaseMetaLearner
from .nn import ce_loss, clip_grad, grad_step, mse_loss
from .predictor import Predictor


class Reptile(BaseMetaLearner):
    name = "reptile"

    def __init__(self, cfg, seed: int | None = None):
        super().__init__(cfg, seed)
        self._theta = None

    def _inner_step(self, params: dict, task) -> dict:
        bb = self._backbone
        bb.set_params(params)
        if task.kind == "cls":
            out = bb.forward(task.x_support)
            _, d = ce_loss(out, to_onehot(task.y_support, task.n_way))
        else:
            out = bb.forward(task.x_support)
            _, d = mse_loss(out, task.y_support)
        return grad_step(params, bb.backward(d), self.cfg.lr_inner)

    def meta_train(self, sampler) -> "Reptile":
        task0 = sampler.sample()
        dim = task0.x_support.shape[1]
        is_cls = task0.kind == "cls"
        out_dim = task0.n_way if is_cls else 1
        bb = self._build(dim, out_dim, "softmax" if is_cls else "linear", seed=self.seed)
        self._backbone = bb
        theta = bb.get_params()
        for _ in range(self.cfg.meta_iters):
            accum = None
            for _ in range(self.cfg.meta_batch):
                task = sampler.sample()
                phi = {k: v.copy() for k, v in theta.items()}
                for _ in range(self.cfg.inner_steps):
                    phi = self._inner_step(phi, task)
                diff = {k: phi[k] - theta[k] for k in theta}
                accum = diff if accum is None else {k: accum[k] + diff[k] for k in theta}
            avg = {k: accum[k] / self.cfg.meta_batch for k in theta}
            avg = clip_grad(avg, max_norm=5.0)
            theta = {k: theta[k] + self.cfg.lr_outer * avg[k] for k in theta}
        bb.set_params(theta)
        self._theta = theta
        return self

    def adapt(self, task) -> Predictor:
        bb = self._backbone
        phi = {k: v.copy() for k, v in self._theta.items()}
        for _ in range(self.cfg.inner_steps):
            phi = self._inner_step(phi, task)
        bb.set_params(phi)
        is_cls = task.kind == "cls"

        def pred(x):
            o = bb.forward(x)
            return np.argmax(o, axis=-1) if is_cls else o.reshape(-1)

        def proba(x):
            o = bb.forward(x)
            return o if is_cls else o.reshape(-1)

        return Predictor(pred, proba, kind=task.kind)
