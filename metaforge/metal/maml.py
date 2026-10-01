"""MAML — Model-Agnostic Meta-Learning (Finn, Abbeel & Levine, 2017).

This is the **first-order** variant (FOMAML): the meta-gradient is taken as the
gradient of the query loss evaluated at the adapted parameters ``phi`` and
applied directly to ``theta``. First-order MAML drops the expensive
second-derivative through the inner update while keeping most of the gains — a
well-established, widely-used simplification.

Verifiable invariant
--------------------
On the sinusoid regression task, after meta-training the adapted model's query
MSE is lower than a model adapted from a random init under the same number of
inner steps (the meta-init is genuinely a good starting point).
"""

from __future__ import annotations

import numpy as np

from ..core.util import to_onehot
from .base import BaseMetaLearner
from .nn import ce_loss, clip_grad, grad_step, mse_loss
from .predictor import Predictor


class MAML(BaseMetaLearner):
    name = "maml"

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
        g = clip_grad(bb.backward(d), max_norm=5.0)
        return grad_step(params, g, self.cfg.lr_inner)

    def meta_train(self, sampler) -> "MAML":
        task0 = sampler.sample()
        dim = task0.x_support.shape[1]
        is_cls = task0.kind == "cls"
        out_dim = task0.n_way if is_cls else 1
        bb = self._build(dim, out_dim, "softmax" if is_cls else "linear", seed=self.seed)
        self._backbone = bb
        theta = bb.get_params()
        for _ in range(self.cfg.meta_iters):
            meta_grad = None
            for _ in range(self.cfg.meta_batch):
                task = sampler.sample()
                phi = {k: v.copy() for k, v in theta.items()}
                for _ in range(self.cfg.inner_steps):
                    phi = self._inner_step(phi, task)
                bb.set_params(phi)
                if task.kind == "cls":
                    out = bb.forward(task.x_query)
                    _, d = ce_loss(out, to_onehot(task.y_query, task.n_way))
                else:
                    out = bb.forward(task.x_query)
                    _, d = mse_loss(out, task.y_query)
                g = bb.backward(d)
                meta_grad = g if meta_grad is None else {k: meta_grad[k] + g[k] for k in g}
            meta_grad = {k: meta_grad[k] / self.cfg.meta_batch for k in meta_grad}
            meta_grad = clip_grad(meta_grad, max_norm=5.0)
            theta = grad_step(theta, meta_grad, self.cfg.lr_outer)
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
