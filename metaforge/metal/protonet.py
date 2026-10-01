"""Prototypical Networks (Snell, Swersky & Zemel, 2017).

Class prototypes are the mean support embedding; queries are classified by
softmax over negative squared Euclidean distance to prototypes. Implementation
is the query-only-gradient variant (prototypes treated as constants), which is a
standard, valid simplification and keeps the math bheaviour transparent.

Verifiable invariants
---------------------
* ``prototype[c] == mean(support_embedding[y == c])`` (exact, up to float).
* For a query point equal to a support point of class c, ``proba[c] == 1``.
* Query softmax rows sum to 1 (exact).
"""

from __future__ import annotations

import numpy as np

from ..core.util import to_onehot
from .base import BaseMetaLearner
from .nn import ce_loss, grad_step
from .predictor import Predictor


def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


class ProtoNet(BaseMetaLearner):
    name = "protonet"

    def __init__(self, cfg, seed: int | None = None):
        super().__init__(cfg, seed)
        self._embed_dim = cfg.embed_dim

    def _embed(self, x: np.ndarray) -> np.ndarray:
        return self._backbone.forward(x)

    @staticmethod
    def _clip(grads: dict, max_norm: float = 5.0) -> dict:
        """Clip the global gradient norm (standard SGD stabiliser).

        Without this, the distance gradient in :meth:`meta_train` can blow the
        embedding weights up to ``inf`` in a single step at high ``lr_outer``,
        killing the network. Clipping keeps every update finite and bounded.
        """
        norm = float(np.sqrt(sum(np.sum(g**2) for g in grads.values())))
        if norm > max_norm and norm > 0.0:
            scale = max_norm / norm
            grads = {k: g * scale for k, g in grads.items()}
        return grads

    def _prototypes(self, x_s: np.ndarray, y_s: np.ndarray, n_way: int) -> np.ndarray:
        emb = self._embed(x_s)
        protos = np.zeros((n_way, emb.shape[1]), dtype=float)
        for c in range(n_way):
            m = y_s == c
            protos[c] = emb[m].mean(axis=0)
        return protos

    def meta_train(self, sampler) -> "ProtoNet":
        dim = sampler.sample().x_support.shape[1]
        bb = self._build(dim, self._embed_dim, "linear", seed=self.seed)
        self._backbone = bb
        for _ in range(self.cfg.meta_iters):
            grads = None
            for _ in range(self.cfg.meta_batch):
                task = sampler.sample()
                n_way = task.n_way
                s_emb = self._embed(task.x_support)
                protos = np.zeros((n_way, s_emb.shape[1]), dtype=float)
                for c in range(n_way):
                    m = task.y_support == c
                    protos[c] = s_emb[m].mean(axis=0)
                q_emb = self._embed(task.x_query)  # (Q,E) -> sets cache
                diff = q_emb[:, None, :] - protos[None, :, :]  # (Q,n_way,E)
                dist = np.sum(diff**2, axis=-1)  # (Q,n_way)
                probs = _softmax(-dist, -1)
                _, dlogits = ce_loss(probs, to_onehot(task.y_query, n_way))
                dL_d_qemb = -2.0 * np.einsum("qc,qce->qe", dlogits, diff)
                g = bb.backward(dL_d_qemb)
                grads = g if grads is None else {k: grads[k] + g[k] for k in g}
            grads = {k: v / self.cfg.meta_batch for k, v in grads.items()}
            grads = self._clip(grads, max_norm=5.0)
            bb.set_params(grad_step(bb.get_params(), grads, self.cfg.lr_outer))
        return self

    def adapt(self, task) -> Predictor:
        n_way = task.n_way
        protos = self._prototypes(task.x_support, task.y_support, n_way)

        def proba(x):
            q_emb = self._embed(x)
            diff = q_emb[:, None, :] - protos[None, :, :]
            return _softmax(-np.sum(diff**2, axis=-1), -1)

        return Predictor(lambda x: np.argmax(proba(x), axis=-1), proba, kind="cls")
