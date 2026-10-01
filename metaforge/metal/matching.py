"""Matching Networks (Vinyals et al., 2016), basic form (no FCE).

Classification is done non-parametrically: each query point's prediction is the
attention-weighted average of support one-hot labels, where attention is softmax
over cosine similarity between learned embeddings. The embedding network is
trained episodically to make this attention score well. Backprop flows through
*both* support and query embeddings (we call ``backward`` twice with the two
caches).
"""

from __future__ import annotations

import numpy as np

from ..core.util import to_onehot
from .base import BaseMetaLearner
from .nn import ce_loss, clip_grad, grad_step
from .predictor import Predictor


def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def _cosine(q: np.ndarray, s: np.ndarray) -> np.ndarray:
    qn = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    sn = s / (np.linalg.norm(s, axis=-1, keepdims=True) + 1e-12)
    return qn @ sn.T


class MatchingNet(BaseMetaLearner):
    name = "matching"

    def __init__(self, cfg, seed: int | None = None):
        super().__init__(cfg, seed)
        self._embed_dim = cfg.embed_dim

    def meta_train(self, sampler) -> "MatchingNet":
        dim = sampler.sample().x_support.shape[1]
        bb = self._build(dim, self._embed_dim, "linear", seed=self.seed)
        self._backbone = bb
        for _ in range(self.cfg.meta_iters):
            grads = None
            for _ in range(self.cfg.meta_batch):
                task = sampler.sample()
                n_way = task.n_way
                s_emb = bb.forward(task.x_support)  # cache_s
                q_emb = bb.forward(task.x_query)  # cache_q
                sim = _cosine(q_emb, s_emb)  # (Q,S)
                attn = _softmax(sim, -1)  # (Q,S)
                ys = to_onehot(task.y_support, n_way)  # (S,n_way)
                pred = attn @ ys  # (Q,n_way)
                _, d = ce_loss(pred, to_onehot(task.y_query, n_way))  # (Q,n_way)
                d_attn = d @ ys.T  # (Q,S)
                d_sim = attn * (d_attn - np.sum(d_attn * attn, axis=-1, keepdims=True))
                a = 1.0 / (np.linalg.norm(q_emb, axis=-1, keepdims=True) + 1e-12)  # (Q,1)
                b = 1.0 / (np.linalg.norm(s_emb, axis=-1, keepdims=True) + 1e-12)  # (S,1)
                W = a * b.T  # (Q,S)
                dL_d_qemb = (d_sim * W) @ s_emb  # (Q,E)
                dL_d_s_emb = (d_sim * W).T @ q_emb  # (S,E)
                g_q = bb.backward(dL_d_qemb)  # uses cache_q
                bb.forward(task.x_support)  # restore cache_s
                g_s = bb.backward(dL_d_s_emb)  # uses cache_s
                g = {k: g_q[k] + g_s[k] for k in g_q}
                grads = g if grads is None else {k: grads[k] + g[k] for k in g}
            grads = {k: v / self.cfg.meta_batch for k, v in grads.items()}
            grads = clip_grad(grads, max_norm=5.0)
            bb.set_params(grad_step(bb.get_params(), grads, self.cfg.lr_outer))
        return self

    def adapt(self, task) -> Predictor:
        bb = self._backbone
        n_way = task.n_way
        ys = to_onehot(task.y_support, n_way)

        def proba(x):
            s_emb = bb.forward(task.x_support)
            q_emb = bb.forward(x)
            attn = _softmax(_cosine(q_emb, s_emb), -1)
            return attn @ ys

        return Predictor(lambda x: np.argmax(proba(x), axis=-1), proba, kind="cls")
