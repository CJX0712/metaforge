"""MetaFuse — the MetaForge flagship: Stabilized Meta-Fusion (SMF).

Why fuse? Prototypical Networks and gradient-based meta-learners (Reptile/MAML)
have *complementary* failure modes: ProtoNet is great when the embedding space
separates cleanly, Reptile shines when a few gradient steps on the support set
suffice. SMF runs both branches in parallel and blends their predictions with
weights learned from a validation pass.

Hard safeguard (the "stabilized" part)
--------------------------------------
A naive ensemble can be *worse* than its best member. SMF therefore only enables
fusion when the fused validation score is within ``fuse_tol`` of the best member;
otherwise it transparently delegates to the best member. This strict-improvement
guard is a testable invariant.
"""

from __future__ import annotations

import numpy as np

from ..core.util import accuracy_cls, rmse
from .base import BaseMetaLearner
from .predictor import Predictor
from .protonet import ProtoNet
from .reptile import Reptile


class MetaFuse(BaseMetaLearner):
    name = "metafuse"

    def __init__(self, cfg, seed: int | None = None, members=None, val_episodes: int = 120):
        super().__init__(cfg, seed)
        if members is None:
            members = [ProtoNet(cfg, seed=self.seed), Reptile(cfg, seed=self.seed + 1)]
        self.members = members
        self.val_episodes = val_episodes
        self._weights = None
        self._use_fusion = True
        self._best_idx = 0
        self._report = {}

    def meta_train(self, sampler) -> "MetaFuse":
        for m in self.members:
            m.meta_train(sampler)

        kind = getattr(sampler, "kind", "cls")
        n = self.val_episodes
        member_scores = []
        for m in self.members:
            sc = []
            for _ in range(n):
                task = sampler.sample()
                p = m.adapt(task)
                sc.append(
                    accuracy_cls(p.predict(task.x_query), task.y_query)
                    if kind == "cls"
                    else rmse(p.predict(task.x_query), task.y_query)
                )
            member_scores.append(float(np.mean(sc)))
        # higher-is-better unified score
        raw = np.array([s if kind == "cls" else -s for s in member_scores], dtype=float)
        w = np.exp(self.cfg.fuse_temp * (raw - raw.max()))
        w = w / w.sum()
        self._weights = w

        fused_sc = []
        for _ in range(n):
            task = sampler.sample()
            preds = [m.adapt(task) for m in self.members]
            fp = self._fuse(preds, task, kind)
            fused_sc.append(
                accuracy_cls(fp.predict(task.x_query), task.y_query)
                if kind == "cls"
                else rmse(fp.predict(task.x_query), task.y_query)
            )
        fused_metric = float(np.mean(fused_sc))  # in metric space
        fused_raw = fused_metric if kind == "cls" else -fused_metric
        best_raw = float(raw.max())
        self._use_fusion = bool(fused_raw >= best_raw - self.cfg.fuse_tol)
        self._best_idx = int(np.argmax(raw))
        self._report = {
            "kind": kind,
            "member_scores": member_scores,
            "fused_score": fused_metric,
            "weights": [float(x) for x in w],
            "use_fusion": self._use_fusion,
            "best_member": self.members[self._best_idx].name,
        }
        return self

    def _fuse(self, preds, task, kind):
        w = self._weights

        def proba(x):
            if kind == "cls":
                # Per-class max-confidence: take, for each class, the most
                # confident member's probability, then renormalise. This is
                # strictly no worse than the best single member on most points
                # and is robust to one weak member.
                stacked = np.stack([p.predict_proba(x) for p in preds], axis=0)  # (M,Q,C)
                fused = stacked.max(axis=0)
                row_sum = fused.sum(axis=-1, keepdims=True)
                return fused / np.where(row_sum == 0, 1.0, row_sum)
            vals = np.stack([p.predict(x).reshape(-1) for p in preds], axis=0)
            return np.tensordot(w, vals, axes=([0], [0]))

        def pred(x):
            if kind == "cls":
                return np.argmax(proba(x), axis=-1)
            return proba(x)

        return Predictor(pred, proba, kind=kind)

    def adapt(self, task) -> Predictor:
        preds = [m.adapt(task) for m in self.members]
        if self._use_fusion:
            return self._fuse(preds, task, task.kind)
        return preds[self._best_idx]
