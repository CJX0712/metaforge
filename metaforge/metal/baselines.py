"""Honest non-meta baselines.

These deliberately do *not* meta-learn. They answer the question "how good is a
reasonable learner trained only on the few support examples (or a trivial rule)?"
and give the meta-learners a real floor to beat.

* ``baseline_majority`` / ``baseline_mean`` — constant predictors (lower bound).
* ``baseline_knn`` / ``baseline_logreg`` — scikit-learn models fit on the support
  set only (top-OSS reuse, no meta-training).
* ``baseline_scratch`` / ``baseline_scratch_reg`` — a tiny MLP fit from scratch on
  the support set (the natural "no meta" upper-ish baseline).
"""

from __future__ import annotations

import numpy as np

from ..core.util import to_onehot
from .base import BaseMetaLearner
from .nn import ce_loss, clip_grad, grad_step, mse_loss
from .predictor import Predictor


class MajorityClassifier(BaseMetaLearner):
    name = "baseline_majority"

    def meta_train(self, sampler):
        return self

    def adapt(self, task) -> Predictor:
        vals, counts = np.unique(task.y_support, return_counts=True)
        majority = int(vals[int(np.argmax(counts))])
        return Predictor(lambda x: np.full(x.shape[0], majority), kind="cls")


class KnnClassifier(BaseMetaLearner):
    name = "baseline_knn"

    def __init__(self, cfg, seed=None, k=3):
        super().__init__(cfg, seed)
        self.k = k

    def meta_train(self, sampler):
        return self

    def adapt(self, task) -> Predictor:
        from sklearn.neighbors import KNeighborsClassifier

        k = min(self.k, task.x_support.shape[0])
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(task.x_support, task.y_support)
        return Predictor(clf.predict, clf.predict_proba, kind="cls")


class LogisticClassifier(BaseMetaLearner):
    name = "baseline_logreg"

    def meta_train(self, sampler):
        return self

    def adapt(self, task) -> Predictor:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=1000)
        clf.fit(task.x_support, task.y_support)
        return Predictor(clf.predict, clf.predict_proba, kind="cls")


class ScratchClassifier(BaseMetaLearner):
    name = "baseline_scratch"

    def __init__(self, cfg, seed=None, epochs=200, lr=0.05):
        super().__init__(cfg, seed)
        self.epochs = epochs
        self.lr = lr

    def meta_train(self, sampler):
        return self

    def adapt(self, task) -> Predictor:
        dim = task.x_support.shape[1]
        n_way = task.n_way
        bb = self._build(dim, n_way, "softmax", seed=self.seed)
        for _ in range(self.epochs):
            out = bb.forward(task.x_support)
            _, d = ce_loss(out, to_onehot(task.y_support, n_way))
            g = clip_grad(bb.backward(d), max_norm=5.0)
            bb.set_params(grad_step(bb.get_params(), g, self.lr))

        def pred(x):
            return np.argmax(bb.forward(x), axis=-1)

        return Predictor(pred, lambda x: bb.forward(x), kind="cls")


class MeanRegressor(BaseMetaLearner):
    name = "baseline_mean"

    def meta_train(self, sampler):
        return self

    def adapt(self, task) -> Predictor:
        mu = float(np.mean(task.y_support))
        return Predictor(lambda x: np.full(x.shape[0], mu), kind="reg")


class ScratchRegressor(BaseMetaLearner):
    name = "baseline_scratch_reg"

    def __init__(self, cfg, seed=None, epochs=300, lr=0.01):
        super().__init__(cfg, seed)
        self.epochs = epochs
        self.lr = lr

    def meta_train(self, sampler):
        return self

    def adapt(self, task) -> Predictor:
        dim = task.x_support.shape[1]
        bb = self._build(dim, 1, "linear", seed=self.seed)
        for _ in range(self.epochs):
            out = bb.forward(task.x_support)
            _, d = mse_loss(out, task.y_support)
            g = clip_grad(bb.backward(d), max_norm=5.0)
            bb.set_params(grad_step(bb.get_params(), g, self.lr))

        def pred(x):
            return bb.forward(x).reshape(-1)

        return Predictor(pred, pred, kind="reg")
