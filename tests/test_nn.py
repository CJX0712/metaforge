"""Gradient-check invariant for the hand-rolled MLP.

The analytic gradient produced by :meth:`MLP.backward` must match a central
finite-difference gradient to high precision. This is the single most important
invariant of the whole system: every meta-learner relies on it.
"""

import numpy as np

from metaforge.core.util import to_onehot
from metaforge.metal.nn import MLP, ce_loss, mse_loss


def _numeric_grad(bb, X, Y, loss_fn, eps=1e-6):
    params0 = bb.get_params()
    num = {}
    for k in params0:
        num[k] = np.zeros_like(params0[k])
        it = np.nditer(params0[k], flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            pp = {kk: v.copy() for kk, v in params0.items()}
            pp[k][idx] = params0[k][idx] + eps
            bb.set_params(pp)
            lp, _ = loss_fn(bb.forward(X), Y)
            pp[k][idx] = params0[k][idx] - eps
            bb.set_params(pp)
            pm, _ = loss_fn(bb.forward(X), Y)
            num[k][idx] = (lp - pm) / (2 * eps)
            it.iternext()
    bb.set_params(params0)
    return num


def test_mlp_regression_gradcheck():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 3))
    Y = rng.normal(size=(8, 1))
    bb = MLP([3, 5, 1], out_activation="linear", seed=1)
    out = bb.forward(X)
    _, d = mse_loss(out, Y)
    ana = bb.backward(d)
    num = _numeric_grad(bb, X, Y, mse_loss)
    for k in ana:
        rel = np.abs(ana[k] - num[k]) / (np.abs(num[k]) + 1e-8)
        assert rel.max() < 1e-4, f"{k} max rel err {rel.max():.2e}"


def test_mlp_classification_gradcheck():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(10, 4))
    Y = rng.integers(0, 3, size=(10,))
    bb = MLP([4, 6, 3], out_activation="softmax", seed=3)
    out = bb.forward(X)
    _, d = ce_loss(out, to_onehot(Y, 3))
    ana = bb.backward(d)
    num = _numeric_grad(bb, X, to_onehot(Y, 3), ce_loss)
    for k in ana:
        rel = np.abs(ana[k] - num[k]) / (np.abs(num[k]) + 1e-8)
        assert rel.max() < 1e-4, f"{k} max rel err {rel.max():.2e}"


def test_mlp_softmax_rows_sum_to_one():
    bb = MLP([4, 6, 3], out_activation="softmax", seed=0)
    X = np.random.default_rng(1).normal(size=(5, 4))
    out = bb.forward(X)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-8)
