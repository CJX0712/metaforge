import numpy as np

from metaforge.core.config import MetaConfig
from metaforge.data.samplers import SinusoidSampler
from metaforge.metal.maml import MAML
from metaforge.metal.nn import mse_loss


def _query_mse_after_adapt(model, task, inner_steps, lr):
    """Adapt from the model's current params and return mean query MSE."""
    model._backbone.set_params(model._theta)
    phi = {k: v.copy() for k, v in model._theta.items()}
    for _ in range(inner_steps):
        model._backbone.set_params(phi)
        out = model._backbone.forward(task.x_support)
        _, d = mse_loss(out, task.y_support)
        g = model._backbone.backward(d)
        phi = {k: phi[k] - lr * g[k] for k in phi}
    model._backbone.set_params(phi)
    out = model._backbone.forward(task.x_query)
    loss, _ = mse_loss(out, task.y_query)
    return loss


def test_maml_adaptation_reduces_query_loss():
    cfg = MetaConfig(
        seed=1, meta_iters=500, eval_episodes=30, lr_inner=0.01, lr_outer=0.1, inner_steps=10
    )
    s_train = SinusoidSampler(n_support=10, n_query=10, seed=11)
    m = MAML(cfg)
    m.meta_train(s_train)

    s_eval = SinusoidSampler(n_support=10, n_query=10, seed=777)
    before, after = [], []
    for _ in range(30):
        task = s_eval.sample()
        # before adaptation (raw meta-init)
        m._backbone.set_params(m._theta)
        out = m._backbone.forward(task.x_query)
        b, _ = mse_loss(out, task.y_query)
        before.append(b)
        after.append(_query_mse_after_adapt(m, task, cfg.inner_steps, cfg.lr_inner))
    mean_before, mean_after = float(np.mean(before)), float(np.mean(after))
    assert mean_after < mean_before - 1e-3, (
        f"adaptation did not reduce loss: {mean_before:.3f} -> {mean_after:.3f}"
    )


def test_maml_beats_mean_regressor():
    cfg = MetaConfig(
        seed=2, meta_iters=600, eval_episodes=60, lr_inner=0.01, lr_outer=0.1, inner_steps=10
    )
    s_train = SinusoidSampler(n_support=10, n_query=10, seed=13)
    m = MAML(cfg)
    m.meta_train(s_train)
    s_eval = SinusoidSampler(n_support=10, n_query=10, seed=999)
    accs = [m.evaluate(s_eval.sample()) for _ in range(60)]
    assert float(np.mean(accs)) < 2.2, f"MAML rmse {np.mean(accs):.3f} too high"
