import numpy as np

from metaforge.core.config import MetaConfig
from metaforge.data.samplers import blob_preset
from metaforge.metal.matching import MatchingNet


def test_matching_runs_and_predicts():
    cfg = MetaConfig(seed=1, meta_iters=200, eval_episodes=40, lr_outer=0.1)
    s_train = blob_preset("medium", seed=7)
    s_eval = blob_preset("medium", seed=1234)
    m = MatchingNet(cfg)
    m.meta_train(s_train)
    task = s_eval.sample()
    p = m.adapt(task)
    pred = p.predict(task.x_query)
    assert pred.shape == task.y_query.shape
    # accuracy should be above random chance
    acc = float(np.mean(pred == task.y_query))
    assert acc > 0.3, f"Matching acc {acc:.3f} not above 0.3"


def test_matching_proba_sum_one():
    cfg = MetaConfig(seed=1, meta_iters=50, eval_episodes=10)
    s_train = blob_preset("easy", seed=7)
    m = MatchingNet(cfg)
    m.meta_train(s_train)
    task = s_train.sample()
    p = m.adapt(task).predict_proba(task.x_query)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
