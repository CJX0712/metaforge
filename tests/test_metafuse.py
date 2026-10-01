import numpy as np

from metaforge.core.config import MetaConfig
from metaforge.data.samplers import blob_preset
from metaforge.metal.metafuse import MetaFuse


def test_metafuse_guard_invariant():
    """The strict-improvement guard must be internally consistent with the
    recorded validation scores."""
    cfg = MetaConfig(seed=5, meta_iters=300, eval_episodes=40, fuse_tol=0.02)
    s_train = blob_preset("medium", seed=31)
    mf = MetaFuse(cfg)
    mf.meta_train(s_train)

    rep = mf._report
    member_scores = rep["member_scores"]
    fused = rep["fused_score"]
    best = max(member_scores)
    if rep["use_fusion"]:
        assert fused >= best - cfg.fuse_tol - 1e-6, (
            f"guard says use_fusion but fused {fused} < best {best} - tol"
        )
    else:
        assert fused < best - cfg.fuse_tol + 1e-6, (
            f"guard says delegate but fused {fused} >= best {best} - tol"
        )
    # best_member must actually be the strongest member
    assert rep["best_member"] == mf.members[int(np.argmax(member_scores))].name


def test_metafuse_predict_shape():
    cfg = MetaConfig(seed=5, meta_iters=100, eval_episodes=10)
    s = blob_preset("medium", seed=9)
    mf = MetaFuse(cfg)
    mf.meta_train(s)
    task = s.sample()
    p = mf.adapt(task)
    assert p.predict(task.x_query).shape == task.y_query.shape
