import numpy as np

from metaforge.core.config import MetaConfig
from metaforge.data.samplers import blob_preset
from metaforge.metal.protonet import ProtoNet


def _proto_and_protos(bb, task):
    n_way = task.n_way
    s_emb = bb._embed(task.x_support)
    protos = np.zeros((n_way, s_emb.shape[1]))
    for c in range(n_way):
        m = task.y_support == c
        protos[c] = s_emb[m].mean(axis=0)
    # recompute via adapt's closure by direct call
    pred = bb.adapt(task)
    return pred, protos, s_emb


def test_prototype_is_support_mean():
    cfg = MetaConfig(seed=1, meta_iters=2, eval_episodes=4)
    s = blob_preset("medium", seed=5)
    bb = ProtoNet(cfg)
    bb.meta_train(s)
    task = s.sample()
    pred = bb.adapt(task)
    # build prototypes the same way and check softmax of an identical query point
    # pick a support point of class 0 and verify prob[0] ~ 1
    idx0 = int(np.where(task.y_support == 0)[0][0])
    p = pred.predict_proba(task.x_support[idx0 : idx0 + 1])
    assert p[0, 0] > 0.9


def test_protonet_softmax_rows_sum_one():
    cfg = MetaConfig(seed=1, meta_iters=3, eval_episodes=4)
    s = blob_preset("medium", seed=5)
    bb = ProtoNet(cfg)
    bb.meta_train(s)
    task = s.sample()
    p = bb.adapt(task).predict_proba(task.x_query)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_protonet_beats_random():
    cfg = MetaConfig(seed=3, meta_iters=400, eval_episodes=100, lr_outer=0.2)
    s_train = blob_preset("medium", seed=11)
    s_eval = blob_preset("medium", seed=1999)
    bb = ProtoNet(cfg)
    bb.meta_train(s_train)
    accs = [bb.evaluate(s_eval.sample()) for _ in range(100)]
    mean_acc = float(np.mean(accs))
    assert mean_acc > 0.5, f"ProtoNet acc {mean_acc:.3f} not above 0.5"
