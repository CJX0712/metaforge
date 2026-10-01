import numpy as np

from metaforge.core.config import MetaConfig
from metaforge.data.samplers import blob_preset
from metaforge.metal.baselines import MajorityClassifier
from metaforge.metal.reptile import Reptile


def test_reptile_beats_majority():
    cfg = MetaConfig(
        seed=4, meta_iters=600, eval_episodes=100, lr_inner=0.1, lr_outer=0.2, inner_steps=10
    )
    s_train = blob_preset("medium", seed=21)
    s_eval = blob_preset("medium", seed=2999)
    r = Reptile(cfg)
    r.meta_train(s_train)
    accs = np.array([r.evaluate(s_eval.sample()) for _ in range(100)])
    mean_acc = float(accs.mean())
    maj = MajorityClassifier(cfg)
    maj_acc = float(np.mean([maj.evaluate(s_eval.sample()) for _ in range(100)]))
    assert mean_acc > 0.3, f"Reptile acc {mean_acc:.3f}"
    assert mean_acc > maj_acc + 0.05, (
        f"Reptile {mean_acc:.3f} not clearly above majority {maj_acc:.3f}"
    )


def test_reptile_deterministic_with_seed():
    def run(seed):
        cfg = MetaConfig(seed=seed, meta_iters=50, eval_episodes=20)
        s_train = blob_preset("medium", seed=seed)
        s_eval = blob_preset("medium", seed=seed + 5000)
        r = Reptile(cfg)
        r.meta_train(s_train)
        return float(np.mean([r.evaluate(s_eval.sample()) for _ in range(20)]))

    assert abs(run(7) - run(7)) < 1e-9
