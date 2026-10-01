"""Algorithm registry + default selection per domain."""

from __future__ import annotations

from .baselines import (
    KnnClassifier,
    LogisticClassifier,
    MajorityClassifier,
    MeanRegressor,
    ScratchClassifier,
    ScratchRegressor,
)
from .maml import MAML
from .matching import MatchingNet
from .metafuse import MetaFuse
from .protonet import ProtoNet
from .reptile import Reptile

REGISTRY = {
    "protonet": ProtoNet,
    "reptile": Reptile,
    "maml": MAML,
    "matching": MatchingNet,
    "metafuse": MetaFuse,
    "baseline_majority": MajorityClassifier,
    "baseline_knn": KnnClassifier,
    "baseline_logreg": LogisticClassifier,
    "baseline_scratch": ScratchClassifier,
    "baseline_mean": MeanRegressor,
    "baseline_scratch_reg": ScratchRegressor,
}


def build(name: str, cfg, **kw):
    if name not in REGISTRY:
        raise KeyError(f"unknown algorithm {name!r}; known={sorted(REGISTRY)}")
    return REGISTRY[name](cfg, **kw)


def default_algorithms(cfg, domain: str):
    if domain == "cls":
        return [
            ProtoNet(cfg),
            Reptile(cfg),
            MAML(cfg),
            MatchingNet(cfg),
            MetaFuse(cfg),
            MajorityClassifier(cfg),
            KnnClassifier(cfg),
            LogisticClassifier(cfg),
            ScratchClassifier(cfg),
        ]
    # regression
    return [
        Reptile(cfg),
        MAML(cfg),
        MetaFuse(cfg, members=[Reptile(cfg, seed=cfg.seed + 1), MAML(cfg, seed=cfg.seed + 2)]),
        MeanRegressor(cfg),
        ScratchRegressor(cfg),
    ]
