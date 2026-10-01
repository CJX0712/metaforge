"""Configuration object for MetaForge.

All knobs live in one dataclass so the CLI, pipeline and tests share the same
single source of truth. ``from_env`` lets a deployment override any field with
``METAFORGE_<FIELD>`` (e.g. ``METAFORGE_SEED=7``).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class MetaConfig:
    seed: int = 20261002
    dim: int = 4
    # classification episode geometry
    n_way: int = 5
    k_shot: int = 5
    q_shot: int = 15
    # regression episode geometry
    reg_support: int = 10
    reg_query: int = 10
    # optimization
    lr_inner: float = 0.05
    lr_outer: float = 0.1
    inner_steps: int = 5
    meta_iters: int = 400
    meta_batch: int = 4
    eval_episodes: int = 200
    # backbone
    hidden: tuple = (32, 16)
    embed_dim: int = 16
    # fusion
    fuse_tol: float = 0.02
    fuse_temp: float = 3.0

    @classmethod
    def from_env(cls, **overrides: Any) -> "MetaConfig":
        cfg = cls()
        for f in cls.__dataclass_fields__:  # type: ignore[attr-defined]
            env = os.environ.get(f"METAFORGE_{f.upper()}")
            if env is not None:
                cur = getattr(cfg, f)
                try:
                    if isinstance(cur, tuple):
                        setattr(cfg, f, tuple(float(x) for x in env.split(",")))
                    elif isinstance(cur, bool):
                        setattr(cfg, f, env.lower() in ("1", "true", "yes"))
                    elif isinstance(cur, int):
                        setattr(cfg, f, int(env))
                    elif isinstance(cur, float):
                        setattr(cfg, f, float(env))
                    else:
                        setattr(cfg, f, env)
                except ValueError:
                    continue
        for k, v in overrides.items():
            if k in cls.__dataclass_fields__:  # type: ignore[attr-defined]
                setattr(cfg, k, v)
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)

    def clone(self, **overrides: Any) -> "MetaConfig":
        d = self.to_dict()
        d.update(overrides)
        return MetaConfig(**d)
