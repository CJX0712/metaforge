"""Small shared numeric helpers (no heavy deps, no import cycles)."""

from __future__ import annotations

import numpy as np


def to_onehot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    labels = np.asarray(labels).astype(int).reshape(-1)
    out = np.zeros((labels.shape[0], n_classes), dtype=float)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def accuracy_cls(pred: np.ndarray, y: np.ndarray) -> float:
    pred = np.asarray(pred).reshape(-1)
    y = np.asarray(y).reshape(-1)
    return float(np.mean(pred == y))


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    pred = np.asarray(pred).reshape(-1)
    y = np.asarray(y).reshape(-1)
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def seeded_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
