"""Adapter returned by ``adapt`` — decouples a fitted learner from its maths."""

from __future__ import annotations

import numpy as np


class Predictor:
    def __init__(self, predict_fn, proba_fn=None, kind: str = "cls"):
        self._predict = predict_fn
        self._proba = proba_fn if proba_fn is not None else predict_fn
        self.kind = kind

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._predict(x))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._proba(x))
