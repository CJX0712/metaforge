"""Tiny numpy MLP with hand-written forward / backward.

Why hand-rolled? Meta-learning (MAML/Reptile) needs to copy, perturb and
re-optimise the *same* parameter tensor many times per iteration, and to read
raw gradients. A tiny transparent MLP keeps that control explicit and makes the
gradient-check invariant trivial to assert.

Conventions
-----------
* The last layer is always treated as **linear** in backprop. Any non-linear
  output activation (e.g. softmax) is absorbed into the loss-gradient passed to
  :meth:`backward`, so the chain rule stays correct and simple.
* Parameters are a dict ``{"W0","b0","W1","b1",...}``; gradients mirror it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["MLP", "mse_loss", "ce_loss", "grad_step", "clip_grad", "num_params"]


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def mse_loss(out: np.ndarray, y: np.ndarray):
    """Returns (scalar loss, dL/d out). ``out`` linear."""
    y = y.reshape(out.shape)
    loss = float(np.mean((out - y) ** 2))
    d = 2.0 * (out - y) / y.shape[0]
    return loss, d


def ce_loss(out_softmax: np.ndarray, y_onehot: np.ndarray):
    """Returns (scalar loss, dL/d pre-softmax). ``out_softmax`` already softmaxed."""
    eps = 1e-12
    loss = float(-np.mean(np.sum(y_onehot * np.log(out_softmax + eps), axis=1)))
    d = (out_softmax - y_onehot) / y_onehot.shape[0]
    return loss, d


def num_params(params: dict) -> int:
    return int(sum(int(v.size) for v in params.values()))


def grad_step(params: dict, grads: dict, lr: float) -> dict:
    return {k: params[k] - lr * grads[k] for k in params}


def clip_grad(grads: dict[str, np.ndarray], max_norm: float) -> dict[str, np.ndarray]:
    """Clip the global gradient norm (standard SGD stabiliser).

    When the raw gradient norm exceeds ``max_norm`` every entry is scaled by
    ``max_norm / norm``. This keeps meta-learning updates finite and bounded even
    under large ``lr`` / batch-scale gradients, preventing the embedding weights
    from diverging to ``inf`` in a single meta-iteration. Identical to the clip
    used by ProtoNet, so all metalearners share one well-tested primitive.
    """
    norm = float(np.sqrt(sum(np.sum(g**2) for g in grads.values())))
    if norm > max_norm and norm > 0.0:
        scale = max_norm / norm
        return {k: g * scale for k, g in grads.items()}
    return grads


class MLP:
    """Minimal multi-layer perceptron.

    ``layer_sizes`` e.g. ``[in, hidden1, hidden2, out]``. Hidden layers use
    ReLU; the output layer is linear (or softmax for convenience of the
    forward ``predict_proba`` display only).
    """

    def __init__(
        self,
        layer_sizes: list[int],
        out_activation: str = "linear",
        seed: int = 0,
        w_scale: float = 1.0,
    ):
        self.layer_sizes = list(layer_sizes)
        self.out_activation = out_activation
        rng = np.random.default_rng(seed)
        self.params: dict[str, np.ndarray] = {}
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            # He (Kaiming) init for ReLU hidden layers keeps pre-activations at
            # unit variance so gradients stay healthy; a smaller scale for the
            # output layer avoids saturating the softmax / exploding the last step.
            if i < len(layer_sizes) - 2:
                s = np.sqrt(2.0 / fan_in)
            else:
                s = np.sqrt(1.0 / fan_in)
            s = s * w_scale  # w_scale lets tests shrink init if needed
            self.params[f"W{i}"] = (rng.standard_normal((fan_in, layer_sizes[i + 1])) * s).astype(
                float
            )
            self.params[f"b{i}"] = np.zeros(layer_sizes[i + 1], dtype=float)
        self._cache: dict | None = None

    # ---- parameter plumbing ------------------------------------------------
    def get_params(self) -> dict:
        return {k: v.copy() for k, v in self.params.items()}

    def set_params(self, params: dict) -> None:
        for k, v in params.items():
            self.params[k] = np.asarray(v, dtype=float)

    # ---- forward ----------------------------------------------------------
    def forward(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        L = len(self.layer_sizes) - 1
        zs: list[np.ndarray] = []
        acts: list[np.ndarray] = []
        a = X
        for i in range(L):
            z = a @ self.params[f"W{i}"] + self.params[f"b{i}"]
            zs.append(z)
            if i < L - 1:
                a = _relu(z)
            else:
                a = _softmax(z, axis=-1) if self.out_activation == "softmax" else z
            acts.append(a)
        self._cache = {"X": X, "zs": zs, "acts": acts}
        return acts[-1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    # ---- backward ---------------------------------------------------------
    def backward(self, dL_da: np.ndarray, cache: dict | None = None) -> dict:
        """Gradient w.r.t. parameters given dL/d( last activation ).

        The last layer is treated as linear; the caller must pass the
        loss-gradient already combined with the output activation (see
        :func:`mse_loss` / :func:`ce_loss`).
        """
        cache = cache or self._cache
        if cache is None:
            raise RuntimeError("call forward() before backward()")
        X = cache["X"]
        zs = cache["zs"]
        acts = cache["acts"]
        L = len(self.layer_sizes) - 1
        grads: dict[str, np.ndarray] = {}
        da = np.asarray(dL_da, dtype=float)
        for i in reversed(range(L)):
            z = zs[i]
            a_in = X if i == 0 else acts[i - 1]
            if i == L - 1:
                dz = da
            else:
                dz = da * _relu_grad(z)
            # ``dL_da`` handed in by the loss already carries the 1/B batch-mean
            # factor (see :func:`mse_loss` / :func:`ce_loss`), so the weight /
            # bias gradients below must NOT divide by ``B`` again.
            grads[f"W{i}"] = a_in.T @ dz
            grads[f"b{i}"] = np.sum(dz, axis=0)
            if i > 0:
                da = dz @ self.params[f"W{i}"].T
        return grads
