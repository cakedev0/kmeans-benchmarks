"""The shared quality metric: mean squared distance of every sample to its nearest center.

Every implementation is scored by this function, from the centers it returns, never by its
own ``inertia_``. The nearest center is found with a float32 GEMM
(``|c|^2 - 2 x.c``, argmin over the centers), then the squared distance of each sample to
that center is recomputed exactly in float64 and summed in float64.

The float32 search can only pick a wrong center when two centers are within the float32
rounding error of the GEMM of each other (about ``1e-7 * |x|^2``); the distance it then
reports differs from the true minimum by at most that much, a relative error below 1e-6
per affected sample, so the result is exact for the purpose of ranking runs. Pass
``exact=True`` to do the search in float64 as well (slow, used by the tests).

The same code runs on NumPy, CuPy and PyTorch arrays, so GPU workers evaluate on the
device that already holds the data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InertiaResult:
    inertia: float          # mean squared distance to the nearest center
    n_empty: int            # centers that no sample is closest to
    n_nonfinite: int        # centers with NaN or infinite coordinates (excluded)


class _Numpy:
    def __init__(self):
        self.xp = np

    def asarray(self, a, like_float64=False):
        return np.asarray(a, dtype=np.float64 if like_float64 else np.float32)

    def to_float64(self, a):
        return a.astype(np.float64)

    def argmin(self, a):
        return np.argmin(a, axis=1)

    def take(self, a, idx):
        return a[idx]

    def scalar(self, a):
        return float(a)

    def bincount(self, labels, k):
        return np.bincount(labels, minlength=k)

    def to_numpy(self, a):
        return np.asarray(a)


class _Cupy(_Numpy):
    def __init__(self):
        import cupy

        self.xp = cupy

    def asarray(self, a, like_float64=False):
        return self.xp.asarray(a, dtype=self.xp.float64 if like_float64 else self.xp.float32)

    def argmin(self, a):
        return self.xp.argmin(a, axis=1)

    def bincount(self, labels, k):
        return self.xp.bincount(labels, minlength=k)

    def to_numpy(self, a):
        return self.xp.asnumpy(a)


class _Torch:
    def __init__(self, device):
        import torch

        self.torch = torch
        self.device = device

    def asarray(self, a, like_float64=False):
        dtype = self.torch.float64 if like_float64 else self.torch.float32
        if not isinstance(a, self.torch.Tensor):
            a = self.torch.as_tensor(np.asarray(a))
        return a.to(device=self.device, dtype=dtype)

    def to_float64(self, a):
        return a.to(self.torch.float64)

    def argmin(self, a):
        return self.torch.argmin(a, dim=1)

    def take(self, a, idx):
        return a[idx]

    def scalar(self, a):
        return float(a.item())

    def bincount(self, labels, k):
        return self.torch.bincount(labels, minlength=k)

    def to_numpy(self, a):
        return a.detach().cpu().numpy()


def _backend_for(X):
    module = type(X).__module__
    if module.startswith("cupy"):
        return _Cupy()
    if module.startswith("torch"):
        return _Torch(X.device)
    return _Numpy()


def evaluate_inertia(X, centers, *, exact=False, chunk_bytes=256 << 20) -> InertiaResult:
    """Mean squared distance of the rows of ``X`` to their nearest row of ``centers``.

    ``X`` is a NumPy, CuPy or PyTorch array of shape (n, d); ``centers`` is any array of
    shape (k, d) (moved to the device of ``X``). Centers with non-finite coordinates are
    dropped (and counted) rather than poisoning the result.
    """
    be = _backend_for(X)
    centers = np.asarray(centers if isinstance(centers, np.ndarray) else _to_host(centers))
    if centers.ndim != 2 or centers.shape[1] != X.shape[1]:
        raise ValueError(f"centers of shape {centers.shape} do not match X of shape {X.shape}")
    finite = np.all(np.isfinite(centers), axis=1)
    n_nonfinite = int((~finite).sum())
    centers = centers[finite].astype(np.float64)
    n_samples, n_features = int(X.shape[0]), int(X.shape[1])
    k = centers.shape[0]

    C64 = be.asarray(centers, like_float64=True)
    C_search = C64 if exact else be.asarray(centers.astype(np.float32))
    C_norm = (C_search * C_search).sum(1)
    C_search_T = C_search.T

    # Bound the float64 copy of a chunk and the (chunk, k) score matrix.
    row_bytes = 8 * max(n_features, k)
    chunk = max(1024, min(n_samples, chunk_bytes // row_bytes))
    total = 0.0
    counts = None
    for start in range(0, n_samples, chunk):
        Xc = X[start:start + chunk]
        Xs = be.to_float64(Xc) if exact else Xc
        if not exact and Xs.dtype != C_search.dtype:
            Xs = be.asarray(Xs)
        scores = C_norm[None, :] - 2 * (Xs @ C_search_T)
        labels = be.argmin(scores)
        diff = be.to_float64(Xc) - be.take(C64, labels)
        total += be.scalar((diff * diff).sum())
        c = be.bincount(labels, k)
        counts = c if counts is None else counts + c
    n_empty = int((be.to_numpy(counts) == 0).sum()) + n_nonfinite
    return InertiaResult(inertia=total / n_samples, n_empty=n_empty, n_nonfinite=n_nonfinite)


def _to_host(a):
    module = type(a).__module__
    if module.startswith("cupy"):
        import cupy

        return cupy.asnumpy(a)
    if module.startswith("torch"):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def inertia_reference(X, centers) -> float:
    """Brute-force float64 NumPy inertia (mean over samples), for tests."""
    X = np.asarray(X, dtype=np.float64)
    C = np.asarray(centers, dtype=np.float64)
    d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    return float(d2.min(axis=1).mean())
