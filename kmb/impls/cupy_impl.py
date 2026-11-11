"""The CuPy k-means of ogrisel/cursor-sandbox#22: array operations, or fused JIT kernels.

``init="k-means++"`` runs ``kmeans_cupy`` (greedy k-means++ then Lloyd, ``n_init`` times).
``init="random"`` picks ``k`` distinct random rows per init and runs the same Lloyd
(``lloyd_cupy`` or ``lloyd_cupy_fused``); the branch itself only ships k-means++.
``tol`` follows scikit-learn (relative to the mean feature variance).

Each fit runs in its own process with CuPy's default memory pool (never with cuML's RMM
allocator, which made this code up to 2.6x slower).
"""

from __future__ import annotations

from kmb.impls.base import FitOutput, Impl


class CupyKMeans(Impl):
    DEFAULTS = {"init": "k-means++", "n_init": 1, "max_iter": 300, "tol": 1e-4}
    DISTRIBUTIONS = ("cupy-cuda13x", "numpy", "nvidia-cuda-nvrtc", "nvidia-cublas")

    def __init__(self, fused):
        self.fused = fused

    def setup(self):
        import cupy as cp

        from kmb import vendor

        vendor.ensure_importable()
        from kmeans_lloyd import cupy_lloyd

        self.cp = cp
        self.mod = cupy_lloyd

    def versions_extra(self):
        from kmb import vendor

        return {"kmeans_lloyd": f"cakedev0/cursor-sandbox@{vendor.COMMIT[:7]}"}

    def to_device(self, X):
        return self.cp.asarray(X)

    def sync(self):
        self.cp.cuda.Device().synchronize()

    def release(self):
        self.cp.get_default_memory_pool().free_all_blocks()

    def to_host(self, centers):
        return self.cp.asnumpy(centers)

    def fit(self, X, k, params, seed):
        p = dict(params)
        init = p.pop("init")
        if init == "k-means++":
            best, runs = self.mod.kmeans_cupy(X, k, fused=self.fused, random_state=seed, **p)
        elif init == "random":
            best, runs = self._fit_random(X, k, seed, **p)
        else:
            raise ValueError(f"unknown init {init!r}")
        centers, _, inertia, n_iter = best
        total = int(sum(r[1] for r in runs))
        return FitOutput(centers, n_iter=int(n_iter), n_iter_total=total, epochs=total,
                         lib_inertia=float(inertia) / X.shape[0])

    def _fit_random(self, X, k, seed, n_init, max_iter, tol):
        import numpy as np

        cp = self.cp
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        augmented = None
        if not self.fused:
            augmented = cp.concatenate([X, cp.ones((n, 1), dtype=X.dtype)], axis=1)
        best, runs = None, []
        for _ in range(n_init):
            rows = cp.asarray(np.sort(rng.choice(n, k, replace=False)))
            centers = X[rows]
            if self.fused:
                result = self.mod.lloyd_cupy_fused(X, centers, max_iter=max_iter, tol=tol)
            else:
                result = self.mod.lloyd_cupy(X, centers, max_iter=max_iter, tol=tol,
                                             augmented=augmented)
            runs.append((result[2], result[3]))
            if best is None or result[2] < best[2]:
                best = result
        return best, runs
