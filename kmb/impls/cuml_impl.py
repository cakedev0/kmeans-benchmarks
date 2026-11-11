"""cuML ``KMeans`` on the GPU.

``init="k-means++"`` is greedy k-means++, ``"scalable-k-means++"`` is k-means||, and
``"random"`` picks random rows. ``n_init`` is always explicit (cuML's ``"auto"`` runs a
single init for the k-means++ variants). ``n_iter_`` covers only the best init. The
stopping rule compares float32 inertias, and ``tol`` is not relative to the data variance.
cuML keeps empty clusters.
"""

from __future__ import annotations

from kmb.impls.base import FitOutput, Impl


class CumlKMeans(Impl):
    DEFAULTS = {"init": "scalable-k-means++", "n_init": 1, "max_iter": 300, "tol": 1e-4,
                "oversampling_factor": 2.0}
    DISTRIBUTIONS = ("cuml-cu13", "libcuml-cu13", "libcuvs-cu13", "rmm-cu13", "cupy-cuda13x",
                     "numpy", "nvidia-cublas")

    def setup(self):
        import cupy as cp
        import cuml  # noqa: F401

        self.cp = cp

    def to_device(self, X):
        return self.cp.asarray(X)

    def sync(self):
        self.cp.cuda.Device().synchronize()

    def release(self):
        self.cp.get_default_memory_pool().free_all_blocks()

    def to_host(self, centers):
        return self.cp.asnumpy(centers)

    def fit(self, X, k, params, seed):
        from cuml.cluster import KMeans

        km = KMeans(n_clusters=k, random_state=seed, output_type="cupy", **params)
        km.fit(X)
        n_iter = int(km.n_iter_)
        return FitOutput(km.cluster_centers_, n_iter=n_iter,
                         n_iter_total=n_iter if params["n_init"] == 1 else None,
                         epochs=n_iter if params["n_init"] == 1 else None,
                         lib_inertia=float(km.inertia_) / X.shape[0])
