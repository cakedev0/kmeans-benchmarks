"""scikit-learn ``KMeans`` (Lloyd or Elkan) and ``MiniBatchKMeans``, on the CPU.

``tol`` is relative to the mean feature variance in scikit-learn (unlike cuML and FAISS).
"""

from __future__ import annotations

from kmb.impls.base import FitOutput, Impl


class _Sklearn(Impl):
    DISTRIBUTIONS = ("scikit-learn", "numpy", "scipy", "threadpoolctl")
    WARMUP_ROWS = 100_000   # no JIT: a subset is enough to pay imports and first calls

    def setup(self):
        import sklearn  # noqa: F401
        from threadpoolctl import threadpool_info, threadpool_limits

        self._limits = threadpool_limits(limits=self.threads)
        self._pools = threadpool_info()

    def versions_extra(self):
        return {"threadpools": [{k: p.get(k) for k in ("internal_api", "num_threads", "version")}
                                for p in self._pools]}


class SklearnKMeans(_Sklearn):
    DEFAULTS = {"init": "k-means++", "n_init": 1, "max_iter": 300, "tol": 1e-4}

    def __init__(self, algorithm):
        self.algorithm = algorithm

    def fit(self, X, k, params, seed):
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, algorithm=self.algorithm, random_state=seed, **params).fit(X)
        n_iter = int(km.n_iter_)
        return FitOutput(km.cluster_centers_, n_iter=n_iter,
                         n_iter_total=n_iter if params["n_init"] == 1 else None,
                         epochs=n_iter if params["n_init"] == 1 else None,
                         lib_inertia=float(km.inertia_) / X.shape[0])


class SklearnMiniBatch(_Sklearn):
    WARMUP_OVERRIDES = {"max_iter": 1, "n_init": 1}
    DEFAULTS = {"init": "k-means++", "n_init": 1, "max_iter": 100, "batch_size": 1024,
                "tol": 0.0, "max_no_improvement": 10, "reassignment_ratio": 0.01}

    def fit(self, X, k, params, seed):
        from sklearn.cluster import MiniBatchKMeans

        km = MiniBatchKMeans(n_clusters=k, random_state=seed, **params).fit(X)
        steps = int(km.n_steps_)
        return FitOutput(km.cluster_centers_, n_iter=steps,
                         epochs=steps * params["batch_size"] / X.shape[0],
                         lib_inertia=float(km.inertia_) / X.shape[0],
                         extra={"n_epochs_run": int(km.n_iter_)})
