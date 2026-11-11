"""FAISS ``Kmeans`` (Lloyd with random init), on the CPU or the GPU.

FAISS subsamples by default (``max_points_per_centroid=256``, so 65,536 points at k=256).
Here ``max_points_per_centroid=None`` (the default of this adapter) trains on all the data;
give a number to sweep the subsampling deliberately. ``niter`` is a fixed iteration count
(no tolerance), ``nredo`` is the number of inits. ``obj`` is measured on the training
subsample, which is why the shared evaluator recomputes the inertia on all the data.

On the GPU, FAISS's training loop runs on the host: the data stays in host memory and
only the nearest-centroid search runs on the GPU, so its fit time includes the transfers
of every iteration. That is how ``faiss.Kmeans(gpu=True)`` works, not a choice of ours.
"""

from __future__ import annotations

from kmb.impls.base import FitOutput, Impl


class FaissKMeans(Impl):
    DEFAULTS = {"niter": 25, "nredo": 1, "max_points_per_centroid": None}

    def __init__(self, gpu):
        self.gpu = gpu
        self.DISTRIBUTIONS = ("faiss-gpu" if gpu else "faiss-cpu", "libfaiss", "faiss", "numpy")

    def setup(self):
        import faiss

        self.faiss = faiss
        faiss.omp_set_num_threads(self.threads)
        if self.gpu and faiss.get_num_gpus() < 1:
            raise RuntimeError("faiss sees no GPU")

    def versions_extra(self):
        return {"faiss_compile_options": self.faiss.get_compile_options()}

    def fit(self, X, k, params, seed):
        n, d = X.shape
        mppc = params["max_points_per_centroid"]
        if mppc is None:
            mppc = n // k + 1
        km = self.faiss.Kmeans(d, k, niter=params["niter"], nredo=params["nredo"], seed=seed,
                               gpu=self.gpu, max_points_per_centroid=mppc,
                               min_points_per_centroid=1, verbose=False)
        km.train(X)
        n_train = min(n, mppc * k)
        total = params["niter"] * params["nredo"]
        return FitOutput(km.centroids, n_iter=params["niter"], n_iter_total=total,
                         epochs=total * n_train / n,
                         lib_inertia=float(km.obj[-1]) / n_train if len(km.obj) else None,
                         extra={"n_train": n_train})
