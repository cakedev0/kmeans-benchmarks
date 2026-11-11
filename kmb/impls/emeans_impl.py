"""e-means (FAISS ``contrib.e_means``, Meta FAIR, arXiv 2611.09501): mini-batch k-means with
exponential moving average updates and random init.

- ``emeans-cpu``: ``NumpyEmeans`` on host data, searches with ``faiss.knn`` in float32.
- ``emeans-cpu-cython``: the same, with its accumulation step in a compiled OpenMP kernel.
- ``emeans-gpu``: ``QuantizedEmeans(quantizer=None)`` on a CUDA float32 tensor: native FP32
  with the Triton kernels (``input_precision="ieee"``), the fair comparison.
- ``emeans-gpu-bf16``: ``Emeans`` on the same tensor, which converts FP32 input to BF16
  operands by default: the reduced-precision category, labeled as such.

``epochs`` (e-means' ``nepochs``, fractional allowed) is the main budget knob; batch size
and step size follow the paper's scaling laws unless ``batch_size`` / ``alpha0`` are set.
"""

from __future__ import annotations

from kmb.impls.base import FitOutput, Impl

_DEFAULTS = {"epochs": 5, "batch_law": "auto", "batch_size": None, "alpha0": None}
_RECIPE_KEYS = ("batch_size", "alpha0", "steps_per_pass", "assignment_backend")


def _output(em, centers, params):
    recipe = {key: em.recipe.get(key) for key in _RECIPE_KEYS if em.recipe}
    recipe = {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
              for k, v in recipe.items()}
    steps = recipe.get("steps_per_pass")
    return FitOutput(centers, n_iter=(int(round(steps * params["epochs"])) if steps else None),
                     epochs=float(params["epochs"]),
                     extra={"recipe": recipe, "n_splits": int(getattr(em, "n_splits", 0)),
                            "obj_last_pass": float(em.obj[-1]) if em.obj is not None
                            and len(em.obj) else None})


def _kwargs(params, seed):
    kw = {"nepochs": params["epochs"], "batch_law": params["batch_law"], "seed": seed}
    for key in ("batch_size", "alpha0"):
        if params[key] is not None:
            kw[key] = params[key]
    return kw


class EmeansCPU(Impl):
    DEFAULTS = _DEFAULTS
    DISTRIBUTIONS = ("faiss-cpu", "libfaiss", "numpy", "threadpoolctl")

    def setup(self):
        import faiss
        from faiss.contrib.e_means import NumpyEmeans
        from threadpoolctl import threadpool_limits

        faiss.omp_set_num_threads(self.threads)
        self._limits = threadpool_limits(limits=self.threads)
        self.cls = NumpyEmeans

    def fit(self, X, k, params, seed):
        em = self.cls(X.shape[1], k, **_kwargs(params, seed))
        centers, _ = em.train(X)
        return _output(em, centers, params)


class EmeansCPUCython(EmeansCPU):
    """``NumpyEmeans`` with its accumulation step in an OpenMP Cython kernel
    (``kmb/emeans_cpu.py``): same algorithm, same recipe, compiled step."""

    def setup(self):
        from threadpoolctl import threadpool_limits

        from kmb.emeans_cpu import FastNumpyEmeans

        # The parallelism is in the kernel's OpenMP loops; its sgemm calls must be serial.
        self._limits = threadpool_limits(limits=1, user_api="blas")
        self.cls = FastNumpyEmeans

    def fit(self, X, k, params, seed):
        em = self.cls(X.shape[1], k, n_threads=self.threads, **_kwargs(params, seed))
        centers, _ = em.train(X)
        return _output(em, centers, params)

    def versions_extra(self):
        return {"kernel": "kmb/_emeans_kernel.pyx (Cython, OpenMP)"}


class EmeansGPU(Impl):
    DEFAULTS = _DEFAULTS
    DISTRIBUTIONS = ("torch", "triton", "faiss-cpu", "numpy", "nvidia-cublas")

    def __init__(self, bf16):
        self.bf16 = bf16
        self.precision = "bf16" if bf16 else "fp32"

    def setup(self):
        import torch

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.set_num_threads(self.threads)
        from faiss.contrib.e_means import Emeans, QuantizedEmeans

        self.torch = torch
        self.Emeans, self.QuantizedEmeans = Emeans, QuantizedEmeans

    def to_device(self, X):
        return self.torch.from_numpy(X).cuda()

    def sync(self):
        self.torch.cuda.synchronize()

    def release(self):
        self.torch.cuda.empty_cache()

    def to_host(self, centers):
        return centers.detach().float().cpu().numpy()

    def fit(self, X, k, params, seed):
        kw = _kwargs(params, seed)
        if self.bf16:
            em = self.Emeans(X.shape[1], k, **kw)
        else:
            em = self.QuantizedEmeans(X.shape[1], k, quantizer=None, **kw)
        centers, _ = em.train(X)
        return _output(em, centers, params)
