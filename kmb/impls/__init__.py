"""Registry of implementations: name -> (environment, device, module:class).

Each adapter subclasses :class:`kmb.impls.base.Impl`. The worker imports only the adapter it
runs, inside the pixi environment named here, so libraries never share a process.
"""

from __future__ import annotations

import importlib

# name: (pixi environment, device, "module:Class", extra constructor kwargs)
REGISTRY = {
    "sklearn-lloyd": ("cpu", "cpu", "kmb.impls.sklearn_impl:SklearnKMeans", {"algorithm": "lloyd"}),
    "sklearn-elkan": ("cpu", "cpu", "kmb.impls.sklearn_impl:SklearnKMeans", {"algorithm": "elkan"}),
    "sklearn-minibatch": ("cpu", "cpu", "kmb.impls.sklearn_impl:SklearnMiniBatch", {}),
    "faiss-cpu": ("cpu", "cpu", "kmb.impls.faiss_impl:FaissKMeans", {"gpu": False}),
    "faiss-gpu": ("faiss-gpu", "gpu", "kmb.impls.faiss_impl:FaissKMeans", {"gpu": True}),
    "emeans-cpu": ("cpu", "cpu", "kmb.impls.emeans_impl:EmeansCPU", {}),
    "emeans-cpu-cython": ("cpu", "cpu", "kmb.impls.emeans_impl:EmeansCPUCython", {}),
    "emeans-gpu": ("torch", "gpu", "kmb.impls.emeans_impl:EmeansGPU", {"bf16": False}),
    "emeans-gpu-bf16": ("torch", "gpu", "kmb.impls.emeans_impl:EmeansGPU", {"bf16": True}),
    "cuml": ("cuml", "gpu", "kmb.impls.cuml_impl:CumlKMeans", {}),
    "cupy": ("cupy", "gpu", "kmb.impls.cupy_impl:CupyKMeans", {"fused": False}),
    "cupy-fused": ("cupy", "gpu", "kmb.impls.cupy_impl:CupyKMeans", {"fused": True}),
}


def env_of(name: str) -> str:
    return REGISTRY[name][0]


def device_of(name: str) -> str:
    return REGISTRY[name][1]


def create(name: str, threads: int):
    env, device, target, kwargs = REGISTRY[name]
    module, cls = target.split(":")
    impl = getattr(importlib.import_module(module), cls)(**kwargs)
    impl.name = name
    impl.device = device
    impl.env = env
    impl.threads = threads
    return impl


def defaults_of(name: str) -> dict:
    """Default parameters, read without importing the library (for config validation)."""
    env, device, target, kwargs = REGISTRY[name]
    module, cls = target.split(":")
    klass = getattr(importlib.import_module(module), cls)
    return dict(klass.DEFAULTS)
