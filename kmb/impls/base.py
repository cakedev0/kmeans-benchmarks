"""Common interface of the implementation adapters.

Library imports happen in :meth:`Impl.setup`, never at module import time, so the runner
can read ``DEFAULTS`` from any environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FitOutput:
    centers: object                    # (k, d) array on the fit's device or host
    n_iter: int | None = None          # iterations of the returned solution (library's count)
    n_iter_total: int | None = None    # iterations over all inits, when known
    epochs: float | None = None        # passes over the full data, all inits, when known
    lib_inertia: float | None = None   # library's own objective, mean per sample (not trusted)
    extra: dict = field(default_factory=dict)


class Impl:
    #: parameters every fit gets unless the config overrides them
    DEFAULTS: dict = {}
    #: distributions whose versions are recorded with every fit
    DISTRIBUTIONS: tuple = ()
    #: "fp32", or the reduced-precision category ("bf16", "tf32", ...)
    precision = "fp32"

    name: str
    device: str
    env: str
    threads: int

    def setup(self):
        """Import the library, set thread counts and precision modes."""

    def to_device(self, X):
        """Move the host float32 array where the fit expects it (device memory for GPUs)."""
        return X

    def sync(self):
        """Wait for queued device work (no-op on CPU)."""

    def release(self):
        """Free cached memory pools before a fit, so peak memory is measured from scratch."""

    def to_host(self, centers):
        return centers

    def fit(self, X, k: int, params: dict, seed: int) -> FitOutput:
        raise NotImplementedError

    def versions_extra(self) -> dict:
        return {}

    #: budget overrides for the untimed warm-up fit (same data, same other settings)
    WARMUP_OVERRIDES = {"max_iter": 2, "n_init": 1, "niter": 2, "nredo": 1}
    #: rows of the warm-up fit; None for the full data (needed when JIT compilation or
    #: autotuning depends on the number of rows). A subset suits libraries without JIT.
    WARMUP_ROWS: int | None = None

    def warmup_params(self, params: dict) -> dict:
        """Settings of the warm-up fit, run once per process on the full data before the
        timed fits so that module loading, JIT compilation and shape-dependent autotuning
        are paid there (and reported as ``warmup_s``), not in the fit time."""
        return {**params, **{k: v for k, v in self.WARMUP_OVERRIDES.items() if k in params}}
