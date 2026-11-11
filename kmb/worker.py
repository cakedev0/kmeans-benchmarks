"""Runs the fits of one job (one implementation, one setting, several seeds) in its own
pixi environment, and appends one JSON line per fit to the results file as soon as the
fit is scored.

Launched by the runner as ``pixi run -e <env> python -m kmb.worker`` with the job as JSON
on stdin. Per process:

1. ``import_s``: import the library (``Impl.setup``).
2. ``load_s``: read the cached ``.npy`` into host memory.
3. ``transfer_s``: copy it to the device (GPU implementations), synchronized.
4. ``warmup_s``: one short untimed fit on the full data (see ``Impl.warmup_params``), so
   that module loading and JIT compilation are not in the fit time.
5. For each seed: free memory pools, then time ``fit`` (synchronized before and after,
   initialization included), then copy the centers to the host and score them with the
   shared evaluator, on the device that holds the data.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

from kmb import datasets, impls, sysinfo
from kmb.inertia import evaluate_inertia


def _append(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record, default=_json_default) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
    except ImportError:
        pass
    return str(o)


def run(job: dict):
    t_start = time.perf_counter()
    impl = impls.create(job["impl"], threads=job["threads"])
    with sysinfo.Timer() as t_import:
        impl.setup()
    with sysinfo.Timer() as t_load:
        X = datasets.load(job["dataset"], job.get("n"))
    n, d = X.shape
    with sysinfo.Timer() as t_transfer:
        Xd = impl.to_device(X)
        impl.sync()
    params, k = job["params"], job["k"]

    gpu = impl.device == "gpu"
    sampler = sysinfo.GPUSampler() if gpu else None
    common = {
        "schema": 1,
        "run": job["run"],
        "role": job["role"],
        "machine": job["machine"],
        "dataset": job["dataset_label"],
        "n": n,
        "d": d,
        "k": k,
        "impl": impl.name,
        "env": impl.env,
        "device": impl.device,
        "precision": impl.precision,
        "params": params,
        "threads": job["threads"],
        "import_s": round(t_import.s, 4),
        "load_s": round(t_load.s, 4),
        "transfer_s": round(t_transfer.s, 4) if gpu else 0.0,
        "host": sysinfo.host_info(),
        "gpu": sysinfo.gpu_info() if gpu else None,
        "versions": {**sysinfo.versions(impl.DISTRIBUTIONS), **impl.versions_extra()},
    }

    try:
        with sysinfo.Timer() as t_warm:
            rows = impl.WARMUP_ROWS
            Xw = Xd if rows is None or rows >= n or rows < 4 * k else Xd[:rows]
            impl.fit(Xw, k, impl.warmup_params(params), job["seeds"][0])
            impl.sync()
        common["warmup_s"] = round(t_warm.s, 4)
    except Exception as exc:  # the timed fits would fail the same way
        for seed in job["seeds"]:
            _append(job["results"], {**common, "seed": seed, "status": "error",
                                     "error": f"warm-up: {exc!r}",
                                     "traceback": traceback.format_exc()[-4000:],
                                     "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
        return

    for seed in job["seeds"]:
        record = {**common, "seed": seed, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        try:
            impl.release()
            impl.sync()
            rss_before = sysinfo.rss_bytes()
            hwm_reset = sysinfo.reset_peak_rss()
            if sampler:
                sampler.start()
            t0 = time.perf_counter()
            out = impl.fit(Xd, k, params, seed)
            impl.sync()
            fit_s = time.perf_counter() - t0
            if sampler:
                record.update(sampler.stop())
            else:
                peak = sysinfo.peak_rss_bytes()
                record["peak_mem_bytes"] = max(peak - rss_before, 0) if hwm_reset else None
                record["peak_mem_method"] = ("VmHWM after reset minus RSS before the fit "
                                             "(kernel RSS counters are approximate, +-~20 MB)")
            with sysinfo.Timer() as t_host:
                centers = impl.to_host(out.centers)
            with sysinfo.Timer() as t_eval:
                # Score on the device that holds the data (the same code for every backend).
                Xe = Xd if type(Xd).__module__.split(".")[0] in ("cupy", "torch") else X
                score = evaluate_inertia(Xe, centers)
            if centers.shape != (k, d):
                raise ValueError(f"centers of shape {centers.shape}, expected {(k, d)}")
            record.update({
                "status": "ok",
                "fit_s": round(fit_s, 6),
                "inertia": score.inertia,
                "n_empty": score.n_empty,
                "n_nonfinite": score.n_nonfinite,
                "lib_inertia": out.lib_inertia,
                "n_iter": out.n_iter,
                "n_iter_total": out.n_iter_total,
                "epochs": out.epochs,
                "extra": out.extra,
                "to_host_s": round(t_host.s, 4),
                "eval_s": round(t_eval.s, 4),
            })
            print(f"  {impl.name} seed={seed} fit={fit_s:.3f}s inertia={score.inertia:.6g} "
                  f"n_iter={out.n_iter} empty={score.n_empty}", file=sys.stderr, flush=True)
        except Exception as exc:
            if sampler and sampler._thread is not None and sampler._thread.is_alive():
                sampler.stop()
            record.update({"status": "error", "error": repr(exc),
                           "traceback": traceback.format_exc()[-4000:]})
            print(f"  {impl.name} seed={seed} ERROR {exc!r}", file=sys.stderr, flush=True)
        _append(job["results"], record)
    print(f"  worker total {time.perf_counter() - t_start:.1f}s (import {t_import.s:.1f}, "
          f"load {t_load.s:.1f}, transfer {t_transfer.s:.2f}, warm-up {t_warm.s:.1f})",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    run(json.loads(sys.stdin.read()))
