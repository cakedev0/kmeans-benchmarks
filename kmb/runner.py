"""The runner: expands a config into jobs, interleaves them, runs each in its environment.

Only the standard library is used here. Each job (one implementation, one setting, all
seeds) runs in a fresh ``pixi run -e <env> python -m kmb.worker`` subprocess, which
appends one JSON line per fit to ``results/<config name>.jsonl``. Fits already in the file
(same machine, dataset, k, implementation, settings and seed) are skipped, so a run can be
interrupted and resumed.

Within one comparison (dataset, k), jobs go round-robin over the implementations, rotating
which one goes first in each round, so that slow drifts of the machine (GPU power cap,
thermals) spread over all contenders.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from kmb import impls
from kmb.config import Config, canonical, interleave, jobs_by_comparison

REPO = Path(__file__).resolve().parent.parent


def _fit_key(machine, dataset, k, impl, params, seed, threads):
    return (machine, dataset, int(k), impl, canonical(params), int(seed), int(threads or 0))


def done_keys(path: Path, retry_errors: bool) -> set:
    keys = set()
    if not path.exists():
        return keys
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line cut by an interrupted write
        if r.get("status") == "ok" or not retry_errors:
            keys.add(_fit_key(r["machine"], r["dataset"], r["k"], r["impl"], r["params"],
                              r["seed"], r.get("threads")))
    return keys


def _clean_env(threads: int) -> dict:
    """The environment of the worker: no leftovers of an outer pixi/conda activation, and
    explicit thread counts for every threading layer."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PIXI_", "CONDA_", "PYTHON")) and k not in ("VIRTUAL_ENV",)}
    envs_dir = str(REPO / ".pixi" / "envs")
    env["PATH"] = os.pathsep.join(p for p in os.environ.get("PATH", "").split(os.pathsep)
                                  if not p.startswith(envs_dir))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMBA_NUM_THREADS"):
        env[var] = str(threads)
    env["NVIDIA_TF32_OVERRIDE"] = "0"
    env["CUPY_TF32"] = "0"
    return env


def _fmt(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m" if seconds >= 3600 else \
        f"{seconds // 60}m{seconds % 60:02d}s"


def run(cfg: Config, *, impl_filter=None, dry_run=False, max_rounds=None,
        retry_errors=False, pixi="pixi"):
    results = REPO / cfg.results_path
    results.parent.mkdir(parents=True, exist_ok=True)
    log_dir = results.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    done = done_keys(results, retry_errors)

    plan = []
    for dataset, k, per_impl in jobs_by_comparison(cfg, impl_filter):
        jobs = interleave(per_impl)
        if max_rounds is not None:
            n_impl = len(per_impl)
            jobs = jobs[:max_rounds * n_impl]
        for job in jobs:
            job.seeds = [s for s in job.seeds if _fit_key(cfg.machine, dataset.label, k, job.impl,
                                                          job.params, s, cfg.threads) not in done]
            if job.seeds:
                plan.append(job)

    n_fits = sum(len(j.seeds) for j in plan)
    print(f"{cfg.name}: {len(plan)} jobs, {n_fits} fits to run on machine {cfg.machine!r} "
          f"with {cfg.threads} threads "
          f"({len(done)} fits already in {cfg.results_path})", flush=True)
    if dry_run:
        for job in plan:
            print(f"  {job.dataset.label} k={job.k} {job.impl:18s} {canonical(job.params)} "
                  f"seeds={job.seeds}")
        return

    started = time.perf_counter()
    job_times = defaultdict(list)
    remaining = defaultdict(int)
    for job in plan:
        remaining[job.impl] += 1
    log_path = log_dir / f"{cfg.name}.log"
    for i, job in enumerate(plan, 1):
        env_name = impls.env_of(job.impl)
        payload = {
            "run": cfg.name, "role": cfg.role, "machine": cfg.machine,
            "dataset": job.dataset.name, "n": job.dataset.n, "dataset_label": job.dataset.label,
            "k": job.k, "impl": job.impl, "params": job.params, "seeds": job.seeds,
            "threads": cfg.threads, "results": str(results),
        }
        header = (f"[{i}/{len(plan)}] {job.dataset.label} k={job.k} {job.impl} "
                  f"{canonical(job.params)} seeds={job.seeds}")
        print(header, flush=True)
        t0 = time.perf_counter()
        cmd = [pixi, "run", "--frozen", "--manifest-path", str(REPO / "pixi.toml"), "-e",
               env_name, "python", "-m", "kmb.worker"]
        status = "ok"
        with open(log_path, "a") as log:
            log.write(f"\n{time.strftime('%Y-%m-%dT%H:%M:%S')} {header}\n")
            log.flush()
            try:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=log,
                                        stderr=subprocess.PIPE, text=True, cwd=REPO,
                                        env=_clean_env(cfg.threads))
                _, err = proc.communicate(json.dumps(payload), timeout=cfg.timeout_s)
                if proc.returncode != 0:
                    status = f"exit code {proc.returncode}"
            except subprocess.TimeoutExpired:
                proc.kill()
                _, err = proc.communicate()
                status = f"timeout after {cfg.timeout_s:.0f}s"
            log.write(err or "")
        for line in (err or "").splitlines():
            if line.startswith("  "):
                print(line, flush=True)
        if status != "ok":
            print(f"  FAILED ({status}); stderr tail:\n" + "\n".join((err or "").splitlines()[-15:]),
                  flush=True)
            # Record the fits the worker did not write, so the failure is visible in results.
            written = done_keys(results, retry_errors=False)
            for seed in job.seeds:
                key = _fit_key(cfg.machine, job.dataset.label, job.k, job.impl, job.params, seed,
                               cfg.threads)
                if key not in written:
                    with open(results, "a") as f:
                        f.write(json.dumps({
                            "schema": 1, "run": cfg.name, "role": cfg.role,
                            "machine": cfg.machine, "dataset": job.dataset.label, "k": job.k,
                            "impl": job.impl, "env": env_name,
                            "device": impls.device_of(job.impl), "params": job.params,
                            "threads": cfg.threads,
                            "seed": seed, "status": "error", "error": status,
                            "stderr_tail": (err or "")[-4000:],
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
        elapsed = time.perf_counter() - t0
        job_times[job.impl].append(elapsed)
        remaining[job.impl] -= 1
        eta = sum(remaining[name] * sum(t) / len(t) for name, t in job_times.items())
        unknown = [name for name in remaining if remaining[name] and name not in job_times]
        print(f"  job {elapsed:.1f}s | elapsed {_fmt(time.perf_counter() - started)} | "
              f"ETA {_fmt(eta)}" + (f" + unmeasured {unknown}" if unknown else ""), flush=True)
    print(f"done in {_fmt(time.perf_counter() - started)}; results in {cfg.results_path}")


if __name__ == "__main__":
    sys.exit("use: python -m kmb run <config.toml>")
