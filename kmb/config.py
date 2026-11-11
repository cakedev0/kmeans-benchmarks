"""Benchmark configs (TOML) and their expansion into jobs.

A config names datasets, values of k, implementations and, per implementation, one or
more parameter grids. Each grid is expanded as a cartesian product, merged over the
implementation's defaults; the union of all grids (duplicates removed) is the sweep.
See ``configs/`` for examples. Only the standard library is needed here, so the runner
does not depend on any benchmark environment.
"""

from __future__ import annotations

import itertools
import json
import os
import socket
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from kmb import impls


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    n: int | None = None   # random subset of n rows (fixed seed), None for all

    @property
    def label(self) -> str:
        return self.name if self.n is None else f"{self.name}-n{self.n}"


@dataclass
class Job:
    dataset: DatasetSpec
    k: int
    impl: str
    params: dict
    seeds: list

    @property
    def params_key(self) -> str:
        return canonical(self.params)


@dataclass
class Config:
    name: str
    path: Path
    machine: str
    seeds: list
    threads: int
    timeout_s: float
    role: str
    datasets: list               # list of (DatasetSpec, [k, ...])
    grids: dict                  # impl -> list of param dicts
    impl_datasets: dict = field(default_factory=dict)   # impl -> allowed dataset names
    impl_max_d: dict = field(default_factory=dict)      # impl -> max dimension

    @property
    def results_path(self) -> Path:
        return Path("results") / f"{self.name}.jsonl"


def canonical(params: dict) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def default_machine() -> str:
    return os.environ.get("KMB_MACHINE") or socket.gethostname()


def _as_list(v):
    return v if isinstance(v, list) else [v]


def expand_grid(grid: dict, defaults: dict) -> list:
    unknown = set(grid) - set(defaults)
    if unknown:
        raise ValueError(f"unknown parameters {sorted(unknown)}; known: {sorted(defaults)}")
    keys = list(grid)
    out = []
    for values in itertools.product(*(_as_list(grid[k]) for k in keys)):
        p = dict(defaults)
        p.update(zip(keys, values))
        # TOML has no null: "none" stands for None (e.g. FAISS max_points_per_centroid)
        out.append({k: (None if v == "none" else v) for k, v in p.items()})
    return out


def load(path, machine=None) -> Config:
    path = Path(path)
    raw = tomllib.loads(path.read_text())
    datasets = []
    for d in raw["datasets"]:
        datasets.append((DatasetSpec(d["name"], d.get("n")), [int(k) for k in _as_list(d["k"])]))
    grids, impl_datasets, impl_max_d = {}, {}, {}
    for name, spec in raw["impls"].items():
        if name not in impls.REGISTRY:
            raise ValueError(f"unknown implementation {name!r}; known: {sorted(impls.REGISTRY)}")
        defaults = impls.defaults_of(name)
        seen, points = set(), []
        for grid in spec.get("grid", [{}]):
            for p in expand_grid(grid, defaults):
                if canonical(p) not in seen:
                    seen.add(canonical(p))
                    points.append(p)
        grids[name] = points
        if "datasets" in spec:
            impl_datasets[name] = set(spec["datasets"])
        if "max_d" in spec:
            impl_max_d[name] = int(spec["max_d"])
    return Config(
        name=raw.get("name", path.stem),
        path=path,
        machine=machine or raw.get("machine") or default_machine(),
        seeds=[int(s) for s in raw.get("seeds", [0, 1, 2])],
        threads=int(raw.get("threads", os.cpu_count())),
        timeout_s=float(raw.get("timeout_s", 3600)),
        role=raw.get("role", "sweep"),
        datasets=datasets,
        grids=grids,
        impl_datasets=impl_datasets,
        impl_max_d=impl_max_d,
    )


DIMENSIONS = {"sift-128": 128, "glove-200": 200, "clip-512": 512, "susy": 18,
              "year-msd": 90, "blobs-3d": 3}


def jobs_by_comparison(cfg: Config, impl_filter=None):
    """Yield ``(dataset, k, {impl: [Job, ...]})`` for each comparison of the config."""
    for dataset, ks in cfg.datasets:
        for k in ks:
            per_impl = {}
            for impl, points in cfg.grids.items():
                if impl_filter and impl not in impl_filter:
                    continue
                if impl in cfg.impl_datasets and dataset.name not in cfg.impl_datasets[impl]:
                    continue
                if impl in cfg.impl_max_d and DIMENSIONS.get(dataset.name, 0) > cfg.impl_max_d[impl]:
                    continue
                per_impl[impl] = [Job(dataset, k, impl, p, list(cfg.seeds)) for p in points]
            yield dataset, k, per_impl


def interleave(per_impl: dict) -> list:
    """Round-robin over implementations, rotating which one goes first in each round."""
    names = list(per_impl)
    out = []
    n_rounds = max((len(v) for v in per_impl.values()), default=0)
    for r in range(n_rounds):
        rotated = names[r % len(names):] + names[:r % len(names)]
        for name in rotated:
            if r < len(per_impl[name]):
                out.append(per_impl[name][r])
    return out
