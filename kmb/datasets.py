"""Dataset download, conversion to cached float32 ``.npy`` files, and loading.

Files live in ``$KMB_DATA_DIR`` (default: ``data/`` at the repository root): raw downloads
in ``raw/``, converted arrays as ``<name>.npy`` with a ``<name>.json`` sidecar that
records the shape, the source and the preprocessing.

Downloads use ``curl``: ann-benchmarks answers HTTP 403 to Python's ``urllib``.

Run ``python -m kmb data <name> ...`` (in the ``cpu`` environment, which has h5py and
pyarrow) to download and convert. Everything else only needs NumPy to load the cache.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    return Path(os.environ.get("KMB_DATA_DIR", REPO / "data"))


@dataclass
class Dataset:
    name: str
    shape: tuple
    source: str
    notes: str
    urls: list = field(default_factory=list)
    #: bumped when the conversion changes, so that stale cached arrays are rebuilt
    version: int = 1


ANN = "http://ann-benchmarks.com"
HF_CLIP = ("https://huggingface.co/datasets/Qdrant/wolt-food-clip-ViT-B-32-embeddings/resolve/"
           "main/data/part-{:02d}-wolt-clip-ViT-B-32.parquet")
UCI = "https://archive.ics.uci.edu/static/public"
ROAD_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00246/3D_spatial_network.txt"
ROAD_N, ROAD_COPIES, ROAD_JITTER_KM = 434_874, 10, 0.005

DATASETS = {
    d.name: d for d in [
        Dataset("sift-128", (1_000_000, 128), "ann-benchmarks sift-128-euclidean (train)",
                "SIFT1M descriptors as published (norms ~509).",
                [f"{ANN}/sift-128-euclidean.hdf5"]),
        Dataset("glove-200", (1_183_514, 200), "ann-benchmarks glove-200-angular (train)",
                "GloVe word vectors as published: unnormalized (norms ~6.2 +- 1.1).",
                [f"{ANN}/glove-200-angular.hdf5"]),
        Dataset("clip-512", (1_000_000, 512),
                "Qdrant/wolt-food-clip-ViT-B-32-embeddings, first 1M rows (parts 00-19)",
                "L2-normalized CLIP ViT-B/32 image embeddings; many exact duplicates "
                "(~23% of the first 200k rows), kept on purpose.",
                [HF_CLIP.format(i) for i in range(20)]),
        Dataset("susy", (5_000_000, 18), "UCI SUSY (id 279), label column dropped",
                "18 kinematic features of particle collisions, as published (already "
                "roughly normalized: feature std 0.2 to 1.0).",
                [f"{UCI}/279/susy.zip"], version=2),
        Dataset("year-msd", (515_345, 90), "UCI YearPredictionMSD (id 203), year dropped",
                "12 timbre averages and 78 timbre covariances, standardized per feature: "
                "raw feature std ranges from 4.4 to 1750, which would make Euclidean "
                "distances depend on a handful of covariance features.",
                [f"{UCI}/203/yearpredictionmsd.zip"]),
        Dataset("road-network", (ROAD_N * ROAD_COPIES, 3),
                "UCI 3D Road Network, North Jutland (id 246), OSM id dropped, 10x oversampled",
                "434,874 GPS points along roads (longitude, latitude, altitude), projected "
                "to local kilometers (equirectangular around the mean latitude, altitude "
                "in km), so that distances are physical: in raw units altitude in meters "
                "would weigh 30x more than longitude in degrees. Oversampled 10x: each "
                "point and 9 copies with isotropic Gaussian jitter of 5 m (about half a "
                "road's width; median nearest-neighbor spacing is 16 m), seed 0.",
                [ROAD_URL]),
    ]
}


def path(name: str) -> Path:
    return data_dir() / f"{name}.npy"


def _curl(url: str, dest: Path):
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}", flush=True)
    subprocess.run(["curl", "-fL", "--retry", "5", "--retry-delay", "5", "-C", "-",
                    "-sS", "-o", str(tmp), url], check=True)
    tmp.rename(dest)
    return dest


def _standardize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    return ((X - mean) / std).astype(np.float32)


def _from_hdf5(raw: Path) -> np.ndarray:
    import h5py

    with h5py.File(raw, "r") as f:
        return np.asarray(f["train"], dtype=np.float32)


def _clip(raws: list) -> np.ndarray:
    import pyarrow.parquet as pq

    parts = []
    for raw in raws:
        column = pq.read_table(raw, columns=["vector"]).column("vector")
        strings = [s.as_py() for s in column.cast(column.type.storage_type)] \
            if hasattr(column.type, "storage_type") else column.to_pylist()
        text = ",".join(s.strip()[1:-1] for s in strings)
        block = np.array(text.split(","), dtype=np.float32).reshape(len(strings), -1)
        parts.append(block)
    return np.concatenate(parts)


def _susy(raw: Path) -> np.ndarray:
    with zipfile.ZipFile(raw) as z, gzip.open(z.open("SUSY.csv.gz")) as f:
        import pyarrow.csv as pacsv

        table = pacsv.read_csv(f, read_options=pacsv.ReadOptions(autogenerate_column_names=True))
    return np.column_stack([table.column(i).to_numpy() for i in range(1, table.num_columns)])


def _year_msd(raw: Path) -> np.ndarray:
    import pyarrow.csv as pacsv

    with zipfile.ZipFile(raw) as z:
        table = pacsv.read_csv(z.open("YearPredictionMSD.txt"),
                               read_options=pacsv.ReadOptions(autogenerate_column_names=True))
    a = np.column_stack([table.column(i).to_numpy() for i in range(1, table.num_columns)])
    return _standardize(a)


def _road_network(raw: Path) -> np.ndarray:
    a = np.loadtxt(raw, delimiter=",", dtype=np.float64)[:, 1:]   # lon, lat, altitude (m)
    earth_km = 6371.0088
    lat0 = np.deg2rad(a[:, 1].mean())
    km = np.column_stack([np.deg2rad(a[:, 0] - a[:, 0].mean()) * earth_km * np.cos(lat0),
                          np.deg2rad(a[:, 1] - a[:, 1].mean()) * earth_km,
                          (a[:, 2] - a[:, 2].mean()) / 1000])
    rng = np.random.default_rng(0)
    out = np.repeat(km[:, None, :], ROAD_COPIES, axis=1)    # each point, then its copies
    out[:, 1:] += rng.normal(scale=ROAD_JITTER_KM, size=(len(km), ROAD_COPIES - 1, 3))
    return out.reshape(-1, 3)


def prepare(name: str, force=False) -> Path:
    """Download (if needed) and convert ``name`` to ``<data_dir>/<name>.npy``."""
    spec = DATASETS[name]
    out = path(name)
    if out.exists() and not force and _cached_version(name) == spec.version:
        return out
    raw_dir = data_dir() / "raw"
    t0 = time.perf_counter()
    raws = [_curl(url, raw_dir / Path(url).name) for url in spec.urls]
    if name in ("sift-128", "glove-200"):
        X = _from_hdf5(raws[0])
    elif name == "clip-512":
        X = _clip(raws)
    elif name == "susy":
        X = _susy(raws[0])
    elif name == "year-msd":
        X = _year_msd(raws[0])
    elif name == "road-network":
        X = _road_network(raws[0])
    else:
        raise KeyError(name)
    X = np.ascontiguousarray(X, dtype=np.float32)
    if X.shape != spec.shape:
        raise ValueError(f"{name}: expected shape {spec.shape}, got {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError(f"{name}: non-finite values")
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, X)
    tmp.rename(out)
    norms = np.linalg.norm(X[:100_000], axis=1)
    meta = {"name": name, "version": spec.version, "shape": list(X.shape), "dtype": "float32", "source": spec.source,
            "urls": spec.urls, "notes": spec.notes,
            "norm_mean_first_100k": float(norms.mean()), "norm_std_first_100k": float(norms.std()),
            "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "prepare_s": round(time.perf_counter() - t0, 1)}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{name}: {X.shape} -> {out} ({meta['prepare_s']} s, "
          f"norms {meta['norm_mean_first_100k']:.3g} +- {meta['norm_std_first_100k']:.3g})")
    return out


def _cached_version(name: str) -> int | None:
    sidecar = path(name).with_suffix(".json")
    if not sidecar.exists():
        return None
    return json.loads(sidecar.read_text()).get("version", 1)


def load(name: str, n: int | None = None, subset_seed: int = 0) -> np.ndarray:
    """Load the cached array into memory (float32, C-contiguous, writable), or a random
    subset of ``n`` rows. The array is copied out of the memory map, so that fits never
    pay page faults on the file.

    The subset is drawn without replacement with a fixed seed and kept in the original
    row order, so the same ``(name, n)`` always gives the same rows.
    """
    p = path(name)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `pixi run python -m kmb data {name}`")
    if _cached_version(name) != DATASETS[name].version:
        raise RuntimeError(f"{p} is from an older conversion: run "
                           f"`pixi run python -m kmb data {name}`")
    X = np.load(p, mmap_mode="r")
    if n is not None and n < X.shape[0]:
        rows = np.sort(np.random.default_rng(subset_seed).choice(X.shape[0], n, replace=False))
        return np.array(X[rows], order="C", copy=True)
    return np.array(X, order="C", copy=True)


def label(name: str, n: int | None) -> str:
    """Dataset label used in results and plots: ``glove-200`` or ``glove-200-n100000``."""
    return name if n is None else f"{name}-n{n}"


def remove_raw():
    shutil.rmtree(data_dir() / "raw", ignore_errors=True)
