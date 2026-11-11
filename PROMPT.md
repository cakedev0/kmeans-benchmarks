# Build the first version of kmeans-benchmarks

This repository should become for k-means what
[ann-benchmarks](https://github.com/erikbern/ann-benchmarks) is for nearest-neighbor
search: one plot per dataset and number of clusters, with fit time on one axis and
clustering quality on the other, and one Pareto front per implementation. This file holds
the context and the requirements for a first version.

## Why

We are evaluating how scikit-learn's `KMeans` should run on GPUs: through array API
support (the same Python code on NumPy, CuPy and PyTorch arrays, possibly with JIT fast
paths for some backends), or through a plugin that routes to cuML. Comparing time per
Lloyd iteration is no longer enough:

- Initializations differ in cost and quality. Greedy k-means++ is slow in cuML but gives
  better centers than random init. cuML's default k-means|| ends with worse inertia than
  random init on GloVe ([NVIDIA/cuml#8886](https://github.com/NVIDIA/cuml/issues/8886),
  reposted as #8896).
- New algorithms trade iterations for quality differently. e-means
  ([arXiv 2611.09501](https://arxiv.org/abs/2611.09501),
  [blog](https://e-means.github.io/)), from Meta FAIR, is mini-batch k-means with an
  exponential moving average update of the centers and random init. Its authors report
  reaching the loss of 25 Lloyd epochs in ~1.4 to 2.6 epochs, and a lower loss than
  k-means++ in many cases. It is reported to be available in FAISS (to check).
  Mini-batch k-means in scikit-learn is the older member of that family.

So the question is: for a given time budget, which implementation and settings give the
lowest inertia? That's a Pareto front.

## What we know so far

All on one NVIDIA L4 (an AWS VM with 8 vCPUs), float32, TF32 off, data already on the
device, cuML 26.10, CuPy 14.2:

- **Array API k-means** ([ogrisel/cursor-sandbox#17](https://github.com/ogrisel/cursor-sandbox/pull/17),
  journal at <https://ogrisel.github.io/cursor-sandbox/pr-17/>): a scikit-learn compatible
  `KMeans` written with array API operations, with native scatter-add updates for PyTorch
  and CuPy. Its Lloyd iterations are faster than cuML's at small k (17 clusters) and
  slower from ~100 clusters up (for example 5.6 s against 2.0 s for 10 × 100 iterations
  at k=256, d=128, n=150,000).
- **CuPy k-means++ and Lloyd** ([ogrisel/cursor-sandbox#22](https://github.com/ogrisel/cursor-sandbox/pull/22),
  [journal 024](https://github.com/cakedev0/cursor-sandbox/blob/cupy-kmeans/kmeans_lloyd/journal/024-cupy-kmeans-plusplus-and-lloyd/report.md)):
  two versions, one with CuPy array operations only and one with CUDA kernels that CuPy
  compiles at run time. Against cuML:
  - k-means++: 2.3x to 2.8x faster with array operations, 3.5x to 3.8x with the kernels,
    ~26x on 1M × 3 data.
  - Lloyd iteration: ~1.5x slower with array operations, 1.0x to 1.3x faster with the
    kernels.
  - Full fit (`n_init=10`, `max_iter=100`, k-means++): 1.15x to 1.5x faster with array
    operations, 1.7x to 2.6x with the kernels, same inertia.
- **Why cuML's greedy k-means++ is slow:** an nsys profile on GloVe at k=256 gives ~15 ms
  per center, with 9.8 ms in a skinny GEMM of `X` against the 8 candidates (at ~40% of
  the memory bandwidth) and 4.2 ms recomputing the row norms of `X` at every step.
- **CPU k-means++ in scikit-learn** was sped up 3x to 15x in
  [scikit-learn#35140](https://github.com/scikit-learn/scikit-learn/pull/35140) (one fused
  kernel per step, two-level sampling, direct recomputation of small distances).
- **Data matters for quality, not much for speed.** On Gaussian noise every init ends
  within 0.02% of the others, which makes random init look free. On SIFT1M at k=256, after
  10 × 100 Lloyd iterations, all inits end within 0.05%. Differences show at short
  budgets and on data with structure, which is exactly what a Pareto front exposes.
- Related cuVS issues: slow k-means++ in practice
  ([NVIDIA/cuvs#2266](https://github.com/NVIDIA/cuvs/issues/2266)), and k-means++
  ignoring sample weights ([#1897](https://github.com/NVIDIA/cuvs/issues/1897),
  [#1809](https://github.com/NVIDIA/cuvs/issues/1809)). Super K-Means
  ([arXiv 2603.20009](https://arxiv.org/abs/2603.20009)) is being added to cuVS
  ([NVIDIA/cuvs#2948](https://github.com/NVIDIA/cuvs/pull/2948)).

## Requirements

### Measurement

- **One shared quality metric.** Compute the inertia from the centers each implementation
  returns, with one evaluator for everyone: mean squared distance of every training sample
  to its nearest center, float64 accumulation. Don't trust each library's `inertia_`.
  FAISS, for example, reports the objective on its training subsample.
- **Relative inertia** on the quality axis: inertia divided by the best inertia found for
  that (dataset, k) by any run, minus 1. Keep a long reference run per (dataset, k) so the
  best value is stable (for example greedy k-means++ then Lloyd to convergence, best of
  several seeds).
- **Fit time** on the other axis: wall time of the whole fit, initialization included,
  with the data already in memory (host RAM for CPU runs, device memory for GPU runs).
  Report data transfer and the first-call cost (module loading, JIT compilation)
  separately, not in the fit time.
- **Sweeps generate the points.** Per implementation, sweep the settings that trade time
  for quality: `init` (random, k-means++, k-means||), `n_init`, `max_iter` (or epochs),
  batch size for mini-batch methods, and the tolerance. Plot each implementation's Pareto
  front, as ann-benchmarks does.
- **Also record** iterations actually run, peak memory, library versions (from
  `importlib.metadata`, not stale `__version__` attributes), driver and hardware. Write one
  JSON line per fit as soon as it finishes, so an interrupted run keeps its results.
- **Several seeds** per point (at least 3), and report the spread.

### Fairness

- Same dtype for everyone (float32 by default) and the same thread count on CPU.
- TF32 and other reduced-precision modes off by default. If a library has a faster
  reduced-precision mode, put it in a separate, labeled category.
- Run the contenders of one comparison interleaved, in a rotating order, and on an
  otherwise idle machine. Other jobs on the same GPU or CPU invalidate timings.
- Check that settings mean the same thing across libraries. A faster run that stops
  earlier is not faster: log iterations and inertia next to every time.

### Implementations

First version:

- scikit-learn `KMeans` (Lloyd, and Elkan) and `MiniBatchKMeans`, CPU.
- cuML `KMeans` with `init="random"`, `"k-means++"` (greedy) and `"scalable-k-means++"`
  (k-means||), GPU.
- FAISS `Kmeans` on CPU and GPU (Lloyd), and FAISS e-means if it is released.
- The two CuPy versions from #22 (`kmeans_cupy(fused=False)` and `kmeans_cupy(fused=True)`
  in the `kmeans_lloyd` package of that branch).

Later: scikit-learn-intelex (oneDAL), the array API `KMeans` of #17 on PyTorch and CuPy,
cuVS through its own Python API (including balanced k-means), Super K-Means, Flash-KMeans
(search for the paper and code), and PyTorch-based implementations.

### Datasets

Start with real embeddings, where the init matters, plus one low-dimensional case:

| dataset | shape | source |
| --- | --- | --- |
| SIFT1M | 1,000,000 × 128 | [ann-benchmarks `sift-128-euclidean`](http://ann-benchmarks.com/sift-128-euclidean.hdf5), or the IRISA tarball `ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz` |
| GloVe-200 | 1,183,514 × 200 | [ann-benchmarks `glove-200-angular`](http://ann-benchmarks.com/glove-200-angular.hdf5) |
| CLIP ViT-B/32 | 1,000,000 × 512 | [Qdrant/wolt-food-clip-ViT-B-32-embeddings](https://huggingface.co/datasets/Qdrant/wolt-food-clip-ViT-B-32-embeddings), first 1M rows |
| low-dimensional | ~1,000,000 × 2 to 6 | to choose: a real 2-d or 3-d set (coordinates), or Gaussian blobs |

- k = 256 and 1024 first. Later k = 16 and k = 65,536, and larger n (10M), where FAISS and
  e-means are designed to shine.
- Use the vectors as published: GloVe is unnormalized (norms ~6.2 ± 1.1), the CLIP
  vectors are L2-normalized, and SIFT norms are ~509.
- The CLIP set has many exact duplicates (~23% of the first 200,000 rows). Keep it as
  is, since duplicates are realistic, but know that random init can then pick identical
  centers and leave empty clusters.
- Cache the converted float32 `.npy` files. ann-benchmarks returned HTTP 403 to Python's
  `urllib` but not to `curl`.

### Gotchas we already hit

- **FAISS subsamples by default:** `max_points_per_centroid=256`, so at k=256 it trains on
  65,536 points. Raise it to use all the data, or include the subsampling as a deliberate
  sweep setting. FAISS defaults to `niter=25` and random init.
- **cuML:** `init="k-means++"` is greedy k-means++, `"scalable-k-means++"` is k-means||.
  With `n_init="auto"` it runs one init for those, so set `n_init` explicitly.
  `n_iter_` covers only the best init. Its stopping rule compares float32 inertias, so
  with `tol=1e-9` it stops as soon as the inertia stops decreasing. It keeps empty
  clusters, and its k-means++ gave different centers for the same `random_state` across
  runs.
- **cuML and CuPy in one process:** cuML switches CuPy's allocator to
  `rmm_cupy_allocator`, which pays a `cudaMalloc` and a synchronizing `cudaFree` per
  temporary array. That made CuPy code up to ~2.6x slower. Time CuPy code under
  `cupy.cuda.using_allocator(cupy.get_default_memory_pool().malloc)`, or in its own
  process.
- **scikit-learn's `tol`** is relative to the mean feature variance. cuML's and FAISS's
  are not, so the same `tol` means different things.
- **GPU power caps:** an L4 runs at its 72 W cap under sustained load, with clocks well
  below the maximum. Long sweeps can drift. Interleave contenders and report the
  conditions.
- **Library environments conflict** (cuML, FAISS GPU, PyTorch, CUDA versions). Use one
  environment per library (pixi works well) and run each fit in a subprocess, driven by
  one runner.

## Deliverables of the first version

1. A runner that takes a config (datasets, k values, implementations, sweeps), runs each
   fit in the right environment and appends JSON lines to `results/`.
2. Dataset download and conversion scripts with caching.
3. The shared inertia evaluator, with a test: it must match a float64 NumPy computation
   on small data.
4. A plot script: one figure per (dataset, k, hardware), fit time on a log x axis,
   relative inertia on a log y axis, one Pareto front per implementation.
5. A README that explains how to run one implementation on one dataset in a few minutes,
   and how to read the plots.

Start small: one dataset, k=256, three implementations, and check that the plot answers
the question before scaling up. Estimate the duration of a sweep from its first fits
before launching it.

## Open questions

- Should the reference for relative inertia be the best run found by anyone, or a fixed
  long reference run? The first is simpler, the second is stable across result updates.
- Should CPU and GPU implementations share a plot? One plot per hardware is easier to read,
  but the question of whether a GPU is worth it needs both.
- Should mini-batch methods also report inertia on held-out data, since they see only
  part of the training set per step?
