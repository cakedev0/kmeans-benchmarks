import numpy as np
import pytest

kernel = pytest.importorskip("kmb._emeans_kernel", reason="run `pixi run build-ext`")
pytest.importorskip("faiss")

from kmb.emeans_cpu import FastNumpyEmeans  # noqa: E402
from kmb.inertia import evaluate_inertia  # noqa: E402


def _step_reference(X, index, C):
    rows = X[index].astype(np.float64)
    d2 = ((rows[:, None, :] - C[None].astype(np.float64)) ** 2).sum(-1)
    labels = d2.argmin(1)
    sums = np.zeros(C.shape, dtype=np.float64)
    np.add.at(sums, labels, rows)
    return labels, sums, np.bincount(labels, minlength=len(C)), d2.min(1).sum()


@pytest.mark.parametrize("n_threads", [1, 4])
@pytest.mark.parametrize("d,k", [(3, 7), (64, 100), (200, 256)])
def test_step_matches_numpy(d, k, n_threads):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, d)).astype(np.float32)
    C = X[rng.choice(5000, k, replace=False)] + 0.01
    index = rng.permutation(5000)[:1500].astype(np.int64)
    table = np.zeros((k, d + 1), dtype=np.float32)    # e-means' strided layout
    sums, counts = table[:, :-1], table[:, -1]
    labels = np.empty(len(index), dtype=np.int64)
    objective = kernel.accumulate_step(
        X, index, C, (C * C).sum(1), sums, counts, labels,
        np.empty(len(index), dtype=np.int64), np.empty(k + 1, dtype=np.int64), 64, n_threads)
    ref_labels, ref_sums, ref_counts, ref_obj = _step_reference(X, index, C)
    assert (labels == ref_labels).mean() > 0.999          # float32 ties aside
    agree = labels == ref_labels
    if agree.all():
        np.testing.assert_allclose(sums, ref_sums, rtol=1e-5, atol=1e-4)
        np.testing.assert_array_equal(counts, ref_counts)
    assert objective == pytest.approx(ref_obj, rel=1e-4)


def test_fast_emeans_matches_upstream_quality():
    from faiss.contrib.e_means import NumpyEmeans

    rng = np.random.default_rng(0)
    centers = rng.normal(scale=5, size=(50, 16))
    X = (centers[rng.integers(50, size=40000)] + rng.normal(size=(40000, 16))).astype(np.float32)
    ref = NumpyEmeans(16, 50, nepochs=3, seed=0)
    ref_c, _ = ref.train(X)
    fast = FastNumpyEmeans(16, 50, nepochs=3, seed=0, n_threads=4)
    fast_c, labels = fast.train(X)
    assert fast.recipe["batch_size"] == ref.recipe["batch_size"]
    assert labels.shape == (40000,) and labels.min() >= 0
    ref_i = evaluate_inertia(X, ref_c).inertia
    fast_i = evaluate_inertia(X, fast_c).inertia
    assert fast_i == pytest.approx(ref_i, rel=2e-3)
