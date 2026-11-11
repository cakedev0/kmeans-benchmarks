import numpy as np
import pytest

from kmb.inertia import evaluate_inertia, inertia_reference


def _data(n=3000, d=17, k=40, seed=0, offset=0.0, scale=1.0):
    rng = np.random.default_rng(seed)
    X = (rng.normal(size=(n, d)) * scale + offset).astype(np.float32)
    C = X[rng.choice(n, k, replace=False)] + rng.normal(scale=0.1 * scale, size=(k, d))
    return X, C.astype(np.float32)


@pytest.mark.parametrize("offset,scale", [(0.0, 1.0), (500.0, 30.0), (0.0, 1e-3)])
def test_matches_float64_numpy(offset, scale):
    # A large offset (like SIFT, norms ~509) stresses the float32 GEMM search.
    X, C = _data(offset=offset, scale=scale)
    expected = inertia_reference(X, C)
    got = evaluate_inertia(X, C)
    assert got.inertia == pytest.approx(expected, rel=1e-9)
    assert evaluate_inertia(X, C, exact=True).inertia == pytest.approx(expected, rel=1e-12)


def test_chunking_does_not_change_result():
    X, C = _data(n=5000)
    whole = evaluate_inertia(X, C).inertia
    chunked = evaluate_inertia(X, C, chunk_bytes=1).inertia  # 1024-row chunks
    assert chunked == pytest.approx(whole, rel=1e-12)


def test_centers_equal_to_samples_give_zero():
    X, _ = _data(n=200, k=1)
    result = evaluate_inertia(X, X)
    assert result.inertia == pytest.approx(0.0, abs=1e-12)
    assert result.n_empty == 0


def test_empty_and_nonfinite_centers():
    X, C = _data(n=1000, k=10)
    far = np.full((2, X.shape[1]), 1e6, dtype=np.float32)
    bad = np.full((1, X.shape[1]), np.nan, dtype=np.float32)
    result = evaluate_inertia(X, np.vstack([C, far, bad]))
    assert result.n_empty >= 3
    assert result.n_nonfinite == 1
    assert result.inertia == pytest.approx(inertia_reference(X, C), rel=1e-9)


def test_float64_accumulation_on_large_n():
    # Mean of 2M equal squared distances: float32 accumulation would drift.
    X = np.zeros((2_000_000, 2), dtype=np.float32)
    X[:, 0] = 1.0 + 1e-3
    result = evaluate_inertia(X, np.array([[1.0, 0.0]], dtype=np.float32))
    expected = (np.float64(np.float32(1.0 + 1e-3)) - 1.0) ** 2
    assert result.inertia == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("backend", ["cupy", "torch"])
def test_device_backends_match_numpy(backend):
    X, C = _data(offset=100.0, scale=10.0)
    if backend == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            Xd = cp.asarray(X)
        except Exception:
            pytest.skip("no CUDA device")
    else:
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        Xd = torch.from_numpy(X).cuda()
    assert evaluate_inertia(Xd, C).inertia == pytest.approx(inertia_reference(X, C), rel=1e-9)
