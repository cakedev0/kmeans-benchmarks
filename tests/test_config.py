import pytest

from kmb import config
from kmb.plot import pareto


def test_expand_grid_merges_defaults_and_rejects_unknown():
    defaults = {"init": "k-means++", "n_init": 1, "max_iter": 300}
    points = config.expand_grid({"init": ["random", "k-means++"], "max_iter": [5, 10]}, defaults)
    assert len(points) == 4
    assert all(p["n_init"] == 1 for p in points)
    with pytest.raises(ValueError, match="unknown"):
        config.expand_grid({"n_inits": [1]}, defaults)


def test_none_stands_for_null():
    points = config.expand_grid({"max_points_per_centroid": ["none", 256]},
                                {"max_points_per_centroid": None})
    assert [p["max_points_per_centroid"] for p in points] == [None, 256]


def test_interleave_rotates():
    per_impl = {"a": [1, 2, 3], "b": [4, 5], "c": [6, 7, 8]}
    assert config.interleave(per_impl) == [1, 4, 6, 5, 7, 2, 8, 3]


def test_load_example_configs():
    for path in ["configs/quickstart.toml", "configs/pilot-glove-k256-gpu.toml"]:
        cfg = config.load(path, machine="test")
        assert cfg.grids
        jobs = [j for _, _, per in config.jobs_by_comparison(cfg) for j in config.interleave(per)]
        assert jobs


def test_pareto():
    pts = [{"fit_s": 1, "rel": 0.1}, {"fit_s": 2, "rel": 0.2}, {"fit_s": 3, "rel": 0.01},
           {"fit_s": 0.5, "rel": 0.5}]
    assert [p["fit_s"] for p in pareto(pts)] == [0.5, 1, 3]
