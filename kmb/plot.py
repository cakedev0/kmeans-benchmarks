"""Pareto-front plots: fit time (log x) against relative inertia (log y).

One figure per (dataset, k, machine, device). Each point is one setting of one
implementation: the median over seeds of fit time and of relative inertia, with
whiskers from the fastest to the slowest seed and from the best to the worst inertia.
The line joins each implementation's Pareto front (the settings that no faster
setting of the same implementation beats on inertia); the other settings are faded.

Relative inertia = inertia / best - 1, where ``best`` is the lowest inertia found for the
(dataset, k) by any run in the loaded results, on any machine (the reference runs of
``configs/reference-*.toml`` are there to make it stable). Lower-left is better.

Next to each PNG, a CSV lists every point with its settings and whether it is on the
front, which is how to read off which setting gives a front point.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

# Fixed per implementation family, so a library has the same color in every figure
# (validated categorical palette: blue, orange, aqua, yellow, magenta, green, violet, red).
COLORS = {
    "cuml": "#2a78d6", "sklearn-lloyd": "#2a78d6",
    "cupy": "#eb6834", "sklearn-minibatch": "#eb6834",
    "cupy-fused": "#1baf7a",
    "faiss-gpu": "#eda100", "faiss-cpu": "#eda100",
    "emeans-gpu": "#e87ba4", "emeans-cpu": "#e87ba4",
    "sklearn-elkan": "#008300",
    "emeans-gpu-bf16": "#6250d6", "emeans-cpu-cython": "#6250d6",
}
MARKERS = {"cuml": "o", "sklearn-lloyd": "o", "cupy": "s", "sklearn-minibatch": "s",
           "cupy-fused": "D", "faiss-gpu": "^", "faiss-cpu": "^", "emeans-gpu": "v",
           "emeans-cpu": "v", "sklearn-elkan": "P", "emeans-gpu-bf16": "X",
           "emeans-cpu-cython": "X"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FLOOR = 1e-7   # lowest possible floor; each figure uses one decade below its smallest value


def load_records(paths) -> list:
    records = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "ok":
                records.append(r)
    return records


def best_inertia(records) -> dict:
    best = {}
    for r in records:
        key = (r["dataset"], r["k"])
        best[key] = min(best.get(key, math.inf), r["inertia"])
    return best


def series_label(r, split_by=None) -> str:
    label = r["impl"]
    if split_by:
        label += f" {split_by}={r['params'].get(split_by)}"
    return label


def aggregate(records, best, split_by=None) -> list:
    """One point per (implementation, settings): medians and ranges over seeds."""
    groups = defaultdict(list)
    for r in records:
        if r.get("role") == "reference":
            continue
        machine = r["machine"] + (f"-{r['threads']}t" if r["device"] == "cpu" else "")
        key = (r["dataset"], r["k"], machine, r["device"], series_label(r, split_by),
               json.dumps(r["params"], sort_keys=True))
        groups[key].append(r)
    points = []
    for (dataset, k, machine, device, label, params), rs in groups.items():
        b = best[(dataset, k)]
        rel = [r["inertia"] / b - 1 for r in rs]
        times = [r["fit_s"] for r in rs]
        iters = [r["n_iter"] for r in rs if r.get("n_iter") is not None]
        points.append({
            "dataset": dataset, "k": k, "machine": machine, "device": device,
            "series": label, "impl": rs[0]["impl"], "params": params,
            "settings": " ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in sorted(json.loads(params).items())),
            "n_seeds": len(rs),
            "fit_s": statistics.median(times), "fit_s_min": min(times), "fit_s_max": max(times),
            "rel": statistics.median(rel), "rel_min": min(rel), "rel_max": max(rel),
            "n_iter_median": statistics.median(iters) if iters else None,
            "n_empty_max": max(r.get("n_empty", 0) for r in rs),
            "peak_mem_mb_median": statistics.median(
                [max(r["peak_mem_bytes"], 0) / 2**20 for r in rs
                 if r.get("peak_mem_bytes") is not None]
                or [math.nan]),
            "gpu_sm_clock_mean": statistics.mean(
                [r["gpu_sm_clock_mhz_mean"] for r in rs if r.get("gpu_sm_clock_mhz_mean")]
                or [math.nan]),
        })
    return points


def pareto(points) -> list:
    """Points not beaten by a faster point (ties in time broken by inertia)."""
    front, best_rel = [], math.inf
    for p in sorted(points, key=lambda p: (p["fit_s"], p["rel"])):
        if p["rel"] < best_rel:
            front.append(p)
            best_rel = p["rel"]
    return front


def plot_group(points, title, out_png, hardware_note=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The best run has relative inertia 0, which a log axis cannot show: draw values below
    # the floor (one decade under the smallest positive value) at the floor.
    positive = [v for p in points for v in (p["rel"], p["rel_min"]) if v > 0]
    floor = max(FLOOR, 10 ** math.floor(math.log10(min(positive))) / 10) if positive else 1e-6

    def _clip(v):
        return max(v, floor)

    fig, ax = plt.subplots(figsize=(8.5, 5.6), dpi=150)
    by_series = defaultdict(list)
    for p in points:
        by_series[p["series"]].append(p)
    for series in sorted(by_series):
        pts = by_series[series]
        impl = pts[0]["impl"]
        color = COLORS.get(impl, INK_2)
        marker = MARKERS.get(impl, "o")
        front = pareto(pts)
        on_front = {id(p) for p in front}
        for p in pts:
            p["on_front"] = id(p) in on_front
            alpha = 1.0 if p["on_front"] else 0.28
            ax.errorbar(p["fit_s"], _clip(p["rel"]),
                        xerr=[[p["fit_s"] - p["fit_s_min"]], [p["fit_s_max"] - p["fit_s"]]],
                        yerr=[[_clip(p["rel"]) - _clip(p["rel_min"])],
                              [_clip(p["rel_max"]) - _clip(p["rel"])]],
                        fmt=marker, ms=6.5, color=color, alpha=alpha, elinewidth=0.8,
                        capsize=0, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        xs = [p["fit_s"] for p in front]
        ys = [_clip(p["rel"]) for p in front]
        ax.plot(xs, ys, drawstyle="steps-post", color=color, lw=2, zorder=2,
                label=series, marker=marker, ms=0)
        ax.annotate(series, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=8, color=INK_2, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("fit time (s, initialization included, data already on the device)",
                  color=INK_2, fontsize=9)
    ax.set_ylabel("relative inertia (inertia / best known − 1)", color=INK_2, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    ax.set_title(hardware_note, loc="left", color=INK_2, fontsize=7.5)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.grid(True, which="minor", color=GRID, lw=0.4, alpha=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK)
    clipped = any(v < floor for p in points for v in (p["rel"], p["rel_min"]))
    fig.text(0.01, 0.005, "median over seeds; whiskers: min-max over seeds; faded: off the front"
             + (f"; values at {floor:g} reached the best known inertia" if clipped else ""),
             fontsize=7, color=INK_2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_png)
    plt.close(fig)


def _hardware_note(records, machine, device):
    rs = [r for r in records if r["device"] == device and (
        r["machine"] == machine or f"{r['machine']}-{r.get('threads')}t" == machine)]
    if not rs:
        return ""
    r = rs[-1]
    if device == "gpu" and r.get("gpu"):
        g = r["gpu"]
        return (f"{machine}: {g.get('gpu_name')}, driver {g.get('driver')}, power limit "
                f"{g.get('power_limit_w')} W")
    return f"{r['machine']}: {r['host'].get('cpu_model')}, {r.get('threads')} threads"


def main(paths=None, out="plots", split_by=None):
    paths = paths or sorted(glob.glob("results/*.jsonl"))
    records = load_records(paths)
    if not records:
        raise SystemExit("no successful fits in " + ", ".join(map(str, paths)))
    best = best_inertia(records)
    points = aggregate(records, best, split_by)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for p in points:
        groups[(p["dataset"], p["k"], p["machine"], p["device"])].append(p)
    written = []
    for (dataset, k, machine, device), pts in sorted(groups.items()):
        stem = f"{dataset}-k{k}-{machine}-{device}"
        title = f"{dataset}, k={k} — {device.upper()} on {machine}"
        plot_group(pts, title, out / f"{stem}.png", _hardware_note(records, machine, device))
        with open(out / f"{stem}.csv", "w", newline="") as f:
            fields = ["series", "settings", "on_front", "fit_s", "fit_s_min", "fit_s_max", "rel", "rel_min",
                      "rel_max", "n_seeds", "n_iter_median", "n_empty_max",
                      "peak_mem_mb_median", "gpu_sm_clock_mean"]
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for p in sorted(pts, key=lambda p: (p["series"], p["fit_s"])):
                w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                            for k, v in p.items()})
        written.append(out / f"{stem}.png")
        print(f"{out / stem}.png  ({len(pts)} points, best inertia "
              f"{best[(dataset, k)]:.8g})")
    return written
