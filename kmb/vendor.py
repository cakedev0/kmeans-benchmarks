"""Fetch the CuPy k-means of ogrisel/cursor-sandbox#22 at a pinned commit.

``python -m kmb.vendor`` (or ``pixi run fetch-kmeans-lloyd``) clones the branch into
``.cache/cursor-sandbox``; the ``cupy`` and ``cupy-fused`` implementations import
``kmeans_lloyd`` from there.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/cakedev0/cursor-sandbox.git"
COMMIT = "6d8464c05e58131e7dc6496b65a88a62bf6326ea"  # branch cupy-kmeans, journal 024
ROOT = Path(__file__).resolve().parent.parent / ".cache" / "cursor-sandbox"
SRC = ROOT / "kmeans_lloyd" / "src"


def fetch() -> Path:
    if not (ROOT / ".git").exists():
        ROOT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", str(ROOT)], check=True)
        subprocess.run(["git", "-C", str(ROOT), "remote", "add", "origin", REPO_URL], check=True)
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    if head != COMMIT:
        subprocess.run(["git", "-C", str(ROOT), "fetch", "-q", "--depth", "1", "origin", COMMIT],
                       check=True)
        subprocess.run(["git", "-C", str(ROOT), "checkout", "-q", COMMIT], check=True)
    return SRC


def ensure_importable():
    if not (SRC / "kmeans_lloyd").exists():
        raise RuntimeError("kmeans_lloyd missing: run `pixi run -e cupy fetch-kmeans-lloyd`")
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))


if __name__ == "__main__":
    print(fetch())
