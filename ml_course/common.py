"""
common.py
---------
Shared helpers for the ML / DL fundamentals course.

Runs in the WSL conda env `ft` (numpy, scikit-learn, scipy, torch). Every
lesson also runs on CPU: the models are tiny on purpose, because the point is
to see each effect in seconds and then change the knobs. The GPU is used when
free, but nothing here needs vLLM to be stopped (a few hundred MB at most).

Datasets are synthetic or bundled with scikit-learn (breast_cancer, digits,
diabetes), so nothing is downloaded and every result is reproducible from the
seed.

Each lesson prints result lines in a fixed `[tag] key=value` form, which
`run_all.py` regexes to check OUTCOMES (e.g. "leaky CV beats honest CV by a
lot"), not exact numbers.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("PYTHONHASHSEED", "0")


def parse_args(doc: str | None, extra: list[tuple[str, dict]] | None = None) -> argparse.Namespace:
    """--quick (fewer steps/repeats) and --seed for every lesson, plus lesson-specific flags."""
    ap = argparse.ArgumentParser(description=(doc or "").split("\n")[1] if doc else None)
    ap.add_argument("--quick", action="store_true", help="fewer steps / repeats (seconds per lesson)")
    ap.add_argument("--seed", type=int, default=0)
    for flag, kw in extra or []:
        ap.add_argument(flag, **kw)
    args = ap.parse_args()
    seed_all(args.seed)
    return args


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def device() -> str:
    """cuda if available, else cpu. Tiny models are often as fast on CPU, which is fine."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


def kv(tag: str, **values: float | int | str | bool) -> None:
    """Print one machine-checkable result line: [tag] a=0.912 b=3."""
    parts = []
    for k, v in values.items():
        if isinstance(v, float):
            v = f"{v:.4g}" if (abs(v) < 1e-3 and v != 0) or abs(v) >= 1e4 else f"{v:.4f}"
        parts.append(f"{k}={v}")
    print(f"[{tag}] " + " ".join(parts), flush=True)


@contextmanager
def timed(label: str) -> Iterator[None]:
    t0 = time.perf_counter()
    yield
    print(f"  ({label}: {time.perf_counter() - t0:.1f}s)")


def table(rows: list[dict], cols: list[str] | None = None) -> None:
    """Print a list of dicts as an aligned table."""
    if not rows:
        return
    cols = cols or list(rows[0])

    def cell(v: object) -> str:
        if isinstance(v, float):
            return f"{v:.4g}" if (abs(v) < 1e-3 and v != 0) or abs(v) >= 1e4 else f"{v:.4f}"
        return str(v)

    w = {c: max(len(c), *(len(cell(r.get(c, ""))) for r in rows)) for c in cols}
    print("  " + "  ".join(c.rjust(w[c]) for c in cols))
    for r in rows:
        print("  " + "  ".join(cell(r.get(c, "")).rjust(w[c]) for c in cols))


# ---------------------------------------------------------------------------
# Synthetic data used by several lessons
# ---------------------------------------------------------------------------

def spirals(n_per_class: int = 300, classes: int = 3, noise: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Interleaved spirals (Karpathy's CS231n toy set): not linearly separable, needs a real nonlinearity."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(classes):
        r = np.linspace(0.05, 1, n_per_class)
        t = np.linspace(c * 4, (c + 1) * 4, n_per_class) + rng.normal(0, noise, n_per_class)
        xs.append(np.c_[r * np.sin(t), r * np.cos(t)])
        ys.append(np.full(n_per_class, c))
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys)


def train_test(x: np.ndarray, y: np.ndarray, test: float = 0.3, seed: int = 0) -> tuple[np.ndarray, ...]:
    """Stratified split without importing sklearn (DL lessons stay torch + numpy)."""
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        k = int(round(len(idx) * test))
        te.append(idx[:k])
        tr.append(idx[k:])
    tr, te = rng.permutation(np.concatenate(tr)), rng.permutation(np.concatenate(te))
    return x[tr], x[te], y[tr], y[te]
