"""Shared settings, truths and output helpers for the EM-mechanism study."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Data root: the repository or an extracted release.  Defaults to the checkout
# containing this file; set WBTM_ROOT when the package is installed elsewhere
# (e.g. from a wheel) so that results/ and runs/ are found.
ROOT = (Path(os.environ["WBTM_ROOT"]).resolve() if os.environ.get("WBTM_ROOT")
        else Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "em_mechanism"
RUNS = ROOT / "runs" / "em_mechanism"

# Iterations whose full records (welfare curves, cell probabilities, moments)
# are always kept in the committed outputs, in addition to the dense schedule
# of scalars.  30/55/80 are the paper's stopping counts.
CHECKPOINTS = (0, 1, 3, 10, 30, 55, 80, 100, 300, 1000, 3000, 10_000, 30_000,
               100_000, 300_000, 1_000_000, 3_000_000, 10_000_000)
HEADLINE_ITERS = (30, 55, 80)


def full_schedule(cap: int) -> np.ndarray:
    from experiments.em_mechanism.core import schedule
    s = set(schedule(cap).tolist()) | {c for c in CHECKPOINTS if c <= cap}
    return np.array(sorted(s), dtype=np.int64)


def gamma1_cells():
    from wtm.config import load_frozen, population_cells
    cells = [c for c in population_cells(load_frozen()) if c.gamma == 1]
    assert len(cells) == 30
    return cells


def dists():
    """The five reference families, in frozen-config order."""
    seen = {}
    for c in gamma1_cells():
        seen.setdefault(c.dist_name, c.dist)
    return seen


def reference_regret(cell_or_pop) -> np.ndarray:
    """R(s) = 1 - W^s/W^{t*}, s = 1..T-1, under the reference population."""
    pop = cell_or_pop.population() if hasattr(cell_or_pop, "population") else cell_or_pop
    return np.array([pop.relative_regret(int(s)) for s in pop.dates])


def regret_lookup(R: np.ndarray, dates: np.ndarray) -> np.ndarray:
    d = np.asarray(dates, dtype=int)
    out = np.full(d.shape, np.nan)
    ok = d >= 1
    out[ok] = R[d[ok] - 1]
    return out


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def code_fingerprint(extra: dict) -> str:
    """Hash of this package's code, the wtm modules it uses, and the settings."""
    from wtm.provenance import sha256_file
    # generator code only: editing the analysis/figure scripts must not
    # invalidate computed cells
    files = [ROOT / "experiments" / "em_mechanism" / f for f in
             ("core.py", "common.py", "pairs.py", "run_population.py", "run_finite.py")]
    files += [ROOT / "wtm" / f for f in ("npmle.py", "observation.py", "lp.py", "welfare.py",
                                        "simulate.py", "regret_calibrated.py",
                                        "distributions.py", "config.py")]
    files.append(ROOT / "experiments" / "config" / "frozen_benchmark.yaml")
    h = hashlib.sha256()
    for p in files:
        h.update(f"{p.relative_to(ROOT)}\0{sha256_file(p)}\n".encode())
    h.update(json.dumps(extra, sort_keys=True, default=str).encode())
    return h.hexdigest()


def write_csv_atomic(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    df.to_csv(tmp, index=False, compression="gzip" if path.suffix == ".gz" else None)
    tmp.replace(path)
