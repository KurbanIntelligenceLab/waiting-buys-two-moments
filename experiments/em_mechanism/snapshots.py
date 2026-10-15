"""Access to full-precision weight snapshots, from runs/ or from the snapshot bundle.

The mechanism, maximizer, identified-interval and late-EM validation analyses
need the fitted grid weights at iterations 0, 30, 55, 80 and the cap, plus the
NPMLE weights.  They live in two places:

  runs/em_mechanism/{finite*,population}/*.npz
      written by the generators (run_finite.py, run_population.py); complete,
      git-ignored, about 2.8 GB.
  results/em_mechanism_snapshots/{finite*,population}/*.npz
      the compact analysis-input bundle (build_snapshot_bundle.py): only the
      arrays these analyses read; uniform-start rows only for finite samples.
      Distributed as wbtm-supplement-2026.10-em-snapshots.zip; extract it into
      the repository or release root.

`source()` picks runs/ when present, else the bundle.  Set
WBTM_EM_SNAPSHOTS=bundle (or =runs) to force one; a missing source raises
SnapshotsMissing with instructions instead of failing obscurely.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.em_mechanism.common import ROOT, RUNS

BUNDLE = ROOT / "results" / "em_mechanism_snapshots"
FINITE_SNAPS = (0, 30, 55, 80)


class SnapshotsMissing(RuntimeError):
    pass


def _choice(rel: str) -> tuple[str, Path]:
    want = os.environ.get("WBTM_EM_SNAPSHOTS", "").strip().lower()
    runs, bundle = RUNS / rel, BUNDLE / rel
    if want == "runs" or (not want and runs.exists()):
        if runs.exists():
            return "runs", runs
    if want in ("", "bundle") and bundle.exists():
        return "bundle", bundle
    raise SnapshotsMissing(
        f"no weight snapshots for {rel}: looked for {runs} and {bundle}. Extract "
        "wbtm-supplement-2026.10-em-snapshots.zip into the repository/release root "
        "(see experiments/em_mechanism/README.md, 'Snapshot bundle'), or regenerate the "
        "study with run_finite.py / run_population.py.")


@dataclass
class FiniteCell:
    """Uniform-start snapshots of one finite-sample cell, rows in seed order."""
    key: str
    source: str
    cap: int
    snaps: dict            # iteration -> (S, G) weights
    npmle_weights: np.ndarray
    npmle_curves: np.ndarray
    end_curves: np.ndarray  # (S, T-1) welfare curve of the uniform-start endpoint
    end_ll: np.ndarray      # (S,) per-individual log likelihood at the endpoint

    def at(self, k: int) -> np.ndarray:
        return self.snaps[int(k)]


def finite_cell(sub: str, key: str) -> FiniteCell:
    src, path = _choice(f"{sub}/{key}.npz")
    z = np.load(path, allow_pickle=False)
    if src == "bundle":
        cap = int(z["cap"])
        snaps = {k: z[f"snap_{k}"] for k in FINITE_SNAPS + (cap,)}
        return FiniteCell(key, src, cap, snaps, z["npmle_weights"], z["npmle_curves"],
                          z["end_curves"], z["end_ll"])
    rows = [r.split("|") for r in z["rows"]]
    full = [i for i, (_, t) in enumerate(rows) if not t.startswith("cvfold")]
    uni = [(int(rows[i][0]), j, i) for j, i in enumerate(full) if rows[i][1] == "uniform"]
    uni.sort()
    js = [j for _, j, _ in uni]
    ii = [i for _, _, i in uni]
    caps = sorted(int(k[5:]) for k in z.files if k.startswith("snap_"))
    cap = caps[-1]
    snaps = {k: z[f"snap_{k}"][js] for k in caps}
    return FiniteCell(key, src, cap, snaps, z["npmle_weights"], z["npmle_curves"],
                      z["curves"][-1][ii], z["ll"][-1][ii])


@dataclass
class PopulationTask:
    name: str
    source: str
    rows: list              # [(distribution, init), ...]
    snaps: dict             # iteration -> (B, G)

    @property
    def cap(self) -> int:
        return max(self.snaps)


def population_task(sub: str, name: str) -> PopulationTask:
    src, path = _choice(f"{sub}/{name}.npz")
    z = np.load(path, allow_pickle=False)
    snaps = {int(k[5:]): z[k] for k in z.files if k.startswith("snap_")}
    return PopulationTask(name, src, [tuple(r.split("|")) for r in z["rows"]], snaps)


def population_tasks(sub: str = "population") -> list[str]:
    for base in ((RUNS, BUNDLE) if os.environ.get("WBTM_EM_SNAPSHOTS", "").lower() != "bundle" else (BUNDLE,)):
        d = base / sub
        if d.is_dir() and any(d.glob("*.npz")):
            return sorted(p.stem for p in d.glob("*.npz"))
    raise SnapshotsMissing(f"no population snapshots under {RUNS / sub} or {BUNDLE / sub}")
