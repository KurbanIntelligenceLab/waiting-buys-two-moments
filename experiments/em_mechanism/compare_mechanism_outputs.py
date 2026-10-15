"""Check regenerated EM mechanism outputs against the committed ones.

    python -m experiments.em_mechanism.compare_mechanism_outputs DIR [--v7] [--maximizers]

DIR holds the outputs of `mechanism --out-dir DIR` and
`mechanism_posthoc --out-dir DIR`.  Each table is compared with
results/em_mechanism/mechanism/ after sorting on its key columns: identical
rows, identical non-numeric values, numeric values equal within 1e-9
(relative) -- in practice they are bitwise equal.

--v7          also recompute the late-EM-vs-NPMLE validation (V7) from the
              snapshots and compare with validation/V7_late_em_vs_npmle_finite.csv.gz
--maximizers  also recompute the finite-sample maximizer/witness table and
              compare with analysis/maximizers_finite.csv.gz (~10 min)

Exit 0 if everything matches, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.em_mechanism.common import OUT

TABLES = {
    "events.csv.gz": ["key", "seed_index"],
    "constraint_tests.csv.gz": ["key", "seed_index"],
    "posthoc_moment_tests.csv.gz": ["key", "seed_index"],
    "population_control.csv.gz": ["distribution", "T", "beta", "init"],
}


def compare_frames(name, a, b, keys, rtol=1e-9):
    msgs = []
    if len(a) != len(b):
        return [f"{name}: {len(a)} rows vs committed {len(b)}"]
    if set(a.columns) != set(b.columns):
        return [f"{name}: columns differ: {sorted(set(a.columns) ^ set(b.columns))}"]
    a = a.sort_values(keys).reset_index(drop=True)
    b = b.sort_values(keys).reset_index(drop=True)[a.columns]
    worst = 0.0
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            both_nan = np.isnan(xv) & np.isnan(yv)
            d = np.abs(xv - yv)
            tol = rtol * np.maximum(np.abs(yv), 1e-300)
            bad = ~both_nan & ~(d <= tol)
            if bad.any():
                msgs.append(f"{name}.{c}: {int(bad.sum())} values differ (max abs {np.nanmax(d):.3g})")
            if (~both_nan).any():
                worst = max(worst, float(np.nanmax(np.where(both_nan, 0, d))))
        else:
            xs, ys = x.astype(str), y.astype(str)
            if not (xs == ys).all():
                msgs.append(f"{name}.{c}: {int((xs != ys).sum())} values differ")
    print(f"  {name}: {len(a)} rows, max abs numeric difference {worst:.3g}")
    return msgs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--v7", action="store_true")
    ap.add_argument("--maximizers", action="store_true")
    a = ap.parse_args(argv)
    d = Path(a.dir)
    ref = OUT / "mechanism"
    errs = []
    for f, keys in TABLES.items():
        if not (d / f).exists():
            errs.append(f"missing regenerated {f}")
            continue
        errs += compare_frames(f, pd.read_csv(d / f), pd.read_csv(ref / f), keys)
    sel_new = json.loads((d / "bin_selection.json").read_text())
    sel_old = json.loads((ref / "bin_selection.json").read_text())
    for N in sel_old:
        if sel_new[N]["top"] != sel_old[N]["top"] or sel_new[N]["low"] != sel_old[N]["low"]:
            errs.append(f"bin selection differs at N={N}")
    print("  bin_selection.json: " + ("identical selections" if not any("bin selection" in e for e in errs) else "DIFFERS"))

    if a.v7:
        import experiments.em_mechanism.validate as V
        captured = {}
        V.write_csv_atomic = lambda df, path: captured.__setitem__(Path(path).name, df)
        V.v7({}, "finite")
        errs += compare_frames("V7_late_em_vs_npmle_finite.csv.gz", captured["V7_late_em_vs_npmle_finite.csv.gz"],
                               pd.read_csv(OUT / "validation" / "V7_late_em_vs_npmle_finite.csv.gz"),
                               ["cell", "seed_row"])
    if a.maximizers:
        import experiments.em_mechanism.maximizers as MX
        captured = {}
        MX.write_csv_atomic = lambda df, path: captured.__setitem__(Path(path).name, df)
        MX.finite("finite")
        errs += compare_frames("maximizers_finite.csv.gz", captured["maximizers_finite.csv.gz"],
                               pd.read_csv(OUT / "analysis" / "maximizers_finite.csv.gz"),
                               ["kind", "config_id", "N", "seed_index", "T", "beta"])
    for e in errs:
        print("MISMATCH", e)
    print("REPRODUCED: all compared outputs match the committed ones" if not errs else
          f"NOT REPRODUCED: {len(errs)} mismatches")
    return 0 if not errs else 1


if __name__ == "__main__":
    sys.exit(main())
