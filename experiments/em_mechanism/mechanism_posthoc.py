"""POST-HOC moment-space interventions (added after the prespecified tests of
mechanism.py, on the evidence that T1 assigns the late date in ~99% of worse
events).  Labelled post hoc throughout; not part of MECHANISM_PROTOCOL.md.

At tau = 2 the pre-failure record kernels span polynomials of degree <= 3, so
fixing the fitted cell probabilities Psi w is the same as fixing the raw
moments (m0, m1, m2, m3).  Each intervention below is an L1 projection with
moment equalities (rows p^0..p^3):

  T1r  necessity : closest to w_cap with EM55's (m1, m2, m3)  -> date back to d55?
  Tm1  only m1   : closest to w55 with (m1_cap, m2_55, m3_55) -> late date?
  Tm2  only m2   : closest to w55 with (m1_55, m2_cap, m3_55)
  Tm3  only m3   : closest to w55 with (m1_55, m2_55, m3_cap)
  Tno3 all but m3: closest to w55 with (m1_cap, m2_cap, m3_55) -> stays at d55?
  Ttrue3         : closest to w_cap with (m1_cap, m2_cap, m3_true)  -> repaired?

Diagnostics: relative error of m1, m2, m3 against the true population, at EM55
and at the cap; sign agreement of the m3 change with the welfare-contrast change.

Usage: python -m experiments.em_mechanism.mechanism_posthoc [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp

import numpy as np
import pandas as pd

from experiments.em_mechanism.common import OUT, gamma1_cells, reference_regret, write_csv_atomic
from experiments.em_mechanism.mechanism import CAP, INIT_INDEX_UNIFORM, MOUT, N_INITS, l1_project


def mrows(grid, targets):
    """Rows p^0..p^3 scaled by target magnitude, and right-hand sides."""
    A = np.vstack([grid ** j for j in range(4)])
    b = np.asarray(targets, dtype=float)
    s = np.where(np.abs(b) > 0, np.abs(b), 1.0)
    return A / s[:, None], b / s


def run_cell(args):
    key, events = args
    from experiments.em_mechanism.core import Welfare, grid_and_psi
    from experiments.em_mechanism.pairs import load_pairs
    meta = json.loads((OUT / "finite" / "cells" / f"{key}.done.json").read_text())["settings"]
    cell = next(c for c in gamma1_cells() if c.cid == meta["cid"])
    pair = next(p for p in load_pairs(2) if p["config_id"] == cell.cid)
    truths = {"ref": reference_regret(cell), "lower": np.asarray(pair["regret"][0]),
              "upper": np.asarray(pair["regret"][1])}
    grid, _ = grid_and_psi(200, 2)
    ev = Welfare(grid, cell.T, cell.beta)
    m_true = np.array([1.0] + [cell.dist.raw_moment(j) for j in (1, 2, 3)])
    from experiments.em_mechanism.snapshots import finite_cell
    snap = finite_cell("finite", key)
    pos = {s: i for i, s in enumerate(meta["seeds"])}
    out = []
    for e in events:
        i = pos[e["seed_index"]]
        a = snap.at(55)[i]
        b = snap.at(CAP)[i]
        ma = np.array([a @ grid ** j for j in range(4)])
        mb = np.array([b @ grid ** j for j in range(4)])
        o = {k: e[k] for k in ("key", "config_id", "distribution", "N", "seed_index", "cls", "d55", "dcap",
                               "r55", "rcap", "dC")}
        for j in (1, 2, 3):
            o[f"relerr55_m{j}"] = (ma[j] - m_true[j]) / m_true[j]
            o[f"relerrcap_m{j}"] = (mb[j] - m_true[j]) / m_true[j]
            o[f"relchange_m{j}"] = (mb[j] - ma[j]) / ma[j]
        specs = {
            "T1r_late_structure_55_fit": (b, ma),
            "Tm1_only_m1": (a, [1, mb[1], ma[2], ma[3]]),
            "Tm2_only_m2": (a, [1, ma[1], mb[2], ma[3]]),
            "Tm3_only_m3": (a, [1, ma[1], ma[2], mb[3]]),
            "Tno3_all_but_m3": (a, [1, mb[1], mb[2], ma[3]]),
            "Ttrue3_late_with_true_m3": (b, [1, mb[1], mb[2], m_true[3]]),
        }
        for name, (target, mom) in specs.items():
            A, r = mrows(grid, mom)
            w, res = l1_project(target, A, r)
            if w is None:
                o[f"{name}_feasible"] = False
                continue
            d = int(ev.decide(w[None], ev.curves(w[None]))[0])
            o.update({f"{name}_feasible": True, f"{name}_date": d,
                      **{f"{name}_regret_{t}": float(R[d - 1]) for t, R in truths.items()}})
        out.append(o)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", default=None,
                    help="read events from / write outputs to this directory (reproduction checks)")
    a = ap.parse_args(argv)
    global MOUT
    if a.out_dir:
        from pathlib import Path
        MOUT = Path(a.out_dir).resolve()
    ev = pd.read_csv(MOUT / "events.csv.gz")
    ev = ev[ev.d55 != ev.dcap]
    tasks = [(k, g.to_dict(orient="records")) for k, g in ev.groupby("key")]
    with mp.Pool(a.workers) as pool:
        rows = [o for out in pool.imap_unordered(run_cell, tasks) for o in out]
    df = pd.DataFrame(rows)
    write_csv_atomic(df, MOUT / "posthoc_moment_tests.csv.gz")
    print(len(df), "events")


if __name__ == "__main__":
    main()
