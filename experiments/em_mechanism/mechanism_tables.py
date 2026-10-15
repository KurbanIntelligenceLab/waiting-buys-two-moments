"""m3-bias and asymmetric-loss tables for MECHANISM_REPORT.md section 4.

Writes results/em_mechanism/mechanism/regret_by_m3_error.csv (all estimators
pooled: EM55, EM at the cap, NPMLE; every initialization) and
m3_bias_by_estimator.csv.  Also prints the m3 bias at iterations 0, 55 and
the cap for every initialization, and the pair-member optimal dates.

Usage: python -m experiments.em_mechanism.mechanism_tables
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from experiments.em_mechanism.common import OUT, gamma1_cells, write_csv_atomic

CAP = 1_000_000
BINS = [-1, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 10]


def main():
    cells = {c.cid: c for c in gamma1_cells()}
    mt = {c.dist_name: c.dist.raw_moment(3) for c in gamma1_cells()}
    cols = ["config_id", "distribution", "N", "seed_index", "method", "init", "iteration", "m3",
            "regret_ref", "date"]
    t = pd.concat([pd.read_csv(f, usecols=cols)
                   for f in glob.glob(str(OUT / "finite" / "cells" / "ref_*_table.csv.gz"))])
    t["relerr_m3"] = (t.m3 - t.distribution.map(mt)) / t.distribution.map(mt)
    em = t[(t.method == "em") & t.iteration.isin([0, 55, CAP])]
    print("median signed relative error of m3 by init (EM):")
    print(em.groupby(["N", "iteration", "init"]).relerr_m3.median().unstack(2).round(3).to_string())

    t = t[((t.method == "em") & t.iteration.isin([55, CAP])) | (t.method == "npmle")].copy()
    t["tstar"] = t.config_id.map({k: c.population().t_star for k, c in cells.items()})
    t["date_minus_tstar"] = t.date - t.tstar
    t["estimator"] = np.where(t.method == "npmle", "NPMLE", np.where(t.iteration == 55, "EM55", "EM_cap"))
    t["m3_err_bin"] = pd.cut(t.relerr_m3, BINS)
    g = t.groupby(["N", "m3_err_bin"], observed=True).agg(
        n=("regret_ref", "size"), mean_regret=("regret_ref", "mean"),
        share_too_early=("date_minus_tstar", lambda v: (v < 0).mean()),
        share_too_late=("date_minus_tstar", lambda v: (v > 0).mean())).reset_index()
    g["m3_err_bin"] = g.m3_err_bin.astype(str)
    h = t.groupby(["N", "estimator"]).agg(
        share_m3_under=("relerr_m3", lambda v: (v < 0).mean()), median_relerr=("relerr_m3", "median"),
        share_too_early=("date_minus_tstar", lambda v: (v < 0).mean()),
        share_too_late=("date_minus_tstar", lambda v: (v > 0).mean()),
        mean_regret=("regret_ref", "mean")).reset_index()
    write_csv_atomic(g, OUT / "mechanism" / "regret_by_m3_error.csv")
    write_csv_atomic(h, OUT / "mechanism" / "m3_bias_by_estimator.csv")
    print(g.round(4).to_string(index=False)); print(h.round(4).to_string(index=False))

    from experiments.em_mechanism.pairs import load_pairs
    rows = []
    for p in load_pairs(2):
        tP = cells[p["config_id"]].population().t_star
        tl = min(i + 1 for i, r in enumerate(p["regret"][0]) if r == 0)
        tu = min(i + 1 for i, r in enumerate(p["regret"][1]) if r == 0)
        m4 = [float(np.dot(m["weights"], np.array(m["atoms"]) ** 4)) for m in p["members"]]
        rows.append({"config_id": p["config_id"], "tstar_lower": tl, "tstar_P": tP, "tstar_upper": tu,
                     "m4_lower": m4[0], "m4_upper": m4[1]})
    pm = pd.DataFrame(rows)
    write_csv_atomic(pm, OUT / "mechanism" / "pair_member_optimal_dates.csv")
    print(f"upper earlier than P: {(pm.tstar_upper < pm.tstar_P).sum()}/30; lower == P: "
          f"{(pm.tstar_lower == pm.tstar_P).sum()}/30; upper m4 larger: {(pm.m4_upper > pm.m4_lower).sum()}/30")


if __name__ == "__main__":
    main()
