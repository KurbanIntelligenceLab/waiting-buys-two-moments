"""Approximate optimality versus demonstrated nonuniqueness of the grid NPMLE.

Two fits that both pass a KKT tolerance are not thereby "different
maximizers".  This script reports, for converged EM (each initialization, at
its iteration cap) against the independently optimized NPMLE on the same
target and grid:

  * objective values and gaps, KKT residuals (sup_p D(p)/N),
  * the NPMLE re-solved at tighter tolerances (1e-8 -> 1e-11 -> 1e-13) and
    whether its likelihood, fitted cell probabilities or date move,
  * max |qhat_EM - qhat_NPMLE| over observation cells, weight TV distance,
    welfare-curve difference, and selected dates.

Demonstrated nonuniqueness (witness).  The log likelihood and the directional
derivative D(p) depend on w only through the fitted cell probabilities
qhat = Psi w.  So any w' >= 0 with Psi w' = qhat_NPMLE has exactly the NPMLE's
objective value and KKT residual.  We solve the LP

    min ||w' - w_EM||_1   s.t.  Psi w' = qhat_NPMLE (tightest solve),  w' >= 0

(HiGHS) and report its feasibility residual, its objective and KKT residual
recomputed from w', its L1 distance from w_EM and from w_NPMLE, and its
welfare curve and selected dates.  When w' selects a different date from the
NPMLE, the set of grid maximizers contains decision-relevant variation, up to
the reported LP feasibility residual.  When the witness collapses onto the
NPMLE, nothing is demonstrated.

Usage: python -m experiments.em_mechanism.maximizers [--population population] [--finite finite]
Writes results/em_mechanism/analysis/maximizers_{population,finite}.csv.gz and a JSON summary.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from experiments.em_mechanism.common import OUT, dists, gamma1_cells, write_csv_atomic

AN = OUT / "analysis"
TOLS = (1e-8, 1e-11, 1e-13)


def ll_kkt(w, q, Psi):
    mix = np.maximum(Psi @ w, 1e-300)
    ll = float(np.sum(q * np.log(mix)) / q.sum())
    kkt = float((Psi.T @ (q / mix)).max() / q.sum() - 1.0)
    return ll, kkt


def witness(w_em, q_target, Psi):
    """min ||w - w_em||_1 s.t. Psi w = q_target, w >= 0 (variables w, u)."""
    G = len(w_em)
    c = np.concatenate([np.zeros(G), np.ones(G)])
    I = np.eye(G)
    A_ub = np.block([[I, -I], [-I, -I]])
    b_ub = np.concatenate([w_em, -w_em])
    A_eq = np.hstack([Psi, np.zeros_like(Psi)])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=q_target,
                  bounds=[(0, None)] * G + [(0, None)] * G, method="highs",
                  options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10})
    if not res.success:
        return None, res.message
    w = np.maximum(res.x[:G], 0.0)
    return polish(w, q_target, Psi), ""


def polish(w, q_target, Psi, rounds=5):
    """Least-norm correction on the support so that Psi w = q_target to machine
    precision, keeping w >= 0 (components are only moved where w > 0)."""
    for _ in range(rounds):
        r = q_target - Psi @ w
        if np.abs(r).max() < 1e-16:
            break
        S = np.flatnonzero(w > 1e-12)
        P = Psi[:, S]
        try:
            dw = P.T @ np.linalg.lstsq(P @ P.T, r, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        step = 1.0
        neg = dw < 0
        if neg.any():
            step = min(1.0, 0.99 * float(np.min(w[S][neg] / -dw[neg])))
        w = w.copy()
        w[S] += step * dw
    return np.maximum(w, 0.0)


def compare_one(meta, q, Psi, grid, w_em, np_fits, evals):
    """np_fits: {tol: NPMLEFit}; evals: {(T,beta): Welfare}."""
    tight = np_fits[TOLS[-1]]
    w_np = tight.weights
    q_np = Psi @ w_np
    ll_em, kkt_em = ll_kkt(w_em, q, Psi)
    w_wit, msg = witness(w_em, q_np, Psi)
    w_rev, msg_rev = witness(w_np, Psi @ w_em, Psi)
    row = {**meta, "ll_em": ll_em, "kkt_em": kkt_em}
    for tol, f in np_fits.items():
        row[f"ll_npmle_{tol:.0e}"] = f.ll
        row[f"kkt_npmle_{tol:.0e}"] = f.kkt
        row[f"status_npmle_{tol:.0e}"] = f.status
        row[f"max_abs_cell_diff_npmle_{tol:.0e}_vs_tightest"] = float(np.abs(Psi @ f.weights - q_np).max())
    row["ll_em_minus_npmle_tightest"] = ll_em - tight.ll
    row["max_abs_cell_diff_em_vs_npmle"] = float(np.abs(Psi @ w_em - q_np).max())
    row["weight_tv_em_vs_npmle"] = float(0.5 * np.abs(w_em - w_np).sum())
    row["em_support_gt_1e-6"] = int((w_em > 1e-6).sum())
    row["npmle_support_gt_1e-10"] = int((w_np > 1e-10).sum())
    if w_wit is not None:
        ll_w, kkt_w = ll_kkt(w_wit, q, Psi)
        row.update({"witness_ok": True, "witness_feas_resid": float(np.abs(Psi @ w_wit - q_np).max()),
                    "witness_ll_minus_npmle": ll_w - tight.ll, "witness_kkt": kkt_w,
                    "witness_l1_to_em": float(np.abs(w_wit - w_em).sum()),
                    "witness_l1_to_npmle": float(np.abs(w_wit - w_np).sum())})
    else:
        row.update({"witness_ok": False, "witness_msg": msg})
    # reverse witness: EM's fitted probabilities imposed, as close as possible to the NPMLE
    if w_rev is not None:
        ll_r, kkt_r = ll_kkt(w_rev, q, Psi)
        row.update({"rev_ok": True, "rev_feas_resid": float(np.abs(Psi @ w_rev - Psi @ w_em).max()),
                    "rev_ll_minus_em": ll_r - ll_em, "rev_kkt": kkt_r,
                    "rev_l1_to_npmle": float(np.abs(w_rev - w_np).sum()),
                    "rev_l1_to_em": float(np.abs(w_rev - w_em).sum())})
    else:
        row.update({"rev_ok": False, "rev_msg": msg_rev})
    out = []
    for cfg, ev in evals.items():
        W = np.vstack([w_em, w_np, w_wit if w_wit is not None else w_np,
                       w_rev if w_rev is not None else w_em])
        C = ev.curves(W)
        d = ev.decide(W, C)
        r = {**row, "T": cfg[0], "beta": cfg[1], "date_em": int(d[0]), "date_npmle": int(d[1]),
             "welfare_max_rel_diff_em_vs_npmle": float(np.abs(C[0] - C[1]).max() / C[1].max())}
        for tol, f in np_fits.items():
            Cf = ev.curves(f.weights[None, :])
            r[f"date_npmle_{tol:.0e}"] = int(ev.decide(f.weights[None, :], Cf)[0])
        if w_rev is not None:
            r["date_rev"] = int(d[3])
            r["welfare_max_rel_diff_rev_vs_em"] = float(np.abs(C[3] - C[0]).max() / C[0].max())
        if w_wit is not None:
            r["date_witness"] = int(d[2])
            r["welfare_max_rel_diff_witness_vs_npmle"] = float(np.abs(C[2] - C[1]).max() / C[1].max())
        out.append(r)
    return out


def population(sub):
    from experiments.em_mechanism.core import (Welfare, bin_projection, certified_npmle,
                                               exact_cell_probs, grid_and_psi)
    rows = []
    D = dists()
    configs = sorted({(c.T, c.beta) for c in gamma1_cells()})
    from experiments.em_mechanism.snapshots import population_task, population_tasks
    for name in population_tasks(sub):
        tau, G, variant = int(name.split("_")[0][3:]), int(name.split("_")[1][1:]), name.split("_")[2]
        task = population_task(sub, name)
        grid, Psi = grid_and_psi(G, tau)
        evals = {cfg: Welfare(grid, *cfg) for cfg in configs}
        cap = task.cap
        W = task.snaps[cap]
        meta = task.rows
        for dn, dist in D.items():
            q = exact_cell_probs(dist, tau) if variant == "continuous" else Psi @ bin_projection(dist, grid)
            fits = {tol: certified_npmle(q, grid, tau, Psi, delta=tol) for tol in TOLS}
            for i, (d_, init) in enumerate(meta):
                if d_ != dn:
                    continue
                rows += compare_one({"tau": tau, "G": G, "variant": variant, "distribution": dn,
                                     "init": init, "em_iteration": cap}, q, Psi, grid, W[i], fits, evals)
    df = pd.DataFrame(rows)
    write_csv_atomic(df, AN / "maximizers_population.csv.gz")
    return df


def finite(sub):
    """Uniform start at the cap vs NPMLE, every completed finite cell."""
    from experiments.em_mechanism.core import Welfare, certified_npmle, grid_and_psi
    from experiments.em_mechanism.pairs import load_pairs
    from experiments.em_mechanism.run_finite import simulate_counts
    cells = {c.cid: c for c in gamma1_cells()}
    pairs = {p["config_id"]: p for p in load_pairs(2)}
    rows = []
    for f in sorted(glob.glob(str(OUT / sub / "cells" / "*.done.json"))):
        st = json.loads(Path(f).read_text())["settings"]
        if st["G"] != 200:
            continue
        key = Path(f).name.replace(".done.json", "")
        from experiments.em_mechanism.snapshots import finite_cell
        snapc = finite_cell(sub, key)
        cell = cells[st["cid"]]
        grid, Psi = grid_and_psi(st["G"], 2)
        ev = {(cell.T, cell.beta): Welfare(grid, cell.T, cell.beta)}
        counts, _ = simulate_counts(st["kind"], cell, pairs[cell.cid], st["N"], st["seeds"])
        snap = snapc.at(st["cap"])
        init = "uniform"
        for s_i in range(len(st["seeds"])):
            fits = {tol: certified_npmle(counts[s_i], grid, 2, Psi, delta=tol) for tol in TOLS}
            if fits[TOLS[-1]].status == "invalid":
                continue
            rows += compare_one({"kind": st["kind"], "config_id": cell.cid, "distribution": cell.dist_name,
                                 "N": st["N"], "seed_index": st["seeds"][s_i], "init": init,
                                 "em_iteration": st["cap"]}, counts[s_i], Psi, grid, snap[s_i], fits, ev)
    df = pd.DataFrame(rows)
    if len(df):
        write_csv_atomic(df, AN / "maximizers_finite.csv.gz")
    return df


def summarize(df, label):
    if df is None or not len(df):
        return {}
    w = df[df.witness_ok.astype(bool)]
    tight = f"{TOLS[-1]:.0e}"
    return {
        "rows": int(len(df)),
        "median_abs_ll_em_minus_npmle": float(df.ll_em_minus_npmle_tightest.abs().median()),
        "max_abs_ll_em_minus_npmle": float(df.ll_em_minus_npmle_tightest.abs().max()),
        "median_kkt_em": float(df.kkt_em.median()), "max_kkt_em": float(df.kkt_em.max()),
        "median_kkt_npmle_tightest": float(df[f"kkt_npmle_{tight}"].median()),
        "share_npmle_tightest_certified": float((df[f"status_npmle_{tight}"] == "certified").mean()),
        "median_max_abs_cell_diff_em_vs_npmle": float(df.max_abs_cell_diff_em_vs_npmle.median()),
        "max_max_abs_cell_diff_em_vs_npmle": float(df.max_abs_cell_diff_em_vs_npmle.max()),
        "share_date_em_eq_npmle": float((df.date_em == df.date_npmle).mean()),
        "share_npmle_date_changes_1e-8_to_tightest": float((df["date_npmle_1e-08"] != df[f"date_npmle_{tight}"]).mean()),
        "witness_solved": int(len(w)),
        "witness_max_feas_resid": float(w.witness_feas_resid.max()) if len(w) else None,
        "witness_max_abs_ll_minus_npmle": float(w.witness_ll_minus_npmle.abs().max()) if len(w) else None,
        "witness_date_differs_from_npmle": int((w.date_witness != w.date_npmle).sum()) if len(w) else 0,
        "witness_date_equals_em_and_differs_from_npmle": int(((w.date_witness == w.date_em)
                                                              & (w.date_em != w.date_npmle)).sum()) if len(w) else 0,
        "em_npmle_date_disagreements": int((df.date_em != df.date_npmle).sum()),
        "rev_solved": int(df.rev_ok.astype(bool).sum()),
        "rev_max_feas_resid": float(df[df.rev_ok.astype(bool)].rev_feas_resid.max()),
        "rev_max_abs_ll_minus_em": float(df[df.rev_ok.astype(bool)].rev_ll_minus_em.abs().max()),
        "rev_max_kkt": float(df[df.rev_ok.astype(bool)].rev_kkt.max()),
        "rev_date_differs_from_em": int((df.rev_ok.astype(bool) & (df.date_rev != df.date_em)).sum()),
        "rev_date_equals_npmle_and_differs_from_em": int((df.rev_ok.astype(bool) & (df.date_rev == df.date_npmle)
                                                          & (df.date_em != df.date_npmle)).sum()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="population")
    ap.add_argument("--finite", default=None)
    a = ap.parse_args(argv)
    AN.mkdir(parents=True, exist_ok=True)
    out = {"population": summarize(population(a.population), "population")}
    if a.finite:
        out["finite"] = summarize(finite(a.finite), "finite")
    (AN / "maximizers_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
