"""Exact rational reconstruction and verification of the extreme pairs (Table 1).

For every gamma=1 configuration and tau in {2, 3, 4}, this script

1. re-solves the two horizon-survival LPs of `extreme_pair_rebuild.py` on the
   2,001-point grid (floating point: this only *selects* supports);
2. places atoms at the exact grid points i/2000 of each selected support and
   solves the moment system (unit mass + mixed moments through degree 2*tau-1)
   for the weights in exact rational arithmetic;
3. checks that the weights are nonnegative, so each reconstructed measure is an
   exact member of M_tau;
4. evaluates both welfare trajectories by exact greedy allocation, and reports
   the full optimal-date sets, the welfare spread, and the regret separation
   zeta of Theorem 1, all in rational arithmetic.

Floating point never decides anything reported here.

Usage: python experiments/scripts/extreme_pair_exact.py
Writes results/extreme_pair_exact_raw.csv and results/extreme_pair_exact_summary.csv.
"""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"

from wtm.config import config_id, load_frozen, population_cells  # noqa: E402
from wtm.decision_id import (DecisionProblem, eval_mixed_exact,  # noqa: E402
                             exact_greedy, solve_exact)
from wtm.lp import IdentifiedSet, support_grid  # noqa: E402
from experiments.scripts.extreme_pair_rebuild import GRID_SIZE, extreme_members  # noqa: E402

DEN = GRID_SIZE - 1  # grid point i is exactly i / 2000


def exact_member(prob: DecisionProblem, w_float: np.ndarray):
    """Rational weights on the float LP's support that match M_tau exactly."""
    idx = np.flatnonzero(w_float > 1e-10)
    atoms = [Fraction(int(i), DEN) for i in idx]
    n = prob.n
    if len(atoms) != n + 1:
        raise RuntimeError(f"support size {len(atoms)} != {n + 1}; not a square system")
    A = [[eval_mixed_exact(prob.basis[j], x) for x in atoms] for j in range(n + 1)]
    w = solve_exact(A, list(prob.mu_q))
    # redundant by construction, checked anyway
    for j in range(n + 1):
        assert sum(a * wi for a, wi in zip(A[j], w)) == prob.mu_q[j]
    return atoms, w


def exact_trajectory(prob: DecisionProblem, atoms, weights) -> dict:
    out = {}
    for s in prob.dates:
        P, U = prob.phi_polys(s)
        m = [sum((wi * eval_mixed_exact(p, x) for x, wi in zip(atoms, weights)), Fraction(0))
             for p in P]
        v = [sum((wi * eval_mixed_exact(u, x) for x, wi in zip(atoms, weights)), Fraction(0))
             for u in U]
        out[s] = exact_greedy(m, v, prob.beta_q)
    return out


def main():
    cfg = load_frozen()
    cells = [c for c in population_cells(cfg) if c.gamma == 1]
    assert len(cells) == 30
    grid = support_grid(GRID_SIZE)
    assert np.allclose(grid, np.arange(GRID_SIZE) / DEN, rtol=0, atol=1e-15)

    rows = []
    for tau in (2, 3, 4):
        for cell in cells:
            cid = config_id(dist=cell.dist_name, gamma=1, T=cell.T, beta=cell.beta)
            prob = DecisionProblem(cell.dist, cell.T, cell.beta, tau)
            S = IdentifiedSet(cell.dist, tau, grid)
            lo, hi = extreme_members(grid, S, cell.T)
            assert lo.success and hi.success, cid

            members = [exact_member(prob, lo.x), exact_member(prob, hi.x)]
            nonneg = [all(wi >= 0 for wi in w) for _, w in members]
            W = [exact_trajectory(prob, a, w) for a, w in members]
            best = [max(Wi.values()) for Wi in W]
            opt = [sorted(s for s in prob.dates if Wi[s] == b) for Wi, b in zip(W, best)]
            R = [{s: 1 - Wi[s] / b for s in prob.dates} for Wi, b in zip(W, best)]
            zeta = min(R[0][s] + R[1][s] for s in prob.dates)
            spread = max(
                (abs(W[0][s] - W[1][s]) / max(W[0][s], W[1][s])
                 if max(W[0][s], W[1][s]) > 0 else Fraction(0))
                for s in prob.dates)
            disjoint = not set(opt[0]) & set(opt[1])
            assert disjoint == (zeta > 0)
            rows.append({
                "config_id": cid, "distribution": cell.dist_name, "T": cell.T,
                "beta": cell.beta, "tau": tau,
                "support_lower": " ".join(str(a) for a in members[0][0]),
                "support_upper": " ".join(str(a) for a in members[1][0]),
                "weights_nonnegative": all(nonneg),
                "min_weight": float(min(min(w) for _, w in members)),
                "opt_dates_lower": " ".join(map(str, opt[0])),
                "opt_dates_upper": " ".join(map(str, opt[1])),
                "unique_opt_both": len(opt[0]) == 1 and len(opt[1]) == 1,
                "disjoint_opt_sets": disjoint,
                "welfare_spread": float(spread),
                "zeta": float(zeta), "zeta_over_2": float(zeta / 2),
                "zeta_exact": str(zeta),
            })
            r = rows[-1]
            print(f"tau={tau} {cid} {cell.dist_name:28s} T={cell.T} beta={cell.beta}: "
                  f"nonneg={r['weights_nonnegative']} opt={opt[0]}/{opt[1]} "
                  f"spread={r['welfare_spread']:.4f} zeta/2={r['zeta_over_2']:.3e}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "extreme_pair_exact_raw.csv", index=False)
    summ = []
    for tau, g in df.groupby("tau"):
        dis = g[g.disjoint_opt_sets]
        summ.append({
            "tau": tau, "n_pairs": len(g),
            "n_exact_members": int(g.weights_nonnegative.sum()),
            "n_unique_opt_both": int(g.unique_opt_both.sum()),
            "n_disjoint": len(dis),
            "median_spread": g.welfare_spread.median(),
            "p25_spread": g.welfare_spread.quantile(.25),
            "p75_spread": g.welfare_spread.quantile(.75),
            "max_spread": g.welfare_spread.max(),
            "median_zeta_over_2_disjoint": dis.zeta_over_2.median(),
            "max_zeta_over_2": g.zeta_over_2.max(),
        })
    s = pd.DataFrame(summ)
    s.to_csv(RESULTS / "extreme_pair_exact_summary.csv", index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
