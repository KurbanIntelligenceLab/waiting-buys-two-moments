"""Extreme-pair rebuild, all 30 gamma=1 configs, t=2.
See results/extreme_pair_rebuild_protocol.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"

from wtm.config import config_id, load_frozen, population_cells  # noqa: E402
from wtm.lp import IdentifiedSet, support_grid  # noqa: E402
from wtm.welfare import PopulationConfig  # noqa: E402
from experiments.scripts.lower_bound import Evaluator, kl  # noqa: E402

import sys as _sys
T_PREFIX = int(_sys.argv[1]) if len(_sys.argv) > 1 else 2
GRID_SIZE = 2001


def extreme_members(grid, S, T):
    c = (1.0 - grid) ** (T - 1)
    lo = linprog(c=c, A_eq=S.A, b_eq=S.b, bounds=[(0.0, None)] * S.n, method="highs")
    hi = linprog(c=-c, A_eq=S.A, b_eq=S.b, bounds=[(0.0, None)] * S.n, method="highs")
    return lo, hi


def main():
    cfg = load_frozen()
    cells = [c for c in population_cells(cfg) if c.gamma == 1]
    assert len(cells) == 30, f"expected 30, got {len(cells)}"
    grid = support_grid(GRID_SIZE)

    rows = []
    for cell in cells:
        cid = config_id(dist=cell.dist_name, gamma=1, T=cell.T, beta=cell.beta)
        S = IdentifiedSet(cell.dist, T_PREFIX, grid)
        pop = PopulationConfig(cell.dist, cell.gamma, cell.T, cell.beta)
        ev = Evaluator(grid, cell.gamma, cell.T, cell.beta, T_PREFIX)

        lo, hi = extreme_members(grid, S, cell.T)
        row = {"config_id": cid, "distribution": cell.dist_name, "T": cell.T,
               "beta": cell.beta, "prefix_t": T_PREFIX, "grid_resolution": GRID_SIZE}
        if not (lo.success and hi.success):
            row.update({"FAILED": True, "lo_message": lo.message, "hi_message": hi.message})
            rows.append(row)
            print(f"FAILED: {cid} ({cell.dist_name}, T={cell.T}, beta={cell.beta}): "
                  f"lo={lo.message} hi={hi.message}")
            continue

        w_lo, w_hi = lo.x, hi.x
        res_lo, res_hi = S.moment_residual(w_lo), S.moment_residual(w_hi)

        W_lo, W_hi = ev.trajectory(w_lo), ev.trajectory(w_hi)
        R_lo, R_hi = ev.regret(W_lo), ev.regret(W_hi)
        total = R_lo + R_hi
        s_idx = int(np.argmin(total))
        zeta = float(total[s_idx])

        q_lo, q_hi = ev.record_law(w_lo), ev.record_law(w_hi)
        fwd, rev = kl(q_lo, q_hi), kl(q_hi, q_lo)
        sym = 0.5 * (fwd + rev) if np.isfinite(fwd) and np.isfinite(rev) else np.inf

        dates = ev.dates
        tstar_lo = int(dates[int(np.argmax(W_lo))])
        tstar_hi = int(dates[int(np.argmax(W_hi))])
        rel_gap = np.abs(W_hi - W_lo) / np.maximum(W_hi, W_lo)

        sup_lo = grid[w_lo > 1e-10]
        sup_hi = grid[w_hi > 1e-10]
        shared = np.intersect1d(np.round(sup_lo, 8), np.round(sup_hi, 8))

        row.update({
            "FAILED": False,
            "moment_residual_lower": res_lo, "moment_residual_upper": res_hi,
            "numerically_valid": bool(max(res_lo, res_hi) < 1e-9),
            "n_support_lower": int(len(sup_lo)), "n_support_upper": int(len(sup_hi)),
            "n_shared_support_points": int(len(shared)),
            "t_star_lower": tstar_lo, "t_star_upper": tstar_hi,
            "t_star_separation": abs(tstar_lo - tstar_hi),
            "true_t_star": pop.t_star,
            "max_relative_welfare_gap": float(rel_gap.max()),
            "date_of_max_gap": int(dates[int(np.argmax(rel_gap))]),
            "argmin_date_zeta": int(dates[s_idx]),
            "zeta": zeta, "zeta_over_2": zeta / 2.0,
            "zeta_is_negligible": bool(zeta < 1e-4),
            "kl_forward": fwd, "kl_reverse": rev, "kl_symmetrized": sym,
        })
        rows.append(row)
        print(f"{cid} ({cell.dist_name}, T={cell.T}, beta={cell.beta}): "
              f"t*={tstar_lo}/{tstar_hi} sep={row['t_star_separation']} "
              f"gap={row['max_relative_welfare_gap']:.3f} zeta={zeta:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"extreme_pair_rebuild_t{T_PREFIX}_raw.csv", index=False)

    ok = df[df.FAILED == False]  # noqa: E712
    n_fail = int(df.FAILED.sum())
    disagree = ok[ok.t_star_separation > 0]
    summary = {
        "n_configs": len(df), "n_failed": n_fail, "n_valid": len(ok),
        "n_disagree": len(disagree), "frac_disagree": len(disagree) / len(ok) if len(ok) else np.nan,
        "median_max_rel_welfare_gap": ok.max_relative_welfare_gap.median(),
        "p25_max_rel_welfare_gap": ok.max_relative_welfare_gap.quantile(.25),
        "p75_max_rel_welfare_gap": ok.max_relative_welfare_gap.quantile(.75),
        "max_max_rel_welfare_gap": ok.max_relative_welfare_gap.max(),
        "median_zeta_all": ok.zeta.median(), "max_zeta_all": ok.zeta.max(),
        "median_zeta_disagree_only": disagree.zeta.median() if len(disagree) else np.nan,
        "max_zeta_disagree_only": disagree.zeta.max() if len(disagree) else np.nan,
        "median_zeta_over_2_disagree_only": disagree.zeta_over_2.median() if len(disagree) else np.nan,
        "n_zero_shared_support": int((ok.n_shared_support_points == 0).sum()),
        "n_numerically_valid": int(ok.numerically_valid.sum()),
    }
    pd.DataFrame([summary]).to_csv(RESULTS / f"extreme_pair_rebuild_t{T_PREFIX}_summary.csv", index=False)

    lines = ["# Extreme-pair rebuild -- results\n",
             f"Protocol: `results/extreme_pair_rebuild_protocol.md`. "
             f"Script: `experiments/scripts/extreme_pair_rebuild.py`.\n",
             f"30 configs (gamma=1), t=2, horizon-survival functional, grid=2001.\n",
             f"**Failures: {n_fail}/{len(df)}.**\n" if n_fail else "**All 30 configs solved successfully.**\n",
             f"**Disagreement (t* differs): {len(disagree)}/{len(ok)} = {100*len(disagree)/len(ok):.1f}%**\n",
             f"**Median max relative welfare gap: {100*summary['median_max_rel_welfare_gap']:.1f}%** "
             f"(IQR {100*summary['p25_max_rel_welfare_gap']:.1f}-{100*summary['p75_max_rel_welfare_gap']:.1f}%, "
             f"max {100*summary['max_max_rel_welfare_gap']:.1f}%)\n",
             f"**Median zeta (disagreement instances only): {summary['median_zeta_disagree_only']}, "
             f"floor zeta/2 = {summary['median_zeta_over_2_disagree_only']}**\n" if len(disagree) else
             "No disagreement instances found among the 30 configs at t=2 with this functional.\n",
             f"Zero-shared-support (fully distinguishing pair): {summary['n_zero_shared_support']}/{len(ok)}\n",
             f"Numerically valid (residual < 1e-9): {summary['n_numerically_valid']}/{len(ok)}\n"]
    (RESULTS / f"extreme_pair_rebuild_t{T_PREFIX}_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
