"""Recompute the robust baselines under monotonically refined bounds.

THE REFINEMENT IS MONOTONE BY CONSTRUCTION, not by argument. Each run is
evaluated at the default scan and at a finer one, and the two are combined as

    Lower_combined(s) = min(Lower_default(s), Lower_fine(s))
    Upper_combined(s) = max(Upper_default(s), Upper_fine(s))

so every previously evaluated candidate is retained and the combined interval
is weakly wider than either input, subject to solver tolerance. Nothing that
the default scan already found can be lost by refining.

WHY THAT JUSTIFIES A SUBSET RERUN FOR THE DOMINANCE RULES. The test is
`Lower(s') > Upper(s)`. Under combination,

    Lower_c(s') - Upper_c(s)  <=  Lower_default(s') - Upper_default(s),

so the tested gap is weakly smaller everywhere. A run whose gap was already
non-positive -- i.e. that did not fire -- cannot start firing. Only the runs
that fired at the default scan can change, so those are the ones rerun. This
is a consequence of the construction, not an empirical regularity.

WHY MAXIMIN GETS A FULL RERUN ANYWAY. Maximin selects `argmax_s Lower(s)`.
Every lower estimate weakly decreases under combination, but they decrease by
different amounts at different dates, so the argmax can move in any run,
including runs where nothing fired and runs that were never near a boundary.
No subset is justified and the whole grid is recomputed.

These quantities are NUMERICAL APPROXIMATIONS with measured sensitivity, not
dominance certificates. The attempt at a valid certificate is recorded in
results/certification_attempt.md and was unsuccessful; that is a statement
about the routes tried, not a proof that certification is unattainable.

Failed runs are recorded separately and never silently imputed.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
CKPT = RESULTS / "robust_rerun_cells"

FINE = dict(n_fill=201, n_lambda=201, n_refine=41, n_rounds=7)
GRID = 201


def combine(b_def, b_fine):
    """Retain every evaluated candidate: min of lowers, max of uppers."""
    from wtm.worst_case import WorstCaseBounds
    dates = b_def.dates
    lower, upper = {}, {}
    for s in [int(x) for x in dates]:
        ld, lf = b_def.lower.get(s), b_fine.lower.get(s)
        ud, uf = b_def.upper.get(s), b_fine.upper.get(s)
        cand_l = [v for v in (ld, lf) if v is not None and np.isfinite(v)]
        cand_u = [v for v in (ud, uf) if v is not None and np.isfinite(v)]
        if cand_l:
            lower[s] = min(cand_l)
        if cand_u:
            upper[s] = max(cand_u)
    return WorstCaseBounds(dates, lower, upper,
                           b_def.feasible and b_fine.feasible,
                           b_def.n_lp_solved + b_fine.n_lp_solved,
                           b_def.runtime_s + b_fine.runtime_s,
                           status=b_def.status, set_mode=b_def.set_mode)


def run_cell(args):
    which, dist_name, T, beta, N, tau, seeds = args
    CKPT.mkdir(exist_ok=True)
    tag = f"{which}_{dist_name}_{T}_{beta}_{N}_{tau}.json"
    ck = CKPT / tag
    if ck.exists():
        return str(ck)

    from wtm.config import config_id, load_frozen, population_cells
    from wtm.moment_projection import empirical_path_counts, moments_with_covariance
    from wtm.simulate import simulate_prefix
    from wtm.welfare import PopulationConfig
    from wtm.worst_case import compute_bounds, decide

    cfg = load_frozen()
    cell = next(c for c in population_cells(cfg)
                if c.gamma == 1 and c.dist_name == dist_name
                and c.T == T and c.beta == beta)
    pop = PopulationConfig(cell.dist, 1, T, beta)
    cell_id = config_id(dist=dist_name, gamma=1, T=T, beta=beta, N=N, t=tau)
    base = int(cell_id, 16)
    lazy = (which == "maximin")

    rows = []
    for sd in seeds:
        rec = {"rule": which, "cell_id": cell_id, "distribution": dist_name,
               "T": T, "beta": beta, "N": N, "prefix_t": tau, "seed_index": sd}
        try:
            data = simulate_prefix(cell.dist, N, T, 1, tau,
                                   seed=(base + sd) % 2 ** 32)
            mhat, cov = moments_with_covariance(
                empirical_path_counts(data, tau), tau, 1)
            kw = dict(grid_size=GRID, cov=cov, lazy_upper=lazy)
            bd = compute_bounds(mhat, tau, 1, T, beta, **kw)
            bf = compute_bounds(mhat, tau, 1, T, beta, scan=FINE, **kw)
            if not (bd.feasible and bf.feasible):
                rec.update(success=False, failure_reason="identified set empty")
                rows.append(rec)
                continue
            bc = combine(bd, bf)
            widened = all(bc.lower[s] <= bd.lower[s] + 1e-12 for s in bc.lower) and \
                all(bc.upper[s] >= bd.upper[s] - 1e-12 for s in bc.upper)
            if lazy:
                date_d = int(max(bd.lower, key=lambda s: bd.lower[s]))
                date_c = int(max(bc.lower, key=lambda s: bc.lower[s]))
            else:
                date_d = decide(bd)["selected_date"]
                date_c = decide(bc)["selected_date"]
            ev = {}
            if date_c >= 1:
                ev = {"welfare_efficiency": pop.efficiency(date_c),
                      "relative_regret": pop.relative_regret(date_c),
                      "squared_regret": pop.relative_regret(date_c) ** 2,
                      "exact_argmax_match": bool(date_c == pop.t_star)}
            rec.update(success=True, failure_reason="", widening_held=widened,
                       date_default=date_d, date_combined=date_c,
                       changed=bool(date_d != date_c),
                       true_optimal_date=pop.t_star, **ev)
        except Exception as exc:                                  # noqa: BLE001
            rec.update(success=False, failure_reason=f"{type(exc).__name__}: {exc}")
        rows.append(rec)
    ck.write_text(json.dumps(rows))
    return str(ck)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["worst_case_lp", "maximin"], required=True)
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    bench = pd.read_csv(RESULTS / "main_benchmark_raw_all.csv")
    if args.which == "worst_case_lp":
        # only runs that fired can change; see the module docstring
        sel = bench[(bench.rule == "worst_case_lp") & (bench.selected_date > 1)]
    else:
        sel = bench[bench.rule == "act_immediately"]        # the full grid
    tasks = []
    for key, g in sel.groupby(["distribution", "T", "beta", "N", "prefix_t"]):
        tasks.append((args.which, key[0], int(key[1]), float(key[2]),
                      int(key[3]), int(key[4]),
                      sorted(int(x) for x in g.seed_index.unique())))
    n_runs = sum(len(t[-1]) for t in tasks)
    print(f"{args.which}: {len(tasks)} cells, {n_runs} runs", flush=True)

    t0 = time.perf_counter()
    with mp.Pool(args.workers) as pool:
        paths = pool.map(run_cell, tasks, chunksize=1)
    rows = [r for p in paths for r in json.loads(Path(p).read_text())]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"robust_rerun_{args.which}_raw.csv", index=False)

    ok = df[df.success]
    print(f"{len(df)} runs in {(time.perf_counter()-t0)/60:.1f} min")
    print(f"  failures (tracked separately): {int((~df.success).sum())}")
    if len(ok):
        print(f"  widening held: {int(ok.widening_held.sum())}/{len(ok)}")
        print(f"  decision changed: {int(ok.changed.sum())}/{len(ok)}")


if __name__ == "__main__":
    main()
