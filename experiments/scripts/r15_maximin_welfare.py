"""Exploratory comparator: the maximin-welfare rule.

    t_MM = min( argmax_s  Wlower^s )

Selects the date with the largest welfare LOWER bound over the estimated
identified set, i.e. it maximises guaranteed welfare. This is NOT minimax
regret, which evaluates loss relative to the best date under the *same*
compatible population; maximin ignores that coupling entirely.

Why it is worth running. The two conservative rules already in the benchmark
are both strict-dominance stopping rules: at date s they ask whether any later
date is provably better (L(s') > U(s)), and stop as soon as none is. That test
is demanding, so they stop early and track immediate allocation. Maximin uses
the magnitude of the lower bounds instead of a dominance test, so it can select
any date. Whether that helps is an open empirical question.

Only LOWER bounds are needed, so `lazy_upper=True` is exactly right: it
computes every lower bound and defers the uppers that only the dominance walk
consults. The dominance decisions themselves are not recomputed here -- they are
already stored per run in main_benchmark_raw_all.csv and are joined at analysis
time.

STATUS: exploratory. This comparator was proposed after inspecting the original
benchmark and is not part of the pre-declared comparison. It must be reported as
a robustness analysis.

Settings match benchmark.py exactly: WORST_CASE_GRID = 201, FROZEN_SET_MODE,
the frozen seed protocol. Results are checkpointed per (config, N, t) cell so a
killed run resumes instead of restarting.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
CKPT = RESULTS / "r15_maximin_cells"
CKPT.mkdir(exist_ok=True)

NS = [1000, 5000, 20000, 100000]
PREFIXES = [2, 3, 4]
WORST_CASE_GRID = 201          # benchmark.py: decision-identical to G=501
N_WORKERS = int(os.environ.get("WTM_WORKERS", "4"))


def run_cell(args):
    dist_name, gamma, T, beta, N, t, seed_list = args
    from wtm.config import config_id, load_frozen, population_cells
    from wtm.moment_projection import (empirical_path_counts,
                                       moments_with_covariance)
    from wtm.rules import evaluate
    from wtm.simulate import simulate_prefix
    from wtm.welfare import PopulationConfig
    from wtm.worst_case import FROZEN_SET_MODE, compute_bounds

    cfg = load_frozen()
    cell = next(c for c in population_cells(cfg)
                if c.gamma == gamma and c.dist_name == dist_name
                and c.T == T and c.beta == beta)
    cell_id = config_id(dist=dist_name, gamma=gamma, T=T, beta=beta, N=N, t=t)
    ck = CKPT / f"{cell_id}.csv"
    if ck.exists():
        return str(ck)

    pop = PopulationConfig(cell.dist, gamma, T, beta)
    base = int(cell_id, 16)

    rows = []
    for s in seed_list:
        sim_seed = (base + int(s)) % (2 ** 32)
        t0 = time.perf_counter()
        data = simulate_prefix(cell.dist, N, T, gamma, t, seed=sim_seed)
        mhat, cov = moments_with_covariance(empirical_path_counts(data, t),
                                            t, gamma)
        row = {"config_id": config_id(dist=dist_name, gamma=gamma, T=T, beta=beta),
               "distribution": dist_name, "gamma": gamma, "T": T, "beta": beta,
               "cell_id": cell_id, "N": N, "prefix_t": t, "seed_index": int(s),
               "rule": "maximin_welfare"}
        try:
            b = compute_bounds(mhat, t, gamma, T, beta,
                               grid_size=WORST_CASE_GRID, cov=cov,
                               set_mode=FROZEN_SET_MODE, lazy_upper=True)
        except Exception as exc:                            # noqa: BLE001
            b = None
            row.update({"success": False,
                        "failure_reason": f"exception:{type(exc).__name__}"})

        if b is not None:
            finite = {int(k): v for k, v in b.lower.items() if np.isfinite(v)}
            if not b.feasible or not finite:
                row.update({"success": False,
                            "failure_reason": b.status or "no_finite_lower_bound"})
            else:
                best = max(finite.values())
                # earliest maximiser, matching every other rule's convention
                date = min(k for k, v in finite.items() if v == best)
                row.update({"success": True, "failure_reason": "",
                            "n_finite_lower": len(finite),
                            "lower_at_selected": float(best),
                            **evaluate(date, pop)})

        if not row.get("success"):
            row.update({"selected_date": -1, "true_optimal_date": pop.t_star,
                        "exact_argmax_match": False,
                        "welfare_efficiency": np.nan,
                        "relative_regret": np.nan, "squared_regret": np.nan})
        row["runtime_s"] = time.perf_counter() - t0
        row["stakes"] = pop.stakes
        rows.append(row)

    pd.DataFrame(rows).to_csv(ck, index=False)
    return str(ck)


def main() -> None:
    from wtm.config import load_frozen, population_cells

    cfg = load_frozen()
    seed_list = cfg["seeds"]["list"]
    cells = [c for c in population_cells(cfg) if c.gamma == 1]
    tasks = [(c.dist_name, c.gamma, c.T, c.beta, N, t, seed_list)
             for c in cells for N in NS for t in PREFIXES]
    done = len(list(CKPT.glob("*.csv")))
    print(f"{len(tasks)} cells x {len(seed_list)} seeds "
          f"({len(tasks) * len(seed_list)} runs), {N_WORKERS} workers; "
          f"{done} cells already checkpointed", flush=True)

    t0 = time.perf_counter()
    n = 0
    with mp.Pool(N_WORKERS) as pool:
        for _ in pool.imap_unordered(run_cell, tasks):
            n += 1
            if n % 20 == 0 or n == len(tasks):
                el = time.perf_counter() - t0
                print(f"[{n}/{len(tasks)} cells] {el:.0f}s "
                      f"({el / 3600:.2f}h)", flush=True)

    df = pd.concat([pd.read_csv(f) for f in sorted(CKPT.glob("*.csv"))],
                   ignore_index=True)
    df.to_csv(RESULTS / "r15_maximin_welfare_raw.csv", index=False)
    core_h = df.runtime_s.sum() / 3600.0
    print(f"\nwrote {RESULTS / 'r15_maximin_welfare_raw.csv'} ({len(df)} rows)")
    print(f"measured core-hours: {core_h:.2f}")
    print(f"failures: {int((~df.success.astype(bool)).sum())}")


if __name__ == "__main__":
    main()
