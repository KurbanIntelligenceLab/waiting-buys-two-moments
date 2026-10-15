"""Iteration-count robustness check for the early-stopped NPMLE baseline.
See results/early_stopped_robustness_protocol.md. Predeclared set {30,55,80}.
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

ITERS = [30, 55, 80]
NS = [1000, 5000, 20000, 100000]
PREFIXES = [2, 3, 4]
SEEDS = list(range(1001, 1021))
N_WORKERS = int(os.environ.get("WTM_WORKERS", "2"))


def run_cell(args):
    dist_name, gamma, T, beta, N, t, fixed_iter = args
    from wtm.config import config_id, population_cells, load_frozen
    from wtm.lp import group_matrices
    from wtm.npmle import fit_npmle, record_matrix
    from wtm.regret_calibrated import NPMLE_GRID, _trajectory, record_counts_from_prefix
    from wtm.rules import evaluate
    from wtm.simulate import simulate_prefix
    from wtm.welfare import PopulationConfig

    cfg = load_frozen()
    cell = next(c for c in population_cells(cfg)
                if c.gamma == gamma and c.dist_name == dist_name and c.T == T and c.beta == beta)
    pop = PopulationConfig(cell.dist, gamma, T, beta)
    cell_id = config_id(dist=dist_name, gamma=gamma, T=T, beta=beta, N=N, t=t)
    base = int(cell_id, 16)
    grid = np.linspace(0.0, 1.0, NPMLE_GRID)
    Psi = record_matrix(grid, t, gamma)
    cache = [group_matrices(grid, int(s), gamma, T) for s in range(1, T)]

    rows = []
    for seed_index in SEEDS:
        sim_seed = (base + int(seed_index)) % (2 ** 32)
        data = simulate_prefix(cell.dist, N, T, gamma, t, seed=sim_seed)
        n = np.asarray(record_counts_from_prefix(data, t), dtype=float)
        fit = fit_npmle(n, grid, t, gamma, tol=1e-300, max_iter=fixed_iter, Psi=Psi)
        if not np.isfinite(fit.loglik):
            continue
        W = _trajectory(fit.weights, cache, beta)
        date = int(np.argmax(W)) + 1
        ev = evaluate(date, pop)
        rows.append({**cell.as_row(), "cell_id": cell_id, "N": N, "prefix_t": t,
                     "seed_index": seed_index, "true_optimal_date": pop.t_star,
                     "iterations": fixed_iter, "n_support": int(fit.n_support),
                     "loglik": fit.loglik, "rule": f"early_stopped_iter{fixed_iter}",
                     "selected_date": date, **ev})
    return rows


def main():
    from wtm.config import load_frozen, population_cells
    t0_all = time.perf_counter()
    cfg = load_frozen()
    cells = [c for c in population_cells(cfg) if c.gamma == 1]
    tasks = [(c.dist_name, c.gamma, c.T, c.beta, N, t, it)
             for it in ITERS for c in cells for N in NS for t in PREFIXES]
    print(f"{len(tasks)} cells x {len(SEEDS)} seeds = {len(tasks)*len(SEEDS)} runs "
          f"across iters {ITERS}, {N_WORKERS} workers", flush=True)

    all_rows = []
    done = 0
    with mp.Pool(N_WORKERS) as pool:
        for rows in pool.imap_unordered(run_cell, tasks):
            all_rows.extend(rows)
            done += 1
            if done % 60 == 0 or done == len(tasks):
                print(f"[{done}/{len(tasks)}] elapsed {time.perf_counter()-t0_all:.0f}s", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS / "early_stopped_robustness_raw.csv", index=False)
    print(f"\nWall: {time.perf_counter()-t0_all:.1f}s. wrote {len(df)} rows")


if __name__ == "__main__":
    main()
