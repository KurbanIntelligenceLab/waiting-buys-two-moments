"""Early-stopped NPMLE baseline (R16), full frozen benchmark.
See results/early_stopped_npmle_protocol.md -- iteration count (55) fixed
before this run, from a disjoint diagnostic sample.
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

FIXED_ITER = 55
NS = [1000, 5000, 20000, 100000]
PREFIXES = [2, 3, 4]
SEEDS = list(range(1001, 1021))
N_WORKERS = int(os.environ.get("WTM_WORKERS", "2"))


def run_cell(args):
    dist_name, gamma, T, beta, N, t = args
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
        t0 = time.perf_counter()
        data = simulate_prefix(cell.dist, N, T, gamma, t, seed=sim_seed)
        n = np.asarray(record_counts_from_prefix(data, t), dtype=float)

        fit = fit_npmle(n, grid, t, gamma, tol=1e-300, max_iter=FIXED_ITER, Psi=Psi)
        secs = time.perf_counter() - t0
        if not np.isfinite(fit.loglik):
            rows.append({"cell_id": cell_id, "N": N, "prefix_t": t, "seed_index": seed_index,
                         "rule": "early_stopped_npmle", "success": False,
                         "failure_reason": "non_finite_loglik", "selected_date": 1,
                         "runtime_s": secs})
            continue
        W = _trajectory(fit.weights, cache, beta)
        date = int(np.argmax(W)) + 1
        ev = evaluate(date, pop)
        meta = {**cell.as_row(), "cell_id": cell_id, "N": N, "prefix_t": t,
                "seed_index": seed_index, "true_optimal_date": pop.t_star,
                "n_support": int(fit.n_support), "iterations": FIXED_ITER, "runtime_s": secs}
        rows.append({**meta, "rule": "early_stopped_npmle", "success": True,
                     "failure_reason": "", "selected_date": date, **ev})
    return rows


def main():
    from wtm.config import load_frozen, population_cells

    t0_all = time.perf_counter()
    cfg = load_frozen()
    cells = [c for c in population_cells(cfg) if c.gamma == 1]
    assert len(cells) == 30
    tasks = [(c.dist_name, c.gamma, c.T, c.beta, N, t)
             for c in cells for N in NS for t in PREFIXES]
    print(f"{len(tasks)} cells x {len(SEEDS)} seeds = {len(tasks)*len(SEEDS)} runs, "
          f"{N_WORKERS} workers, {FIXED_ITER} EM iterations fixed", flush=True)

    all_rows = []
    done = 0
    with mp.Pool(N_WORKERS) as pool:
        for rows in pool.imap_unordered(run_cell, tasks):
            all_rows.extend(rows)
            done += 1
            if done % 30 == 0 or done == len(tasks):
                elapsed = time.perf_counter() - t0_all
                print(f"[{done}/{len(tasks)}] elapsed {elapsed:.0f}s", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS / "early_stopped_npmle_raw.csv", index=False)
    total_secs = time.perf_counter() - t0_all
    core_hours = df["runtime_s"].sum() / 3600.0
    print(f"\nWall time: {total_secs:.1f}s ({total_secs/3600:.3f}h) with {N_WORKERS} workers")
    print(f"Measured core-hours (sum runtime_s): {core_hours:.3f}")
    print(f"wrote {RESULTS / 'early_stopped_npmle_raw.csv'} ({len(df)} rows)")


if __name__ == "__main__":
    main()
