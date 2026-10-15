"""Full compute-matched bagging baseline, Bboot=200, all 7200 runs.
See results/bagged_full_b200_protocol.md for the pre-registered scope.

Parallelized with multiprocessing.Pool across (config, N, t) cells (360 cells,
20 seeds each), 10 workers, matching this project's established core-hour
accounting convention.
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
TRAJ_DIR = RESULTS / "bagged_full_b200_trajectories"
TRAJ_DIR.mkdir(exist_ok=True)

DELTA = 1e-8
NS = [1000, 5000, 20000, 100000]
PREFIXES = [2, 3, 4]
BBOOT = 200
SEEDS = list(range(1001, 1021))
# 10 is the project's established accounting convention.  Override when the host
# has fewer cores, so the recorded runtime_s stays a real per-run measurement
# instead of absorbing scheduler contention.
N_WORKERS = int(os.environ.get("WTM_WORKERS", "10"))


def bagged_date(Wb):
    return int(np.argmax(Wb.mean(axis=0))) + 1


def single_date(Wsingle):
    return int(np.argmax(np.asarray(Wsingle))) + 1


def run_cell(args):
    """Run all 20 seeds for one (config, N, t) cell. Imports happen inside the
    worker process (required for multiprocessing on Windows, spawn start method)."""
    dist_name, gamma, T, beta, N, t = args
    from wtm.config import config_id, population_cells, load_frozen
    from wtm.lp import group_matrices
    from wtm.npmle import fit_npmle_certified, record_matrix
    from wtm.regret_calibrated import NPMLE_GRID, _trajectory, expected_regret_rule, record_counts_from_prefix
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
    n_atoms = Psi.shape[0]
    cache = [group_matrices(grid, int(s), gamma, T) for s in range(1, T)]

    rows = []
    for seed_index in SEEDS:
        sim_seed = (base + int(seed_index)) % (2 ** 32)
        rule_seed = (sim_seed + 2 ** 31 - 1) % (2 ** 32)
        t0 = time.perf_counter()

        data = simulate_prefix(cell.dist, N, T, gamma, t, seed=sim_seed)
        n = np.asarray(record_counts_from_prefix(data, t), dtype=float)
        Nn = n.sum()

        fit, cert = fit_npmle_certified(n, grid, t, gamma, delta=DELTA, Psi=Psi)
        if fit.n_support > n_atoms or not np.isfinite(fit.loglik):
            rows.append({"cell_id": cell_id, "N": N, "prefix_t": t, "seed_index": seed_index,
                         "SKIP": "invalid_single_fit"})
            continue
        W_single = _trajectory(fit.weights, cache, beta)

        rng = np.random.default_rng(int(rule_seed))
        q = n / Nn
        Nint = int(round(Nn))
        Wbs = []
        for _ in range(BBOOT):
            nb = rng.multinomial(Nint, q).astype(float)
            fb, _cb = fit_npmle_certified(nb, grid, t, gamma, delta=DELTA, Psi=Psi)
            if fb.n_support > n_atoms or not np.isfinite(fb.loglik):
                continue
            Wb = _trajectory(fb.weights, cache, beta)
            Wbs.append(Wb)

        secs = time.perf_counter() - t0
        n_boot_used = len(Wbs)
        if n_boot_used == 0:
            rows.append({"cell_id": cell_id, "N": N, "prefix_t": t, "seed_index": seed_index,
                         "SKIP": "zero_valid_boot_fits"})
            continue

        Wb_mat = np.vstack(Wbs).astype(np.float32)
        np.savez_compressed(TRAJ_DIR / f"{cell_id}_{seed_index}.npz",
                             W_b=Wb_mat, W_single=np.asarray(W_single, dtype=np.float32))

        d_single = single_date(np.asarray(W_single))
        d_bagged = bagged_date(Wb_mat)
        d_rc, rbar = expected_regret_rule(Wb_mat)

        meta = {**cell.as_row(), "cell_id": cell_id, "N": N, "prefix_t": t,
                "seed_index": seed_index, "true_optimal_date": pop.t_star,
                "n_bootstrap_used": n_boot_used, "kkt_residual": cert,
                "n_support": int(fit.n_support), "runtime_s": secs}
        for rule_name, date in [("single", d_single), ("bagged", d_bagged), ("rc", d_rc)]:
            ev = evaluate(int(date), pop)
            rows.append({**meta, "rule": rule_name, "selected_date": date, **ev})
    return rows


def main():
    from wtm.config import load_frozen, population_cells

    t0_all = time.perf_counter()
    cfg = load_frozen()
    cells = [c for c in population_cells(cfg) if c.gamma == 1]
    assert len(cells) == 30

    tasks = [(c.dist_name, c.gamma, c.T, c.beta, N, t)
             for c in cells for N in NS for t in PREFIXES]
    print(f"{len(tasks)} (config,N,t) cells x {len(SEEDS)} seeds = "
          f"{len(tasks) * len(SEEDS)} total runs, {N_WORKERS} workers", flush=True)

    all_rows = []
    done = 0
    with mp.Pool(N_WORKERS) as pool:
        for rows in pool.imap_unordered(run_cell, tasks):
            all_rows.extend(rows)
            done += 1
            if done % 10 == 0 or done == len(tasks):
                elapsed = time.perf_counter() - t0_all
                print(f"[{done}/{len(tasks)} cells] elapsed {elapsed:.0f}s "
                      f"({elapsed/3600:.3f}h wall)", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS / "bagged_full_b200_raw.csv", index=False)
    total_secs = time.perf_counter() - t0_all
    measured_core_s = df["runtime_s"].dropna().drop_duplicates().sum() if "runtime_s" in df else float("nan")
    # runtime_s is per (cell,seed) run, duplicated 3x (once per rule) -- dedupe properly
    per_run = df.dropna(subset=["runtime_s"]).drop_duplicates(subset=["cell_id", "N", "prefix_t", "seed_index"])
    measured_core_hours = per_run["runtime_s"].sum() / 3600.0
    print(f"\nWall time: {total_secs:.1f}s ({total_secs/3600:.3f}h) with {N_WORKERS} workers")
    print(f"Measured core-hours (sum of runtime_s across unique runs): {measured_core_hours:.2f}")
    print(f"wrote {RESULTS / 'bagged_full_b200_raw.csv'} ({len(df)} rows)")


if __name__ == "__main__":
    main()
