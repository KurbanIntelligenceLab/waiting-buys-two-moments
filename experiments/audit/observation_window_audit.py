"""Which observation window produced each reported result?

Independent check of results/FINAL_VERIFICATION.md, correction 1. Two tests:

1. Statistic invariance. For simulated prefixes across the benchmark grid, each
   procedure family's input statistic is computed twice: from the data as
   simulated, and from the same data with every period-tau failure undone (the
   individual is treated as still active). A pre-failure (Window A) statistic
   cannot change; a statistic that observes period-tau failures (Window B) does.
   The Window-B record counts are rebuilt from the pre-correction definition
   (commit 197fd9e9) to show the test has power.

2. Archive agreement. For every tau = 2, N in {1e3, 1e5} run behind Tables 3-4,
   the archived decision is recomputed from the frozen seed under both windows:
     certified NPMLE  (main_benchmark_raw_all "npmle" = bagged_full_b200 "single")
     EM, 55 iterations (early_stopped_npmle_raw)
     moment-matched Beta (main_benchmark_raw_all "parametric_beta")
   The archive should agree with Window A in every run, and with Window B only
   where the two windows happen to give the same date.

Writes results/audit/observation_window_audit.{csv,json}. Read-only on
canonical results.

Usage: python experiments/audit/observation_window_audit.py [--workers 4] [--out DIR]
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
KEY = ["cell_id", "N", "prefix_t", "seed_index"]


# -- Window B, transcribed from commit 197fd9e9 ------------------------------
def window_b_counts(data, t):
    counts = []
    for s in range(1, t + 1):
        for k in range(s + 1):
            counts.append(int(np.sum((data.exited_by == s) & (data.y_by_period[s - 1] == k))))
    for k in range(t + 1):
        counts.append(int(np.sum((data.exited_by == -1) & (data.y_by_period[t - 1] == k))))
    return np.array(counts, dtype=float)


def window_b_matrix(grid, t, gamma=1):
    from wtm.observation import phi_exit, phi_survivor
    fs = [phi_exit(s, k, gamma) for s in range(1, t + 1) for k in range(s + 1)]
    fs += [phi_survivor(t, k, gamma) for k in range(t + 1)]
    return np.vstack([f.evaluate(grid) for f in fs])


def undo_period_t_failures(data, t):
    import dataclasses
    return dataclasses.replace(data, exited_by=np.where(data.exited_by == t, -1, data.exited_by))


# -- test 1 -------------------------------------------------------------------
def invariance():
    from wtm.config import load_frozen, population_cells
    from wtm.moment_projection import empirical_path_counts, moments_with_covariance
    from wtm.regret_calibrated import record_counts_from_prefix
    from wtm.simulate import simulate_prefix

    cells = [c for c in population_cells(load_frozen()) if c.gamma == 1]
    out = {"datasets": 0, "with_period_tau_failures": 0,
           "window_a_records_changed": 0, "cohort_moments_changed": 0,
           "cohort_covariance_changed": 0, "path_counts_changed": 0,
           "window_b_records_changed": 0}
    for c in cells:
        for tau in (2, 3, 4):
            for N, seed in ((1000, 1), (100000, 2)):
                d = simulate_prefix(c.dist, N, c.T, 1, tau, seed=seed)
                u = undo_period_t_failures(d, tau)
                out["datasets"] += 1
                out["with_period_tau_failures"] += int((d.exited_by == tau).any())
                out["window_a_records_changed"] += int(not np.array_equal(
                    record_counts_from_prefix(d, tau), record_counts_from_prefix(u, tau)))
                pa, pb = empirical_path_counts(d, tau), empirical_path_counts(u, tau)
                out["path_counts_changed"] += int(not np.array_equal(pa, pb))
                ma, ca = moments_with_covariance(pa, tau, 1)
                mb, cb = moments_with_covariance(pb, tau, 1)
                out["cohort_moments_changed"] += int(not np.allclose(ma, mb, rtol=0, atol=1e-13))
                out["cohort_covariance_changed"] += int(not np.allclose(ca, cb, rtol=0, atol=1e-15))
                out["window_b_records_changed"] += int(not np.array_equal(
                    window_b_counts(d, tau), window_b_counts(u, tau)))
    return out


# -- test 2 -------------------------------------------------------------------
def run_cell(args):
    dist_name, T, beta, N, tau, seeds = args
    from wtm.config import config_id, load_frozen, population_cells
    from wtm.lp import group_matrices
    from wtm.moment_projection import empirical_path_counts, moments_with_covariance
    from wtm.npmle import fit_npmle, fit_npmle_certified, record_matrix
    from wtm.plugin_rules import parametric_beta
    from wtm.regret_calibrated import NPMLE_GRID, _trajectory, record_counts_from_prefix
    from wtm.simulate import simulate_prefix

    cell = next(c for c in population_cells(load_frozen())
                if c.gamma == 1 and c.dist_name == dist_name and c.T == T and c.beta == beta)
    cid = config_id(dist=dist_name, gamma=1, T=T, beta=beta, N=N, t=tau)
    grid = np.linspace(0.0, 1.0, NPMLE_GRID)
    psi = {"A": record_matrix(grid, tau, 1), "B": window_b_matrix(grid, tau)}
    cache = [group_matrices(grid, s, 1, T) for s in range(1, T)]
    w0 = np.full(grid.size, 1.0 / grid.size)
    rows = []
    for sd in seeds:
        data = simulate_prefix(cell.dist, N, T, 1, tau, seed=(int(cid, 16) + sd) % 2 ** 32)
        counts = {"A": np.asarray(record_counts_from_prefix(data, tau), float),
                  "B": window_b_counts(data, tau)}
        row = {"cell_id": cid, "N": N, "prefix_t": tau, "seed_index": sd}
        for w in ("A", "B"):
            fit, _ = fit_npmle_certified(counts[w], grid, tau, 1, delta=1e-8, Psi=psi[w])
            ok = fit.n_support <= psi[w].shape[0] and np.isfinite(fit.loglik)
            row[f"npmle_{w}"] = int(np.argmax(_trajectory(fit.weights, cache, beta))) + 1 if ok else -1
            em = fit_npmle(counts[w], grid, tau, 1, tol=1e-300, max_iter=55, Psi=psi[w], w0=w0)
            row[f"em55_{w}"] = int(np.argmax(_trajectory(em.weights, cache, beta))) + 1
        mhat, _ = moments_with_covariance(empirical_path_counts(data, tau), tau, 1)
        o = parametric_beta(counts["A"], T, 1, tau, beta, moments=mhat)
        row["beta_A"] = o.selected_date if o.success else -1
        rows.append(row)
    return rows


def agreement(fresh: pd.DataFrame) -> list[dict]:
    mb = pd.read_csv(RESULTS / "main_benchmark_raw_all.csv", low_memory=False)
    b200 = pd.read_csv(RESULTS / "bagged_full_b200_raw.csv")
    es = pd.read_csv(RESULTS / "early_stopped_npmle_raw.csv")
    wb = RESULTS / "window_b"
    mb_b = pd.read_csv(wb / "main_benchmark_raw_all.csv", low_memory=False)
    es_b = pd.read_csv(wb / "early_stopped_npmle_raw.csv")
    b200_b = pd.read_csv(wb / "bagged_full_b200_raw.csv")

    def archived(df, rule):
        x = df[df.rule == rule] if rule else df
        x = x.drop_duplicates(KEY)
        return x.set_index(KEY).selected_date

    checks = [
        ("Table 3 certified NPMLE (bagged_full_b200 'single')", archived(b200, "single"), "npmle"),
        ("Table 3/4 certified NPMLE (main_benchmark 'npmle')", archived(mb, "npmle"), "npmle"),
        ("Table 3/4 EM55 (early_stopped_npmle_raw)", archived(es, "early_stopped_npmle"), "em55"),
        ("Window-B archive: certified NPMLE (window_b/bagged_full_b200 'single')",
         archived(b200_b, "single"), "npmle"),
        ("Window-B archive: EM55 (window_b/early_stopped_npmle_raw)",
         archived(es_b, "early_stopped_npmle"), "em55"),
        ("Table 4 Beta (main_benchmark 'parametric_beta')", archived(mb, "parametric_beta"), "beta"),
        ("Window-B archive: Beta (window_b/main_benchmark 'parametric_beta')",
         archived(mb_b, "parametric_beta"), "beta"),
    ]
    f = fresh.set_index(KEY)
    out = []
    for label, arch, proc in checks:
        j = f.join(arch.rename("archived"), how="inner")
        j = j[j.archived.notna()]
        rec = {"check": label, "runs_compared": int(len(j))}
        for w in ("A", "B"):
            col = f"{proc}_{w}"
            if col in j:
                ok = j[col] >= 1
                rec[f"agree_window_{w}"] = int((j[col][ok] == j.archived[ok]).sum())
                rec[f"comparable_{w}"] = int(ok.sum())
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(RESULTS / "audit"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    inv = invariance()
    print("statistic invariance:", json.dumps(inv, indent=1), flush=True)

    from wtm.config import load_frozen, population_cells
    seeds = load_frozen()["seeds"]["list"]
    tasks = [(c.dist_name, c.T, c.beta, N, 2, seeds)
             for c in population_cells(load_frozen()) if c.gamma == 1 for N in (1000, 100000)]
    with mp.Pool(args.workers) as pool:
        fresh = pd.DataFrame([r for rs in pool.map(run_cell, tasks) for r in rs])
    fresh.to_csv(out / "observation_window_recomputed_tau2.csv", index=False)
    agree = agreement(fresh)
    for a in agree:
        print(a, flush=True)
    rep = {"statistic_invariance": inv, "archive_agreement": agree,
           "runs": int(len(fresh)), "runtime_s": time.perf_counter() - t0}
    (out / "observation_window_audit.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(f"done in {rep['runtime_s']:.0f}s")


if __name__ == "__main__":
    main()
