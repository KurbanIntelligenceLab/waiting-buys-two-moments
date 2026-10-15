"""Validation of the EM-mechanism study against the archive and for numerical soundness.

Checks (results/em_mechanism/validation/):

 V1  "55 iterations" = exactly 55 EM updates from the uniform state (iteration 0):
     `fit_npmle(tol=1e-300, max_iter=K)` reports K iterations on every archived
     tau=2 dataset (N in {1e3, 1e5}, seeds 1001..1020, all 30 configurations).
 V2  The batch engine reproduces `fit_npmle`'s weights and decisions at K = 30, 55, 80.
 V3  Reproduction of archived decisions: EM30/55/80 (early_stopped_robustness_raw.csv,
     early_stopped_npmle_raw.csv) and certified NPMLE (main_benchmark_raw_all.csv,
     rule 'npmle') on the overlapping ref datasets; EM30/55/80 and certified NPMLE of
     reconstruction-v1 (matched_info_pairs_raw.csv) on the overlapping pair datasets.
 V4  Numerical: monotone likelihood within MONO_TOL, weights >= 0 and summing to one,
     valid fitted cell probabilities, finite values; status counts.
 V5  Welfare convention: batch `Welfare` equals `_trajectory`; welfare of a
     grid-supported distribution via `Welfare` equals `PopulationConfig` on the
     same DiscreteRisk (same greedy allocation convention for truth and estimate).
 V6  NPMLE certificate: distribution of the KKT residual, share meeting 1e-8.
 V7  Late EM versus NPMLE: likelihood difference, fitted-cell difference, weight
     distance, welfare-curve difference and date agreement on the same data.

Usage: python -m experiments.em_mechanism.validate [--finite finite] [--population population]
"""
from __future__ import annotations

import argparse
import glob
import json
import time

import numpy as np
import pandas as pd

from experiments.em_mechanism.common import OUT, ROOT, gamma1_cells, write_csv_atomic

RES = ROOT / "results"


def v1_v2(report, n_check=1200):
    from experiments.em_mechanism.core import Welfare, grid_and_psi, init_weights, run_em
    from wtm.config import config_id
    from wtm.npmle import fit_npmle
    from wtm.regret_calibrated import _trajectory, record_counts_from_prefix
    from wtm.simulate import simulate_prefix
    grid, Psi = grid_and_psi(200, 2)
    iters_ok, n_fit, max_rel, date_mm, n_dates = True, 0, 0.0, 0, 0
    for cell in gamma1_cells():
        ev = Welfare(grid, cell.T, cell.beta)
        for N in (1000, 100000):
            base = int(config_id(dist=cell.dist_name, gamma=1, T=cell.T, beta=cell.beta, N=N, t=2), 16)
            counts = np.array([record_counts_from_prefix(
                simulate_prefix(cell.dist, N, cell.T, 1, 2, seed=(base + s) % 2 ** 32), 2)
                for s in range(1001, 1021)])
            run = run_em(counts, Psi, grid, np.tile(init_weights("uniform", grid), (20, 1)),
                         np.arange(0, 81), {"e": ev}, snapshot_at=(30, 55, 80))
            for K in (30, 55, 80):
                for b in range(20):
                    f = fit_npmle(counts[b], grid, 2, 1, tol=1e-300, max_iter=K, Psi=Psi)
                    n_fit += 1
                    iters_ok &= (f.iterations == K)
                    max_rel = max(max_rel, float(np.abs(f.weights - run.snapshots[K][b]).max()
                                                 / f.weights.max()))
                    d = int(np.argmax(_trajectory(f.weights, ev.cache, cell.beta))) + 1
                    date_mm += int(d != run.dates["e"][K, b])
                    n_dates += 1
    report["V1"] = {"fits_checked": n_fit, "all_report_exactly_K_updates": bool(iters_ok),
                    "initial_state": "uniform 1/G = iteration 0; fit_npmle evaluates ll before each "
                                     "update, so max_iter=K returns the state after K updates"}
    report["V2"] = {"fits_checked": n_fit, "max_relative_weight_difference": max_rel,
                    "date_mismatches": date_mm, "dates_checked": n_dates}


def load_finite(sub):
    files = sorted(glob.glob(str(OUT / sub / "cells" / "*_table.csv.gz")))
    if not files:
        return None, None
    t = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    s = pd.concat([pd.read_csv(f.replace("_table", "_status")) for f in files], ignore_index=True)
    return t, s


def v3(report, t):
    out = {}
    em = t[(t.method == "em") & (t.init == "uniform")]
    ref = em[em.kind == "ref"]
    arch = pd.read_csv(RES / "early_stopped_robustness_raw.csv")
    arch = arch[arch.prefix_t == 2]
    rows = []
    for K in (30, 55, 80):
        m = ref[ref.iteration == K].merge(arch[arch.iterations == K], on=["config_id", "N", "seed_index"],
                                          suffixes=("", "_arch"))
        rows.append({"source": "early_stopped_robustness_raw", "K": K, "n": len(m),
                     "date_mismatches": int((m.date != m.selected_date).sum()),
                     "max_abs_loglik_diff": float((m.ll * m.N - m.loglik).abs().max()) if len(m) else np.nan})
    a55 = pd.read_csv(RES / "early_stopped_npmle_raw.csv")
    a55 = a55[a55.prefix_t == 2]
    m = ref[ref.iteration == 55].merge(a55, on=["config_id", "N", "seed_index"], suffixes=("", "_arch"))
    rows.append({"source": "early_stopped_npmle_raw", "K": 55, "n": len(m),
                 "date_mismatches": int((m.date != m.selected_date).sum())})
    mb = pd.read_csv(RES / "main_benchmark_raw_all.csv", low_memory=False)
    mb = mb[(mb.rule == "npmle") & (mb.prefix_t == 2)]
    npm = t[(t.method == "npmle") & (t.kind == "ref")]
    m = npm.merge(mb, on=["config_id", "N", "seed_index"], suffixes=("", "_arch"))
    rows.append({"source": "main_benchmark_raw_all (npmle)", "K": -1, "n": len(m),
                 "date_mismatches": int((m.date != m.selected_date).sum()),
                 "raw_date_mismatches": int((m.npmle_date != m.selected_date).sum()),
                 "archived_failures": int((~m.success.astype(bool)).sum()),
                 "our_invalid": int((m.npmle_status == "invalid").sum())})
    v1 = pd.read_csv(RES / "matched_info_pairs_raw.csv")
    pe = t[(t.kind == "pair")]
    for proc, sel in (("em_30", (pe.method == "em") & (pe.init == "uniform") & (pe.iteration == 30)),
                      ("em_55", (pe.method == "em") & (pe.init == "uniform") & (pe.iteration == 55)),
                      ("em_80", (pe.method == "em") & (pe.init == "uniform") & (pe.iteration == 80)),
                      ("certified_npmle", pe.method == "npmle")):
        m = pe[sel].merge(v1[v1.procedure == proc], on=["config_id", "N", "seed_index"],
                          suffixes=("", "_v1"))
        ok = m.success.astype(bool) & (m.date > 0)
        rows.append({"source": f"matched_info_pairs_raw ({proc})", "K": -1, "n": len(m),
                     "sim_seed_mismatches": int((m.sim_seed != m.sim_seed_v1).sum()),
                     "date_mismatches_among_valid": int((m.date[ok] != m.selected_date[ok]).sum()),
                     "validity_mismatches": int(((m.date > 0) != m.success.astype(bool)).sum())})
    out = pd.DataFrame(rows)
    write_csv_atomic(out, OUT / "validation" / "V3_reproduction.csv")
    report["V3"] = out.to_dict(orient="records")


def v4_v6(report, t, s, label):
    em = t[t.method == "em"]
    qcols = [c for c in t.columns if c.startswith("q") and c[1:].isdigit()]
    qs = em[qcols].to_numpy()
    st_full = s[~s.init.str.startswith("cvfold")]
    npm = t[t.method == "npmle"]
    report[f"V4_{label}"] = {
        "trajectories": int(len(s)),
        "monotonicity_tolerance": 1e-13,
        "trajectories_with_violations": int((s.mono_violations > 0).sum()),
        "max_single_step_decrease": float(s.max_single_step_drop.max()),
        "min_weight_over_all_records": float(s.min_weight_over_records.min()),
        "max_abs_sum_minus_one": float(s.max_sum_dev_over_records.max()),
        "all_finite": bool(s.finite.all()),
        "min_fitted_cell_prob": float(np.nanmin(qs)),
        "max_abs_cell_sum_minus_one": float(np.nanmax(np.abs(qs.sum(1) - 1))),
        "status_counts_full_data": st_full.status.value_counts().to_dict(),
        "status_counts_by_N": {str(k): v.status.value_counts().to_dict()
                               for k, v in st_full.groupby("N")},
    }
    k = npm.kkt.to_numpy()
    report[f"V6_{label}"] = {
        "npmle_fits": int(len(npm)), "status_counts": npm.npmle_status.value_counts().to_dict(),
        "kkt_median": float(np.median(k)), "share_kkt_le_1e8": float((k <= 1e-8).mean()),
        "share_kkt_le_1e6": float((k <= 1e-6).mean()), "kkt_max": float(k.max()),
        "support_size_mean": float(npm.n_support.mean()),
    }


def v5(report):
    from experiments.em_mechanism.core import Welfare, grid_and_psi
    from wtm.distributions import DiscreteRisk
    from wtm.regret_calibrated import _trajectory
    from wtm.welfare import PopulationConfig
    rng = np.random.default_rng(7)
    grid, _ = grid_and_psi(200, 2)
    worst_traj, worst_pop = 0.0, 0.0
    for cell in gamma1_cells()[::5]:
        ev = Welfare(grid, cell.T, cell.beta)
        W = rng.dirichlet(np.full(200, 0.3), size=20)
        C = ev.curves(W)
        for b in range(20):
            ref = _trajectory(W[b], ev.cache, cell.beta)
            worst_traj = max(worst_traj, float(np.abs(ref - C[b]).max() / ref.max()))
        keep = W[0] > 0
        P = PopulationConfig(DiscreteRisk(points=tuple(grid[keep]), weights=tuple(W[0][keep] / W[0][keep].sum())),
                             1, cell.T, cell.beta)
        worst_pop = max(worst_pop, float(np.abs(P.W - C[0]).max() / P.W.max()))
    report["V5"] = {"max_rel_diff_batch_vs__trajectory": worst_traj,
                    "max_rel_diff_batch_vs_PopulationConfig_on_grid_distribution": worst_pop}


def v7(report, sub):
    """Late EM (cap, uniform init) versus NPMLE on the same datasets (full-precision arrays)."""
    from experiments.em_mechanism.core import grid_and_psi
    grid, Psi = grid_and_psi(200, 2)
    rows = []
    from experiments.em_mechanism.snapshots import finite_cell
    keys = sorted(p.name.replace(".done.json", "") for p in (OUT / sub / "cells").glob("*.done.json"))
    for key in keys:
        cellsnap = finite_cell(sub, key)
        Wcap = cellsnap.at(cellsnap.cap)
        npW, npC = cellsnap.npmle_weights, cellsnap.npmle_curves
        for seed_i in range(Wcap.shape[0]):
            w_em, w_np = Wcap[seed_i], npW[seed_i]
            q_em, q_np = Psi @ w_em, Psi @ w_np
            c_em, c_np = cellsnap.end_curves[seed_i], npC[seed_i]
            cdf_em, cdf_np = np.cumsum(w_em), np.cumsum(w_np)
            rows.append({"cell": key, "seed_row": seed_i,
                         "max_abs_cell_diff": float(np.abs(q_em - q_np).max()),
                         "kl_npmle_vs_em": float(np.sum(q_np * np.log(q_np / q_em))),
                         "weight_tv": float(0.5 * np.abs(w_em - w_np).sum()),
                         "weight_w1": float(np.abs(cdf_em - cdf_np).sum() * (grid[1] - grid[0])),
                         "welfare_curve_max_rel_diff": float(np.abs(c_em - c_np).max() / max(c_np.max(), 1e-300)),
                         "em_support_1e-6": int((w_em > 1e-6).sum()),
                         "npmle_support_1e-10": int((w_np > 1e-10).sum()),
                         "date_em": int(np.argmax(c_em)) + 1, "date_npmle": int(np.argmax(c_np)) + 1})
    if not rows:
        return
    df = pd.DataFrame(rows)
    write_csv_atomic(df, OUT / "validation" / f"V7_late_em_vs_npmle_{sub}.csv.gz")
    report[f"V7_{sub}"] = {
        "pairs_compared": int(len(df)),
        "median_max_abs_cell_diff": float(df.max_abs_cell_diff.median()),
        "p90_max_abs_cell_diff": float(df.max_abs_cell_diff.quantile(.9)),
        "median_weight_tv": float(df.weight_tv.median()),
        "median_welfare_curve_max_rel_diff": float(df.welfare_curve_max_rel_diff.median()),
        "share_same_date": float((df.date_em == df.date_npmle).mean()),
        "median_em_support_1e-6": float(df["em_support_1e-6"].median()),
        "median_npmle_support": float(df["npmle_support_1e-10"].median()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--finite", default="finite")
    ap.add_argument("--skip-v1", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    report = {}
    if not a.skip_v1:
        v1_v2(report)
    v5(report)
    t, s = load_finite(a.finite)
    if t is not None:
        v3(report, t)
        v4_v6(report, t, s, a.finite)
        v7(report, a.finite)
    pop = sorted(glob.glob(str(OUT / "population" / "*_status.csv")))
    if pop:
        ps = pd.concat([pd.read_csv(f) for f in pop])
        report["V4_population"] = {
            "trajectories": int(len(ps)),
            "trajectories_with_violations": int((ps.mono_violations > 0).sum()),
            "max_single_step_decrease": float(ps.max_single_step_drop.max()),
            "min_weight": float(ps.min_weight_over_records.min()),
            "max_abs_sum_minus_one": float(ps.max_sum_dev_over_records.max()),
            "status_counts": {f"tau{k[0]}_G{k[1]}_{k[2]}": v.status.value_counts().to_dict()
                              for k, v in ps.groupby(["tau", "G", "variant"])}}
    report["seconds"] = time.perf_counter() - t0
    (OUT / "validation").mkdir(parents=True, exist_ok=True)
    (OUT / "validation" / "validation_report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
