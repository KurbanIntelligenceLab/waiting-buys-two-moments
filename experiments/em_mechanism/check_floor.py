"""Does the per-update weight floor (core.WEIGHT_FLOOR = 1e-300) change anything?

Reruns representative cases twice -- with the floor, and with it disabled
(core.WEIGHT_FLOOR set to 0.0 at runtime, which makes the floor a no-op and
restores fit_npmle's exact arithmetic) -- and compares every recorded
diagnostic at every recorded iteration:

  selected dates (all evaluators), log likelihood, KKT residual, fitted cell
  probabilities, moments, first certified iteration, terminal status, and the
  weight snapshots.

Cases (fixed by rule):
  A. exact probabilities, tau = 2, G = 200: all 5 families x 8 initializations,
     all six (T, beta) welfare criteria, cap 1e5;
  B. finite samples, tau = 2, G = 200, N = 1e3 and 1e5: one configuration per
     family at T = 20, beta = 0.002, seeds 1001..1010, 8 initializations, cap 1e5;
  C. the Beta(1,32) cases of A and B continued to 1e6 with the uniform start
     (the family whose tail weights underflow, i.e. where the floor binds).

Usage: python -m experiments.em_mechanism.check_floor
Writes results/em_mechanism/validation/floor_check.{csv,json}
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

import experiments.em_mechanism.core as core
from experiments.em_mechanism.common import OUT, dists, full_schedule, gamma1_cells, write_csv_atomic


def run_pair(counts, Psi, grid, W0, sched, evs):
    out = {}
    for tag, floor in (("floor", 1e-300), ("nofloor", 0.0)):
        core.WEIGHT_FLOOR = floor
        out[tag] = core.run_em(counts, Psi, grid, W0, sched, evs,
                               snapshot_at=[int(sched[-1])])
    core.WEIGHT_FLOOR = 1e-300
    return out


def compare(case, a, b, rows_meta):
    st_a, st_b = core.em_status(a), core.em_status(b)
    k = a.iters[-1]
    rel = lambda x, y: np.abs(x - y) / np.maximum(np.abs(y), 1e-300)
    out = []
    for i, meta in enumerate(rows_meta):
        dm = sum(int((a.dates[e][:, i] != b.dates[e][:, i]).sum()) for e in a.dates)
        out.append({"case": case, **meta,
                    "records": len(a.iters), "cap": int(k),
                    "date_mismatches_all_records_all_criteria": dm,
                    "max_abs_ll_diff": float(np.abs(a.ll[:, i] - b.ll[:, i]).max()),
                    "max_rel_kkt_diff": float(rel(a.kkt[:, i], b.kkt[:, i]).max()),
                    "max_abs_cell_prob_diff": float(np.abs(a.qhat[:, i] - b.qhat[:, i]).max()),
                    "max_rel_moment_diff": float(rel(a.mom[:, i], b.mom[:, i]).max()),
                    "max_abs_weight_diff_at_cap": float(np.abs(a.snapshots[k][i] - b.snapshots[k][i]).max()),
                    "first_cert_floor": int(a.first_cert[i]), "first_cert_nofloor": int(b.first_cert[i]),
                    "status_floor": st_a[i], "status_nofloor": st_b[i],
                    "min_weight_nofloor_at_cap": float(b.snapshots[k][i].min()),
                    "n_weights_below_1e-300_nofloor": int((b.snapshots[k][i] < 1e-300).sum()),
                    "seconds_floor": a.seconds, "seconds_nofloor": b.seconds})
    return out


def main():
    t0 = time.perf_counter()
    grid, Psi = core.grid_and_psi(200, 2)
    cells = gamma1_cells()
    rows = []
    # A: exact probabilities
    configs = sorted({(c.T, c.beta) for c in cells})
    evs = {cfg: core.Welfare(grid, *cfg) for cfg in configs}
    D = dists()
    meta = [{"distribution": dn, "init": init, "N": "exact", "seed_index": -1}
            for dn in D for init in core.INIT_NAMES]
    q = np.vstack([core.exact_cell_probs(D[m["distribution"]], 2) for m in meta])
    W0 = np.vstack([core.init_weights(m["init"], grid) for m in meta])
    r = run_pair(q, Psi, grid, W0, full_schedule(100_000), evs)
    rows += compare("A_exact", r["floor"], r["nofloor"], meta)
    # B: finite samples
    from experiments.em_mechanism.run_finite import simulate_counts
    from experiments.em_mechanism.pairs import load_pairs
    pairs = {p["config_id"]: p for p in load_pairs(2)}
    for c in [c for c in cells if c.T == 20 and c.beta == 0.002]:
        ev = {"e": core.Welfare(grid, c.T, c.beta)}
        for N in (1000, 100000):
            counts, _ = simulate_counts("ref", c, pairs[c.cid], N, list(range(1001, 1011)))
            meta = [{"distribution": c.dist_name, "init": init, "N": N, "seed_index": s}
                    for s in range(1001, 1011) for init in core.INIT_NAMES]
            C = np.vstack([counts[s - 1001] for s in range(1001, 1011) for _ in core.INIT_NAMES])
            W0 = np.vstack([core.init_weights(m["init"], grid) for m in meta])
            r = run_pair(C, Psi, grid, W0, full_schedule(100_000), ev)
            rows += compare("B_finite", r["floor"], r["nofloor"], meta)
    # C: Beta(1,32) to 1e6, uniform start
    b32 = next(c for c in cells if c.dist_name == "beta_1_32" and c.T == 20 and c.beta == 0.002)
    ev = {"e": core.Welfare(grid, b32.T, b32.beta)}
    counts, _ = simulate_counts("ref", b32, pairs[b32.cid], 1000, list(range(1001, 1006)))
    C = np.vstack([core.exact_cell_probs(b32.dist, 2)[None, :], counts])
    meta = [{"distribution": "beta_1_32", "init": "uniform", "N": "exact", "seed_index": -1}] + [
        {"distribution": "beta_1_32", "init": "uniform", "N": 1000, "seed_index": s} for s in range(1001, 1006)]
    W0 = np.tile(core.init_weights("uniform", grid), (len(C), 1))
    r = run_pair(C, Psi, grid, W0, full_schedule(1_000_000), ev)
    rows += compare("C_beta132_1e6", r["floor"], r["nofloor"], meta)

    df = pd.DataFrame(rows)
    write_csv_atomic(df, OUT / "validation" / "floor_check.csv")
    summ = {}
    for case, g in df.groupby("case"):
        summ[case] = {
            "trajectories": int(len(g)),
            "date_mismatches": int(g.date_mismatches_all_records_all_criteria.sum()),
            "max_abs_ll_diff": float(g.max_abs_ll_diff.max()),
            "max_rel_kkt_diff": float(g.max_rel_kkt_diff.max()),
            "max_abs_cell_prob_diff": float(g.max_abs_cell_prob_diff.max()),
            "max_rel_moment_diff": float(g.max_rel_moment_diff.max()),
            "max_abs_weight_diff_at_cap": float(g.max_abs_weight_diff_at_cap.max()),
            "first_cert_mismatches": int((g.first_cert_floor != g.first_cert_nofloor).sum()),
            "status_mismatches": int((g.status_floor != g.status_nofloor).sum()),
            "trajectories_where_floor_binds": int((g["n_weights_below_1e-300_nofloor"] > 0).sum()),
            "seconds_floor": float(g.seconds_floor.drop_duplicates().sum()),
            "seconds_nofloor": float(g.seconds_nofloor.drop_duplicates().sum()),
        }
    summ["total_seconds"] = time.perf_counter() - t0
    (OUT / "validation" / "floor_check.json").write_text(json.dumps(summ, indent=1) + "\n")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
