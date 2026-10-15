"""Stage A: EM and NPMLE trajectories on EXACT observation-cell probabilities.

No sampling noise.  For each reference family P and prefix tau, the fitted
objective is the expected log likelihood  sum_c q_c log qhat_c  with
q_c = E_P[psi_c(p)] computed from P's exact moments (`exact_cell_probs`).
EM is the same update with n/N replaced by q.

Two variants separate population ambiguity from grid misspecification:

  continuous   q from the continuous reference population.  P is generally
               not representable on the fitting grid, so the grid optimum
               leaves a residual KL(q || qhat*) > 0, which is reported.
  realizable   the truth is replaced by P_G, P's mass binned to the fitting
               grid points (`bin_projection`).  Then q_G = Psi w_G is attained
               on the grid, ll* = sum q_G log q_G exactly, and regret is
               scored under P_G's own welfare curve.

Every trajectory is evaluated under all six (T, beta) welfare criteria of its
family, and, in the continuous variant, under both members of the matching
exact compatible pair (they share P's pre-failure law exactly).

Usage:
  python -m experiments.em_mechanism.run_population                 # frozen settings
  python -m experiments.em_mechanism.run_population --pilot         # cap 1e4, tau=2, G=200
Outputs: results/em_mechanism/population/  (full arrays: runs/em_mechanism/population/)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.em_mechanism.common import (CHECKPOINTS, OUT, RUNS, code_fingerprint,
                                             dists, full_schedule, gamma1_cells,
                                             reference_regret, regret_lookup,
                                             write_csv_atomic)

# (tau, G, variant, iteration cap).  Caps are set by measured cost: EM does not
# reach the 1e-8 certificate within any feasible budget on most targets (see
# status files), so the cap is reported as an outcome, not hidden.
VARIANTS = ("continuous", "realizable")
TASKS = ([(2, 200, v, 10_000_000) for v in VARIANTS]
         + [(2, G, v, 1_000_000) for G in (400, 800) for v in VARIANTS]
         + [(t, G, v, 1_000_000) for t in (3, 4) for G in (200, 400) for v in VARIANTS])
N_MOM = 10
SHARP_DELTA = 1e-11


def task_name(tau, G, variant):
    return f"tau{tau}_G{G}_{variant}"


def run_task(args):
    tau, G, variant, cap, outdir, rawdir = args
    from experiments.em_mechanism.core import (DELTA, INIT_NAMES, Welfare, bin_projection,
                                               certified_npmle, em_status, exact_cell_probs,
                                               grid_and_psi, init_weights, run_em)
    from experiments.em_mechanism.pairs import load_pairs
    from wtm.distributions import DiscreteRisk
    from wtm.welfare import PopulationConfig

    name = task_name(tau, G, variant)
    outdir, rawdir = Path(outdir), Path(rawdir)
    settings = {"tau": tau, "G": G, "variant": variant, "cap": cap, "inits": INIT_NAMES,
                "n_mom": N_MOM, "sharp_delta": SHARP_DELTA}
    fp = code_fingerprint(settings)
    done = outdir / f"{name}.done.json"
    if done.exists() and json.loads(done.read_text()).get("fingerprint") == fp:
        return name, "cached"

    grid, Psi = grid_and_psi(G, tau)
    A = Psi.shape[0]
    D = dists()
    cells = gamma1_cells()
    pairs = {p["config_id"]: p for p in load_pairs(tau)}
    configs = sorted({(c.T, c.beta) for c in cells})
    evaluators = {cfg: Welfare(grid, *cfg) for cfg in configs}

    # ---- truths and target probabilities per family
    q_by, truths, L_ref_exact, wG_by = {}, {}, {}, {}
    for dn, dist in D.items():
        if variant == "continuous":
            q = exact_cell_probs(dist, tau)
        else:
            wG = bin_projection(dist, grid)
            wG_by[dn] = wG
            q = Psi @ wG
            L_ref_exact[dn] = float(np.sum(q * np.log(q)))
        q_by[dn] = q
        for c in (c for c in cells if c.dist_name == dn):
            t = {"ref": reference_regret(c)}
            if variant == "continuous":
                p = pairs[c.cid]
                t["lower"], t["upper"] = (np.asarray(p["regret"][0]), np.asarray(p["regret"][1]))
            else:
                keep = wG_by[dn] > 0
                PG = DiscreteRisk(points=tuple(grid[keep]), weights=tuple(wG_by[dn][keep]),
                                  name=f"{dn}_grid{G}")
                t["realizable"] = reference_regret(PopulationConfig(PG, 1, c.T, c.beta))
            truths[(dn, c.T, c.beta)] = t

    rows_meta = [(dn, init) for dn in D for init in INIT_NAMES]
    counts = np.vstack([q_by[dn] for dn, _ in rows_meta])
    W0 = np.vstack([init_weights(init, grid) for _, init in rows_meta])
    sched = full_schedule(cap)
    run = run_em(counts, Psi, grid, W0, sched, evaluators, n_moments=N_MOM,
                 snapshot_at=[k for k in (0, 30, 55, 80, cap) if k <= cap])
    status = em_status(run)

    # ---- NPMLE (existing convention) and a sharper reference solve
    pows = np.vstack([grid ** j for j in range(1, N_MOM + 1)]).T
    np_rows, np_dec, L_ref = [], [], {}
    np_curves = {}
    for dn in D:
        fit = certified_npmle(q_by[dn], grid, tau, Psi, delta=DELTA)
        sharp = certified_npmle(q_by[dn], grid, tau, Psi, delta=SHARP_DELTA)
        idx = [i for i, (d_, _) in enumerate(rows_meta) if d_ == dn]
        em_best = float(run.ll[-1, idx].max())
        if variant == "realizable":
            L_ref[dn] = (L_ref_exact[dn], 0.0, "exact_entropy")
        else:
            cands = [(fit.ll, fit.kkt, "npmle"), (sharp.ll, sharp.kkt, "npmle_sharp"),
                     (em_best, float(run.kkt[-1, idx][np.argmax(run.ll[-1, idx])]), "em_final")]
            L_ref[dn] = max(cands, key=lambda x: x[0])
        for tag, f in (("npmle", fit), ("npmle_sharp", sharp)):
            qh = Psi @ f.weights
            r = {"tau": tau, "G": G, "variant": variant, "distribution": dn, "init": tag,
                 "iteration": -1, "ll": f.ll, "gap": L_ref[dn][0] - f.ll,
                 "gap_bound": L_ref[dn][0] - f.ll + L_ref[dn][1], "kkt": f.kkt,
                 "status": f.status, "n_support": f.n_support, "seconds": f.seconds,
                 "kl_to_target": float(np.sum(q_by[dn] * np.log(q_by[dn] / qh))),
                 "max_abs_cell_diff": float(np.abs(q_by[dn] - qh).max())}
            r.update({f"m{j}": float(f.weights @ pows[:, j - 1]) for j in range(1, N_MOM + 1)})
            r.update({f"q{a}": float(qh[a]) for a in range(A)})
            np_rows.append(r)
            for cfg, ev in evaluators.items():
                C = ev.curves(f.weights[None, :])
                d = int(ev.decide(f.weights[None, :], C)[0])
                np_curves[(dn, tag, cfg)] = C[0]
                tr = truths[(dn, *cfg)]
                np_dec.append({"tau": tau, "G": G, "variant": variant, "distribution": dn,
                               "init": tag, "iteration": -1, "T": cfg[0], "beta": cfg[1],
                               "date": d, **{f"regret_{k}": float(v[d - 1]) for k, v in tr.items()}})

    # ---- fit table (one row per family x init x recorded iteration)
    fit_rows = []
    for i, (dn, init) in enumerate(rows_meta):
        Lr, Lk, _ = L_ref[dn]
        q = q_by[dn]
        for r_, k in enumerate(run.iters):
            qh = run.qhat[r_, i]
            row = {"tau": tau, "G": G, "variant": variant, "distribution": dn, "init": init,
                   "iteration": int(k), "ll": run.ll[r_, i], "gap": Lr - run.ll[r_, i],
                   "gap_bound": Lr - run.ll[r_, i] + Lk, "kkt": run.kkt[r_, i],
                   "min_w": run.min_w[r_, i], "sum_dev": run.sum_dev[r_, i],
                   "kl_to_target": float(np.sum(q * np.log(q / np.maximum(qh, 1e-300)))),
                   "max_abs_cell_diff": float(np.abs(q - qh).max())}
            row.update({f"m{j}": run.mom[r_, i, j - 1] for j in range(1, N_MOM + 1)})
            row.update({f"q{a}": qh[a] for a in range(A)})
            fit_rows.append(row)
    fits = pd.DataFrame(fit_rows + np_rows)

    dec_rows, curve_rows = [], []
    ck = set(c for c in CHECKPOINTS if c <= cap)
    for cfg in configs:
        for i, (dn, init) in enumerate(rows_meta):
            tr = truths[(dn, *cfg)]
            dts = run.dates[cfg][:, i].astype(int)
            block = {"tau": tau, "G": G, "variant": variant, "distribution": dn, "init": init,
                     "T": cfg[0], "beta": cfg[1]}
            df = pd.DataFrame({"iteration": run.iters, "date": dts})
            for k_, R in tr.items():
                df[f"regret_{k_}"] = regret_lookup(R, dts)
            for k_, v in block.items():
                df[k_] = v
            dec_rows.append(df)
            for r_, k in enumerate(run.iters):
                if int(k) in ck:
                    for s, w in enumerate(run.curves[cfg][r_, i], start=1):
                        curve_rows.append({**block, "iteration": int(k), "s": s, "W_hat": w})
        for dn in D:
            for tag in ("npmle", "npmle_sharp"):
                for s, w in enumerate(np_curves[(dn, tag, cfg)], start=1):
                    curve_rows.append({"tau": tau, "G": G, "variant": variant, "distribution": dn,
                                       "init": tag, "T": cfg[0], "beta": cfg[1],
                                       "iteration": -1, "s": s, "W_hat": w})
    decisions = pd.concat(dec_rows + [pd.DataFrame(np_dec)], ignore_index=True)
    curves = pd.DataFrame(curve_rows)
    st = pd.DataFrame([{"tau": tau, "G": G, "variant": variant, "distribution": dn, "init": init,
                        "status": status[i], "first_cert_iter": int(run.first_cert[i]),
                        "final_kkt": run.kkt[-1, i], "mono_violations": int(run.mono_viol[i]),
                        "max_single_step_drop": run.max_drop[i], "finite": bool(run.finite[i]),
                        "min_weight_over_records": float(run.min_w[:, i].min()),
                        "max_sum_dev_over_records": float(run.sum_dev[:, i].max()),
                        "L_ref": L_ref[dn][0], "L_ref_source": L_ref[dn][2],
                        "L_ref_kkt": L_ref[dn][1], "cap": cap,
                        "batch_seconds": run.seconds, "tie_fallbacks": sum(
                            ev.n_fallback for ev in evaluators.values())}
                       for i, (dn, init) in enumerate(rows_meta)])
    # true welfare/regret curves for reference
    true_rows = []
    for (dn, T, beta), tr in truths.items():
        for k_, R in tr.items():
            for s, rv in enumerate(R, start=1):
                true_rows.append({"tau": tau, "G": G, "variant": variant, "distribution": dn,
                                  "T": T, "beta": beta, "truth": k_, "s": s, "regret": rv})

    write_csv_atomic(fits, outdir / f"{name}_fits.csv.gz")
    write_csv_atomic(decisions, outdir / f"{name}_decisions.csv.gz")
    write_csv_atomic(curves, outdir / f"{name}_curves.csv.gz")
    write_csv_atomic(st, outdir / f"{name}_status.csv")
    write_csv_atomic(pd.DataFrame(true_rows), outdir / f"{name}_truth.csv.gz")
    rawdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(rawdir / f"{name}.npz", iters=run.iters, ll=run.ll, kkt=run.kkt,
                        qhat=run.qhat, mom=run.mom,
                        **{f"curves_T{c[0]}_b{c[1]}": run.curves[c] for c in configs},
                        **{f"dates_T{c[0]}_b{c[1]}": run.dates[c] for c in configs},
                        **{f"snap_{k}": v for k, v in run.snapshots.items()},
                        rows=np.array([f"{d}|{i}" for d, i in rows_meta]))
    done.write_text(json.dumps({"fingerprint": fp, "settings": settings,
                                "seconds": run.seconds}, indent=1) + "\n")
    return name, f"{run.seconds:.0f}s"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cap", type=int, default=None, help="override every task's cap")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--only", default=None, help="comma list of task names")
    a = ap.parse_args(argv)
    tasks = TASKS
    sub = "population"
    if a.pilot:
        tasks, sub = [(2, 200, v, 10_000) for v in VARIANTS], "pilot_population"
    if a.cap is not None:
        tasks = [(t, G, v, a.cap) for t, G, v, _ in tasks]
    if a.only:
        keep = set(a.only.split(","))
        tasks = [x for x in tasks if task_name(*x[:3]) in keep]
    outdir, rawdir = OUT / sub, RUNS / sub
    # heaviest first
    tasks = sorted(tasks, key=lambda x: -(x[1] * (x[0] + 1) ** 2 * x[3]))
    args = [(t, G, v, cap, str(outdir), str(rawdir)) for t, G, v, cap in tasks]
    t0 = time.perf_counter()
    with mp.Pool(min(a.workers, len(args))) as pool:
        for name, msg in pool.imap_unordered(run_task, args):
            print(f"[{(time.perf_counter() - t0) / 60:.1f} min] {name}: {msg}", flush=True)


if __name__ == "__main__":
    main()
