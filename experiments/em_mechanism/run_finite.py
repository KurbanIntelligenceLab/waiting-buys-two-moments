"""Stage B: EM and NPMLE trajectories on finite samples (common pre-failure window).

Cells are (kind, configuration, N) at tau = 2 on the 200-point grid.

  kind = ref    data drawn from the reference population with the archive's
                seed policy (early_stopped_npmle.py / early_stopped_robustness.py):
                sim_seed = (int(config_id(dist, gamma=1, T, beta, N, t=2), 16) + seed) mod 2^32.
                Seeds 1001..1020 are therefore the archived benchmark datasets.
  kind = pair   data drawn from the lower member of the exact compatible pair with
                reconstruction-v1's seed policy (matched_info_pairs.py):
                sim_seed = (int(config_id(dist='pair_<config_id>', gamma=1, T, beta, N, t=2), 16)
                + seed) mod 2^32.  Seeds 1001..1005 are reconstruction-v1's datasets.

Both members of a pair, and the reference population itself, induce the same
pre-failure observation law (checked to 6e-16 in pairs.py), so ONE dataset is a
valid draw under each of them: every trajectory, from both kinds, is scored
under the reference population and under both pair members.  No procedure sees
period-2 failures or which population is true.

Every dataset is fitted by: EM from each of the eight initializations
(core.INIT_NAMES), the certified NPMLE (existing convention), and, as a
label-free stopping comparator, 2-fold cross-validated EM from the uniform
initialization (the iteration maximizing held-out log likelihood, applied to the
full-data uniform trajectory).  The CV split uses
rng = default_rng((sim_seed + 2^30) mod 2^32).multivariate_hypergeometric.

Usage:
  python -m experiments.em_mechanism.run_finite --pilot            # 2 configs x 2 N x 5 seeds, cap 1e5
  python -m experiments.em_mechanism.run_finite                    # frozen: 60 ref + 60 pair cells
  python -m experiments.em_mechanism.run_finite --kinds ref --seeds 1001-1020   # subsets
Per-cell checkpoints make the run resumable; rerunning skips completed cells
whose fingerprint (code + settings) matches.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.em_mechanism.common import (OUT, RUNS, code_fingerprint, full_schedule,
                                             gamma1_cells, reference_regret, regret_lookup,
                                             write_csv_atomic)

TAU = 2
G_DEFAULT = 200
NS = (1000, 100000)
SEEDS = tuple(range(1001, 1051))
CAP = 1_000_000
N_MOM = 10
TABLE_ITERS = (0, 1, 3, 10, 30, 55, 80, 100, 300, 1000, 3000, 10_000, 30_000, 100_000,
               300_000, 1_000_000)
CURVE_ITERS = (0, 30, 55, 80)          # plus the cap: welfare curves kept in runs/


def cell_key(kind, cid, N, G):
    return f"{kind}_{cid}_N{N}_G{G}"


def simulate_counts(kind, cell, pair, N, seeds):
    from wtm.config import config_id
    from wtm.regret_calibrated import record_counts_from_prefix
    from wtm.simulate import simulate_prefix
    from experiments.em_mechanism.pairs import member_dist
    if kind == "ref":
        base = int(config_id(dist=cell.dist_name, gamma=1, T=cell.T, beta=cell.beta, N=N, t=TAU), 16)
        dist = cell.dist
    else:
        base = int(config_id(dist=f"pair_{cell.cid}", gamma=1, T=cell.T, beta=cell.beta, N=N, t=TAU), 16)
        dist = member_dist(pair, 0)
    sims = [(base + s) % 2 ** 32 for s in seeds]
    counts = np.array([record_counts_from_prefix(simulate_prefix(dist, N, cell.T, 1, TAU, seed=ss), TAU)
                       for ss in sims], dtype=float)
    return counts, np.array(sims, dtype=np.int64)


def run_cell(args):
    kind, cid, N, G, seeds, cap, outdir, rawdir, inits, reuse = args
    from experiments.em_mechanism.core import (Welfare, certified_npmle, em_status, grid_and_psi,
                                               init_weights, run_em)
    from experiments.em_mechanism.pairs import load_pairs
    key = cell_key(kind, cid, N, G)
    outdir, rawdir = Path(outdir), Path(rawdir)
    settings = {"kind": kind, "cid": cid, "N": N, "G": G, "seeds": list(seeds), "cap": cap,
                "inits": list(inits), "tau": TAU, "n_mom": N_MOM}
    fp = code_fingerprint(settings)
    done = outdir / "cells" / f"{key}.done.json"
    if done.exists():
        meta = json.loads(done.read_text())
        if meta.get("fingerprint") == fp or (reuse and meta.get("settings") == settings):
            return key, "cached", 0.0

    cell = next(c for c in gamma1_cells() if c.cid == cid)
    pair = {p["config_id"]: p for p in load_pairs(TAU)}[cid]
    truths = {"ref": reference_regret(cell), "lower": np.asarray(pair["regret"][0]),
              "upper": np.asarray(pair["regret"][1])}
    grid, Psi = grid_and_psi(G, TAU)
    A = Psi.shape[0]
    ev = Welfare(grid, cell.T, cell.beta)
    counts, sims = simulate_counts(kind, cell, pair, N, seeds)
    S = len(seeds)

    # CV folds (uniform initialization only)
    folds = []
    for i, ss in enumerate(sims):
        rng = np.random.default_rng((int(ss) + 2 ** 30) % 2 ** 32)
        n1 = rng.multivariate_hypergeometric(counts[i].astype(np.int64), int(N // 2)).astype(float)
        folds.append((n1, counts[i] - n1))
    rows_meta = [(i, init) for i in range(S) for init in inits]
    rows_meta += [(i, f"cvfold{f}") for i in range(S) for f in (0, 1)]
    C = np.vstack([counts[i] if not tag.startswith("cvfold") else folds[i][int(tag[-1])]
                   for i, tag in rows_meta])
    W0 = np.vstack([init_weights(tag if not tag.startswith("cvfold") else "uniform", grid)
                    for _, tag in rows_meta])
    sched = full_schedule(cap)
    snap = sorted({k for k in CURVE_ITERS if k <= cap} | {cap})
    run = run_em(C, Psi, grid, W0, sched, {"e": ev}, n_moments=N_MOM, snapshot_at=snap)
    status = em_status(run)

    # NPMLE per dataset
    pows = np.vstack([grid ** j for j in range(1, N_MOM + 1)]).T
    npm = [certified_npmle(counts[i], grid, TAU, Psi) for i in range(S)]
    npm_W = np.vstack([f.weights for f in npm])
    npm_C = ev.curves(npm_W)
    npm_d = ev.decide(npm_W, npm_C)

    # likelihood reference per dataset: best of NPMLE and all full-data EM finals
    Lref, Lref_k, Lref_src = np.empty(S), np.empty(S), []
    for i in range(S):
        idx = [r for r, (j, tag) in enumerate(rows_meta) if j == i and not tag.startswith("cvfold")]
        b = idx[int(np.argmax(run.ll[-1, idx]))]
        if npm[i].ll >= run.ll[-1, b] and npm[i].status != "invalid":
            Lref[i], Lref_k[i] = npm[i].ll, npm[i].kkt
            Lref_src.append("npmle")
        else:
            Lref[i], Lref_k[i] = run.ll[-1, b], run.kkt[-1, b]
            Lref_src.append(f"em_{rows_meta[b][1]}")

    # CV-selected iteration (earliest maximizer of the 2-fold held-out log likelihood)
    cv_k, cv_d = np.empty(S, dtype=np.int64), np.empty(S, dtype=np.int64)
    uni = {i: r for r, (i, tag) in enumerate(rows_meta) if tag == "uniform"}
    for i in range(S):
        r0 = [r for r, (j, tag) in enumerate(rows_meta) if j == i and tag == "cvfold0"][0]
        r1 = [r for r, (j, tag) in enumerate(rows_meta) if j == i and tag == "cvfold1"][0]
        h0 = (folds[i][1] * np.log(np.maximum(run.qhat[:, r0], 1e-300))).sum(1)
        h1 = (folds[i][0] * np.log(np.maximum(run.qhat[:, r1], 1e-300))).sum(1)
        h = (h0 + h1) / N
        j = int(np.argmax(h))                 # earliest maximizer
        cv_k[i] = run.iters[j]
        cv_d[i] = run.dates["e"][j, uni[i]] if "uniform" in inits else -1

    # ---- outputs
    base = {"kind": kind, "config_id": cid, "distribution": cell.dist_name, "T": cell.T,
            "beta": cell.beta, "N": N, "tau": TAU, "G": G}
    tab_set = set(t for t in TABLE_ITERS if t <= cap) | {cap}
    tab_rows = []
    rec_idx = [r for r, k in enumerate(run.iters) if int(k) in tab_set]
    for r, (i, tag) in enumerate(rows_meta):
        if tag.startswith("cvfold"):
            continue
        for ri in rec_idx:
            d = int(run.dates["e"][ri, r])
            row = {**base, "seed_index": seeds[i], "sim_seed": int(sims[i]), "method": "em",
                   "init": tag, "iteration": int(run.iters[ri]), "ll": run.ll[ri, r],
                   "gap": Lref[i] - run.ll[ri, r], "gap_bound": Lref[i] - run.ll[ri, r] + Lref_k[i],
                   "kkt": run.kkt[ri, r], "date": d, "npmle_date": int(npm_d[i]),
                   **{f"regret_{k}": float(v[d - 1]) for k, v in truths.items()},
                   **{f"q{a}": run.qhat[ri, r, a] for a in range(A)},
                   **{f"m{j}": run.mom[ri, r, j - 1] for j in range(1, N_MOM + 1)}}
            tab_rows.append(row)
    for i in range(S):
        f = npm[i]
        qh = Psi @ f.weights
        d = int(npm_d[i])
        ok = f.status != "invalid"
        tab_rows.append({**base, "seed_index": seeds[i], "sim_seed": int(sims[i]), "method": "npmle",
                         "init": "npmle", "iteration": -1, "ll": f.ll, "gap": Lref[i] - f.ll,
                         "gap_bound": Lref[i] - f.ll + Lref_k[i], "kkt": f.kkt, "date": d if ok else -1,
                         "npmle_date": d, "npmle_status": f.status, "n_support": f.n_support,
                         "runtime_s": f.seconds,
                         **{f"regret_{k}": float(v[d - 1]) if ok else np.nan for k, v in truths.items()},
                         **{f"q{a}": qh[a] for a in range(A)},
                         **{f"m{j}": float(f.weights @ pows[:, j - 1]) for j in range(1, N_MOM + 1)}})
        if "uniform" in inits:
            dcv = int(cv_d[i])
            tab_rows.append({**base, "seed_index": seeds[i], "sim_seed": int(sims[i]), "method": "em_cv",
                             "init": "uniform", "iteration": int(cv_k[i]), "date": dcv,
                             "npmle_date": int(npm_d[i]),
                             **{f"regret_{k}": float(v[dcv - 1]) for k, v in truths.items()}})
    table = pd.DataFrame(tab_rows)

    per_row_sec = run.seconds / C.shape[0]
    st = pd.DataFrame([{**base, "seed_index": seeds[i], "init": tag, "status": status[r],
                        "first_cert_iter": int(run.first_cert[r]), "final_kkt": run.kkt[-1, r],
                        "mono_violations": int(run.mono_viol[r]),
                        "max_single_step_drop": float(run.max_drop[r]), "finite": bool(run.finite[r]),
                        "min_weight_over_records": float(run.min_w[:, r].min()),
                        "max_sum_dev_over_records": float(run.sum_dev[:, r].max()),
                        "L_ref": Lref[i], "L_ref_kkt": Lref_k[i], "L_ref_source": Lref_src[i],
                        "cap": cap, "runtime_s_per_trajectory": per_row_sec}
                       for r, (i, tag) in enumerate(rows_meta)])
    write_csv_atomic(table, outdir / "cells" / f"{key}_table.csv.gz")
    write_csv_atomic(st, outdir / "cells" / f"{key}_status.csv.gz")
    keep = np.array([not tag.startswith("cvfold") for _, tag in rows_meta])
    # dense scalars, committed: dates (int16), gap and kkt (float32)
    np.savez_compressed(outdir / "cells" / f"{key}_dense.npz", iters=run.iters,
                        dates=run.dates["e"][:, keep].astype(np.int16),
                        gap=(Lref[[i for i, _ in np.array(rows_meta, dtype=object)[keep]]][None, :]
                             - run.ll[:, keep]).astype(np.float32),
                        kkt=run.kkt[:, keep].astype(np.float32),
                        seeds=np.array([seeds[i] for (i, _), k in zip(rows_meta, keep) if k]),
                        inits=np.array([t for (_, t), k in zip(rows_meta, keep) if k]),
                        npmle_dates=npm_d.astype(np.int16), cv_iters=cv_k)
    # full-precision arrays, local only
    rawdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(rawdir / f"{key}.npz", iters=run.iters, ll=run.ll, kkt=run.kkt,
                        qhat=run.qhat, mom=run.mom, curves=run.curves["e"], dates=run.dates["e"],
                        rows=np.array([f"{i}|{t}" for i, t in rows_meta]),
                        npmle_weights=npm_W, npmle_curves=npm_C,
                        **{f"snap_{k}": v[keep] for k, v in run.snapshots.items()})
    done.write_text(json.dumps({"fingerprint": fp, "settings": settings, "seconds": run.seconds,
                                "tie_fallbacks": ev.n_fallback}, indent=1) + "\n")
    return key, f"{run.seconds:.0f}s ({per_row_sec * 1e6 / max(cap, 1):.2f} us/row-iter)", run.seconds


def parse_seeds(s):
    if s is None:
        return list(SEEDS)
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main(argv=None):
    from experiments.em_mechanism.core import INIT_NAMES
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kinds", default="ref,pair")
    ap.add_argument("--ns", default=None)
    ap.add_argument("--seeds", default=None, help="e.g. 1001-1050 or 1001,1002")
    ap.add_argument("--configs", default=None, help="comma list of config_ids")
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--G", type=int, default=G_DEFAULT)
    ap.add_argument("--inits", default=None, help="comma list (default: all eight)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None, help="subdirectory name under results/em_mechanism")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--reuse-by-settings", action="store_true",
                    help="resume: accept a completed cell whose settings match even if the code "
                         "fingerprint differs (use only when generator code is unchanged in substance)")
    a = ap.parse_args(argv)
    kinds = a.kinds.split(",")
    ns = [int(x) for x in a.ns.split(",")] if a.ns else list(NS)
    seeds = parse_seeds(a.seeds)
    cells = gamma1_cells()
    cids = [c.cid for c in cells]
    inits = a.inits.split(",") if a.inits else list(INIT_NAMES)
    sub = a.out or ("finite" if a.G == G_DEFAULT else f"finite_G{a.G}")
    if a.pilot:
        cids = [cells[0].cid, cells[20].cid]
        seeds, a.cap, sub = seeds[:5], min(a.cap, 100_000), a.out or "pilot_finite"
    if a.configs:
        cids = a.configs.split(",")
    outdir, rawdir = OUT / sub, RUNS / sub
    (outdir / "cells").mkdir(parents=True, exist_ok=True)
    tasks = [(k, cid, N, a.G, tuple(seeds), a.cap, str(outdir), str(rawdir), tuple(inits),
              a.reuse_by_settings)
             for k in kinds for N in ns for cid in cids]
    manifest = {"stage": "finite", "tau": TAU, "G": a.G, "kinds": kinds, "ns": ns, "seeds": seeds,
                "configs": cids, "cap": a.cap, "inits": inits, "n_cells": len(tasks)}
    (outdir / "manifest_request.json").write_text(json.dumps(manifest, indent=1) + "\n")
    t0 = time.perf_counter()
    print(f"{len(tasks)} cells x {len(seeds)} seeds x {len(inits)}+2 trajectories, cap {a.cap}", flush=True)
    with mp.Pool(min(a.workers, len(tasks))) as pool:
        for i, (key, msg, _) in enumerate(pool.imap_unordered(run_cell, tasks), 1):
            print(f"[{i}/{len(tasks)}] [{(time.perf_counter() - t0) / 60:.1f} min] {key}: {msg}", flush=True)


if __name__ == "__main__":
    main()
