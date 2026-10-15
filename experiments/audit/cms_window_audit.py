"""Does the CMS bootstrap use period-tau exits?

Seed-level and simulation evidence only. Different outputs for the same seed
(part B) are NOT evidence of information use; the distributional argument and
the factorization audit are in experiments/audit/cms_equivalence_audit.py and
paper/reconciliation/README.md section 06.

`cms_table2_run.py` passes `empirical_path_counts(data, tau)` to
`wtm.cms.bootstrap_rule`, which resamples those counts multinomially. The path
cells distinguish "exited at the end of period tau" from "still active", for
the same warning history. Every moment and covariance is computed through
`moments_with_covariance`, whose map gives identical columns to the two cells
of each such pair (checked below). Each draw's computation therefore depends
on the draw only through the pre-failure (merged) counts. The DRAW itself is
another matter: numpy's multinomial samples categories sequentially, so for a
fixed seed the merged draw can depend on how the observed counts split between
the two cells.

Tests, with every pre-failure observation held fixed:
  A. column identity of the moment map for each (exit at tau, active) pair;
  B. realized draws: for every Table 3 run (tau=2, N in {1e3, 1e5}) and each
     of its 20 archived draw seeds, compare the merged bootstrap draw under
     (i) the observed split, (ii) all period-tau exits merged into "active",
     (iii) the split reversed within each warning history;
  C. bootstrap law: for a few runs, many fresh seeds, compare the merged-draw
     distributions under (i) and (ii); multinomial aggregation predicts
     equality;
  D. selected dates: for a sample of runs, rerun the full CMS rule at
     publication settings (401-point grid, 20 draws, the archived seeds) on
     the merged counts, i.e. the pre-failure implementation, and compare with
     the archived Table 3 decisions.

    python experiments/audit/cms_window_audit.py [--sample 16] [--workers 4] [--out DIR]
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
TAU, NS, N_BOOT = 2, (1000, 100000), 20


def alphabet(tau):
    from wtm.moment_cone import build_path_alphabet
    pa = build_path_alphabet(tau)
    idx = {p: i for i, p in enumerate(pa.paths)}
    pairs = [(idx[(tau, obs)], idx[(None, obs)]) for (e, obs) in pa.paths if e == tau]
    return pa, pairs


def merged(counts, pairs):
    c = np.array(counts, float)
    for a, b in pairs:
        c[b] += c[a]
        c[a] = 0
    return c


def reversed_split(counts, pairs):
    c = np.array(counts, float)
    for a, b in pairs:
        c[a], c[b] = c[b], c[a]
    return c


def collapse(draw, pairs):
    """The pre-failure (merged) view of a fine draw: what every computation sees."""
    return merged(draw, pairs)


def runs():
    from wtm.config import config_id, load_frozen, population_cells
    cfg = load_frozen()
    out = []
    for c in population_cells(cfg):
        if c.gamma != 1:
            continue
        for N in NS:
            cid = config_id(dist=c.dist_name, gamma=1, T=c.T, beta=c.beta, N=N, t=TAU)
            for s in cfg["seeds"]["list"]:
                out.append((c, N, cid, int(s)))
    return out


def counts_for(c, N, cid, s):
    from wtm.moment_projection import empirical_path_counts
    from wtm.simulate import simulate_prefix
    data = simulate_prefix(c.dist, N, c.T, 1, TAU, seed=(int(cid, 16) + s) % 2 ** 32)
    return np.asarray(empirical_path_counts(data, TAU), float)


def test_a():
    from wtm.moment_cone import cohort_design
    res = {}
    for tau in (2, 3, 4):
        pa, pairs = alphabet(tau)
        B = np.linalg.pinv(cohort_design(tau, 1)) @ pa.to_cohort.T
        res[tau] = all(np.array_equal(B[:, a], B[:, b]) for a, b in pairs)
    return res


def test_b():
    _, pairs = alphabet(TAU)
    n_draws = changed_merge = changed_rev = 0
    runs_any = runs_with_exits = 0
    for c, N, cid, s in runs():
        cnt = counts_for(c, N, cid, s)
        if sum(cnt[a] for a, _ in pairs) > 0:
            runs_with_exits += 1
        any_change = False
        for d in range(N_BOOT):
            seed = (int(cid, 16) + s + d) % 2 ** 32
            ref = collapse(np.random.default_rng(seed).multinomial(int(cnt.sum()), cnt / cnt.sum()), pairs)
            m = merged(cnt, pairs)
            alt1 = collapse(np.random.default_rng(seed).multinomial(int(m.sum()), m / m.sum()), pairs)
            r = reversed_split(cnt, pairs)
            alt2 = collapse(np.random.default_rng(seed).multinomial(int(r.sum()), r / r.sum()), pairs)
            n_draws += 1
            changed_merge += int(not np.array_equal(ref, alt1))
            changed_rev += int(not np.array_equal(ref, alt2))
            any_change |= not np.array_equal(ref, alt1)
        runs_any += any_change
    return {"runs": len(runs()), "runs_with_period_tau_exits": runs_with_exits,
            "draws": n_draws, "draws_changed_by_merging": changed_merge,
            "draws_changed_by_reversing_split": changed_rev,
            "runs_with_any_changed_draw": runs_any}


def test_c(n_seeds=20000):
    """Same law: merged-draw mean and covariance under observed vs merged split."""
    from scipy import stats
    _, pairs = alphabet(TAU)
    out = []
    for (c, N, cid, s) in runs()[::150][:4]:
        cnt = counts_for(c, N, cid, s)
        m = merged(cnt, pairs)
        rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(2)
        A = np.array([collapse(rng_a.multinomial(int(cnt.sum()), cnt / cnt.sum()), pairs)
                      for _ in range(n_seeds)])
        Bm = np.array([rng_b.multinomial(int(m.sum()), m / m.sum()) for _ in range(n_seeds)])
        keep = [j for j in range(A.shape[1]) if A[:, j].var() > 0]
        ks = [stats.ks_2samp(A[:, j], Bm[:, j]).pvalue for j in keep]
        out.append({"cell_id": cid, "N": N, "seed": s, "cells_tested": len(keep),
                    "min_ks_p": float(min(ks)),
                    "max_mean_diff_in_sd": float(max(abs(A[:, j].mean() - Bm[:, j].mean())
                                                     / A[:, j].std() for j in keep))})
    return out


def run_d(args):
    c, N, cid, s = args
    from wtm import cms
    _, pairs = alphabet(TAU)
    m = merged(counts_for(c, N, cid, s), pairs)
    risks = []
    for d in range(N_BOOT):
        r = cms.bootstrap_rule(m, TAU, 1, c.T, c.beta, n_boot=1, grid_size=401,
                               seed=(int(cid, 16) + s + d) % 2 ** 32)
        if r.per_draw:
            risks.append(r.per_draw[0])
    dates = sorted(risks[0], key=int)
    avg = {d: float(np.mean([x[d] for x in risks])) for d in dates}
    sel = int(min(avg, key=lambda d: (avg[d], int(d))))
    return {"cell_id": cid, "N": N, "seed_index": s, "date_prefailure": sel, "draws_used": len(risks)}


def test_d(sample, workers, partial: Path):
    from wtm.rules import evaluate
    from wtm.welfare import PopulationConfig
    arch = pd.read_csv(RESULTS / "cms_table2_raw.csv").set_index(["cell_id", "N", "seed_index"])
    rng = np.random.default_rng(20261002)
    all_runs = runs()
    pick = [all_runs[i] for i in sorted(rng.choice(len(all_runs), sample, replace=False))]
    by = {r[2]: r[0] for r in pick}
    rows = []
    with mp.Pool(workers) as pool:
        for r in pool.imap_unordered(run_d, pick, chunksize=1):
            rows.append(r)
            pd.DataFrame(rows).to_csv(partial, index=False)     # survives interruption
            print(f"D: {len(rows)}/{len(pick)} runs done", flush=True)
    for r in rows:
        a = arch.loc[(r["cell_id"], r["N"], r["seed_index"])]
        c = by[r["cell_id"]]
        pop = PopulationConfig(c.dist, 1, c.T, c.beta)
        r.update(date_archived=int(a.selected_date), changed=int(a.selected_date) != r["date_prefailure"],
                 regret_archived=float(a.relative_regret),
                 regret_prefailure=float(evaluate(r["date_prefailure"], pop)["relative_regret"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(RESULTS / "audit"))
    ap.add_argument("--skip-d", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rep = {"A_moment_map_columns_identical": test_a()}
    print("A", rep["A_moment_map_columns_identical"], flush=True)
    rep["B_realized_draws"] = test_b()
    print("B", rep["B_realized_draws"], flush=True)
    rep["C_bootstrap_law"] = test_c()
    print("C", rep["C_bootstrap_law"], flush=True)
    (out / "cms_window_audit.json").write_text(json.dumps(rep, indent=2) + "\n")
    if not a.skip_d:
        d = test_d(a.sample, a.workers, out / "cms_window_audit_dates.partial.csv")
        pd.DataFrame(d).to_csv(out / "cms_window_audit_dates.csv", index=False)
        (out / "cms_window_audit_dates.partial.csv").unlink(missing_ok=True)
        rep["D_selected_dates"] = {"runs": len(d), "changed": sum(r["changed"] for r in d),
                                   "sample": "random runs of the 1,200 (rng 20261002)",
                                   "mean_regret_archived": float(np.mean([r["regret_archived"] for r in d])),
                                   "mean_regret_prefailure": float(np.mean([r["regret_prefailure"] for r in d]))}
        print("D", rep["D_selected_dates"], flush=True)
    rep["runtime_s"] = time.perf_counter() - t0
    (out / "cms_window_audit.json").write_text(json.dumps(rep, indent=2) + "\n")


if __name__ == "__main__":
    main()
