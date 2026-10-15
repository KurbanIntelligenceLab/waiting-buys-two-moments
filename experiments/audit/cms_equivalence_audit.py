"""Does the CMS rule factor through the merged (pre-failure) counts?

The archived Table 3 CMS row resampled the path cells, which separate "exited
at the end of period tau" from "still active" for each warning history
(`resampling="paths"`). The argument in paper/reconciliation/README.md
section 06 is:

  (i)  multinomial aggregation: merging a Multinomial(n, p) draw over a
       partition gives Multinomial(n, merged p), so "resample, then merge" and
       "merge, then resample" have the same law;
  (ii) factorization: the rule sees a draw only through
       `moments_with_covariance(draw)`, and the moment map gives the two cells
       of each pair the same column, so (mhat, Cov) and everything computed
       from them are functions of the merged draw.

(i) is a theorem. This script checks (ii) on the code as it ran, and records
what (ii) does NOT give in floating point:

  A. static: the moment-map columns of every pair are bitwise identical;
  B. every one of the 24,000 archived fine draws, against its merge:
     (mhat, Cov) differences, and every discrete branch of the per-draw
     pipeline (feasibility, projection trigger, projected moments,
     post-projection feasibility, singleton test);
  C. a sample of draws through the full per-draw pipeline (`wtm.cms.draw_risks`):
     the fine draw (must reproduce the archive) and its merge (the
     factorized computation), compared with the archived risks;
  D. decision margins of all 1,200 archived runs, against the per-draw
     perturbation measured in C. The perturbation is measured on a sample
     only, so this is empirical supporting evidence, not a uniform error
     bound: it does not certify every archived decision against rounding;
  E. seed reuse between the data and the bootstrap (analysed in
     cms_seed_audit.py).

(i) and (ii) give the equivalence for the idealized procedure, whose bootstrap
randomness is independent of the data. The archived seeds do not satisfy
that exactly (cms_seed_audit.py).

    python experiments/audit/cms_equivalence_audit.py [--sample 24] [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
RESULTS = ROOT / "results"
GRID = 401

from cms_window_audit import N_BOOT, TAU, alphabet, counts_for, merged, runs  # noqa: E402


def fine_draw(cnt, cid, s, d):
    """The archived draw: `bootstrap_rule(..., n_boot=1, resampling="paths")`."""
    seed = (int(cid, 16) + s + d) % 2 ** 32
    return np.random.default_rng(seed).multinomial(
        int(round(cnt.sum())), cnt / cnt.sum()).astype(float)


def archive():
    """Archived per-draw records: the checkpoint files, or the packaged copy."""
    from experiments.scripts import cms_table2_analyze as an
    out = {}
    for r in an.draw_records():
        out[(r["cell_id"], r["N"], r["seed_index"], r["draw"])] = r
    return out


def test_a():
    from wtm.moment_cone import cohort_design
    res = {}
    for tau in (2, 3, 4):
        pa, pairs = alphabet(tau)
        B = np.linalg.pinv(cohort_design(tau, 1)) @ pa.to_cohort.T
        res[tau] = {"pairs": len(pairs),
                    "identical": all(np.array_equal(B[:, a], B[:, b]) for a, b in pairs)}
    return res


def branches(draw, grid):
    """Every discrete decision `draw_risks` takes, and the moments it then uses."""
    from wtm.cms import on_cone_boundary
    from wtm.lp import IdentifiedSet
    from wtm.moment_projection import moments_with_covariance, project_to_cone
    mhat, cov = moments_with_covariance(draw, TAU, 1)
    feas0 = IdentifiedSet.from_moments(mhat, TAU, grid).feasible_point()[0] is not None
    used, feas1 = mhat, feas0
    if not feas0:
        used = project_to_cone(mhat, cov, grid, TAU).moments
        feas1 = IdentifiedSet.from_moments(used, TAU, grid).feasible_point()[0] is not None
    point = bool(on_cone_boundary(used, TAU, tol=1e-9)[0]) if feas1 else None
    return mhat, cov, used, (feas0, feas1, point)


def b_run(args):
    from wtm.lp import support_grid
    c, N, cid, s = args
    grid = support_grid(GRID)
    _, pairs = alphabet(TAU)
    cnt = counts_for(c, N, cid, s)
    rows = []
    for d in range(N_BOOT):
        f = fine_draw(cnt, cid, s, d)
        g = merged(f, pairs)
        m1, c1, u1, b1 = branches(f, grid)
        m2, c2, u2, b2 = branches(g, grid)
        rows.append({"cell_id": cid, "N": N, "seed_index": s, "draw": d,
                     "mhat_bitwise": bool(np.array_equal(m1, m2)),
                     "cov_bitwise": bool(np.array_equal(c1, c2)),
                     "mhat_maxdiff": float(np.abs(m1 - m2).max()),
                     "cov_maxdiff": float(np.abs(c1 - c2).max()),
                     "branches_equal": b1 == b2, "projected": not b1[0],
                     "used_maxdiff": float(np.abs(u1 - u2).max()),
                     "branches": list(b1)})
    return rows


def c_draw(args):
    from wtm.cms import draw_risks
    from wtm.lp import support_grid
    c, N, cid, s, d = args
    grid = support_grid(GRID)
    _, pairs = alphabet(TAU)
    cnt = counts_for(c, N, cid, s)
    f = fine_draw(cnt, cid, s, d)
    t0 = time.perf_counter()
    rf = draw_risks(f, TAU, 1, c.T, c.beta, grid)
    rm = draw_risks(merged(f, pairs), TAU, 1, c.T, c.beta, grid)
    return {"cell_id": cid, "N": N, "seed_index": s, "draw": d,
            "fine": rf.risks, "merged": rm.risks, "projected": rf.projected,
            "exact": rf.exact, "seconds": time.perf_counter() - t0}


def margins(arch):
    by = defaultdict(list)
    for (cid, N, s, d), r in arch.items():
        if r["success"]:
            by[(cid, N, s)].append(json.loads(r["max_risk"]))
    rows = []
    for k, rs in by.items():
        dates = sorted(rs[0], key=int)
        avg = sorted((float(np.mean([x[d] for x in rs])), int(d)) for d in dates)
        rows.append({"cell_id": k[0], "N": k[1], "seed_index": k[2],
                     "selected": avg[0][1], "margin": avg[1][0] - avg[0][0],
                     "scale": max(abs(avg[0][0]), 1e-300), "n_draws": len(rs)})
    return rows


def seed_reuse():
    """Bootstrap draw d of run s uses default_rng(base + s + d); the data of run s
    use simulate_prefix(seed=base + s), itself default_rng(base + s)."""
    from wtm.config import load_frozen
    seeds = [int(x) for x in load_frozen()["seeds"]["list"]]
    rs = runs()
    same_run = len(rs)                                   # draw 0 of every run
    other_run = sum(1 for (_, _, _, s) in rs for d in range(1, N_BOOT) if s + d in seeds)
    return {"draws": len(rs) * N_BOOT,
            "draw0_seed_equals_own_data_seed": same_run,
            "draw_seed_equals_data_seed_of_another_run_same_config": other_run,
            "applies_to": "both resampling modes (the seeds are the same)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=24)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(RESULTS / "audit"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rep = {"A_columns": test_a()}
    print("A", rep["A_columns"], flush=True)

    with mp.Pool(a.workers) as pool:
        brows = [r for rs in pool.imap_unordered(b_run, runs(), chunksize=4) for r in rs]
    rep["B_all_draws"] = {
        "draws": len(brows),
        "mhat_bitwise_equal": sum(r["mhat_bitwise"] for r in brows),
        "cov_bitwise_equal": sum(r["cov_bitwise"] for r in brows),
        "max_abs_mhat_diff": max(r["mhat_maxdiff"] for r in brows),
        "max_abs_cov_diff": max(r["cov_maxdiff"] for r in brows),
        "branches_equal": sum(r["branches_equal"] for r in brows),
        "projected": sum(r["projected"] for r in brows),
        "max_abs_used_moment_diff": max(r["used_maxdiff"] for r in brows),
        "branch_mismatches": [r for r in brows if not r["branches_equal"]][:20],
    }
    print("B", {k: v for k, v in rep["B_all_draws"].items() if k != "branch_mismatches"},
          flush=True)

    arch = archive()
    # C: stratified sample -- projected, exact (singleton) and plain draws, both N
    rng = np.random.default_rng(20261003)
    byrun = {(cid, N, s): (c, N, cid, s) for (c, N, cid, s) in runs()}
    pool_keys = {"projected": [], "exact": [], "plain_1e3": [], "plain_1e5": [], "failed": []}
    for k, r in arch.items():
        if not r["success"]:
            pool_keys["failed"].append(k)
        elif r.get("n_exact", 0):
            pool_keys["exact"].append(k)
        elif r["n_projected"]:
            pool_keys["projected"].append(k)
        else:
            pool_keys["plain_1e3" if k[1] == 1000 else "plain_1e5"].append(k)
    quota = {"projected": a.sample // 4, "exact": a.sample // 6,
             "plain_1e5": a.sample // 4, "failed": 2}
    quota["plain_1e3"] = a.sample - sum(v for k, v in quota.items() if k != "failed")
    picks, strata = [], {}
    for name, keys in pool_keys.items():
        keys = sorted(keys)
        for i in rng.choice(len(keys), min(quota[name], len(keys)), replace=False):
            cid, N, s, d = keys[i]
            c = byrun[(cid, N, s)][0]
            picks.append((c, N, cid, s, d))
            strata[name] = strata.get(name, 0) + 1
    crows = []
    with mp.Pool(a.workers) as pool:
        for r in pool.imap_unordered(c_draw, picks, chunksize=1):
            ar = arch[(r["cell_id"], r["N"], r["seed_index"], r["draw"])]
            archived = ({int(k): v for k, v in json.loads(ar["max_risk"]).items()}
                        if ar["success"] else None)
            def diff(x):
                if x is None or archived is None:
                    return None if (x is None) == (archived is None) else float("inf")
                return max(abs(x[d] - archived[d]) for d in archived)
            r.update(fine_vs_archive=diff(r["fine"]), merged_vs_archive=diff(r["merged"]),
                     merged_argmin_equal=(None if archived is None or r["merged"] is None else
                                          min(r["merged"], key=lambda d: (r["merged"][d], d)) ==
                                          min(archived, key=lambda d: (archived[d], d))))
            crows.append(r)
            print(f"C {len(crows)}/{len(picks)} fine={r['fine_vs_archive']} "
                  f"merged={r['merged_vs_archive']}", flush=True)
    fin = [r["merged_vs_archive"] for r in crows if r["merged_vs_archive"] is not None]
    rep["C_full_pipeline_sample"] = {
        "draws": len(crows),
        "strata": strata,
        "fine_reproduces_archive_exactly": sum(r["fine_vs_archive"] == 0.0 for r in crows),
        "merged_equals_archive_exactly": sum(r["merged_vs_archive"] == 0.0 for r in crows),
        "max_abs_risk_diff_merged_vs_archive": max(fin) if fin else None,
        "failed_draws_fail_in_both": sum(1 for r in crows if r["fine"] is None and r["merged"] is None),
        "per_draw_argmin_equal": sum(1 for r in crows if r["merged_argmin_equal"]),
        "rows": [{k: v for k, v in r.items() if k not in ("fine", "merged")} for r in crows],
    }
    print("C", {k: v for k, v in rep["C_full_pipeline_sample"].items() if k != "rows"},
          flush=True)

    mrows = margins(arch)
    m = np.array([r["margin"] for r in mrows])
    rel = np.array([r["margin"] / r["scale"] for r in mrows])
    delta = rep["C_full_pipeline_sample"]["max_abs_risk_diff_merged_vs_archive"] or 0.0
    rep["D_decision_margins"] = {
        "runs": len(mrows),
        "exact_ties": int((m == 0).sum()),
        "min_margin": float(m.min()), "min_relative_margin": float(rel.min()),
        "margin_quantiles": {q: float(np.quantile(m, q)) for q in (0.001, 0.01, 0.1, 0.5)},
        "flip_threshold_2delta": 2 * delta,
        "runs_with_margin_below_2delta": int((m <= 2 * delta).sum()),
        "runs_with_margin_below_1e-9": int((m < 1e-9).sum()),
        "smallest": sorted(mrows, key=lambda r: r["margin"])[:5],
    }
    print("D", {k: v for k, v in rep["D_decision_margins"].items() if k != "smallest"}, flush=True)
    rep["E_seed_reuse"] = seed_reuse()
    print("E", rep["E_seed_reuse"], flush=True)
    rep["runtime_s"] = time.perf_counter() - t0
    (out / "cms_equivalence_audit.json").write_text(json.dumps(rep, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
