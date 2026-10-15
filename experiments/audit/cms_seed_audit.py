"""Seed reuse in the archived Table 3 CMS run: what overlaps, and does it matter?

SEED ASSIGNMENT (cms_table2_run.py, as archived). For a cell c = (distribution,
T, beta, N, tau = 2), base_c = int(config_id(...), 16), a 48-bit hash. Run s
(seed index s in 1001..1020):

    data       simulate_prefix(seed = (base_c + s) mod 2^32)  -> default_rng(that)
    draw d     bootstrap_rule(seed = (base_c + s + d) mod 2^32, n_boot = 1)
               -> its own default_rng(that), used for one multinomial draw

So the integer seeds overlap in three ways, all within one cell:

  W. within-run: draw 0 of run s uses the data seed of run s itself;
  X. cross-run, data: draw d >= 1 of run s uses the data seed of run s + d
     (when s + d <= 1020);
  Y. cross-run, bootstrap: draw d of run s and draw d' of run s' use the same
     seed whenever s + d = s' + d'.

numpy's default_rng(k) seeds PCG64 through SeedSequence(k), so equal integers
give identical bit streams. (The multinomial draw and the data simulation
consume the same stream differently, so "identical stream" does not mean
"identical numbers used for the same purpose".)

TESTS
  A. seed map: counts of W, X, Y, and collisions with every other seed used by
     the Table 3 procedures (bagging/RC rule seeds = data seed + 2^31 - 1) and
     with the data seeds of all other cells.
  B. within-run coupling (W): is the draw-0 resampling deviation correlated
     with the data's own sampling error, compared with draws 1..19? The true
     cell probabilities come from one 10^7-person simulation per population,
     seeded independently of everything above.
  C. decision-level exchangeability of draw 0, from the archived per-draw
     risks: leave-one-draw-out decision changes and the rank of each draw's
     risk at the selected date, draw 0 against draws 1..19.
  D. cross-run coupling (Y): correlation of the resampling deviations of two
     draws that share a seed (different runs, same cell), against pairs that
     do not.

    python experiments/audit/cms_seed_audit.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
RESULTS = ROOT / "results"
TRUTH_N = 10_000_000
TRUTH_SEED = 777_000_001          # not of the form base + s for any cell (checked in A)

from cms_window_audit import N_BOOT, TAU, alphabet, counts_for, merged, runs  # noqa: E402


def seed_map():
    from wtm.config import config_id, load_frozen, population_cells
    cfg = load_frozen()
    seeds = [int(x) for x in cfg["seeds"]["list"]]
    rs = runs()
    W = len(rs)                                               # draw 0
    X = sum(1 for (_, _, _, s) in rs for d in range(1, N_BOOT) if s + d in seeds)
    per_cell = defaultdict(list)
    for (c, N, cid, s) in rs:
        for d in range(N_BOOT):
            per_cell[cid].append((int(cid, 16) + s + d) % 2 ** 32)
    distinct = {cid: len(set(v)) for cid, v in per_cell.items()}
    Y_draws_sharing = sum(len(v) - len(set(v)) for v in per_cell.values())
    # every other seed used anywhere in the frozen design (all cells, N, tau)
    data_all, rule_all = {}, {}
    for c in population_cells(cfg):
        for N in cfg["population_size"]["N"]:
            for t in cfg["prefix_length"]["t"]:
                cid = config_id(dist=c.dist_name, gamma=c.gamma, T=c.T, beta=c.beta, N=N, t=t)
                for s in seeds:
                    k = (int(cid, 16) + s) % 2 ** 32
                    data_all[k] = cid
                    rule_all[(k + 2 ** 31 - 1) % 2 ** 32] = cid
    boot = {}
    for cid, v in per_cell.items():
        for k in v:
            boot.setdefault(k, cid)
    cross_cell_data = sum(1 for k, cid in boot.items() if k in data_all and data_all[k] != cid)
    with_rule = sum(1 for k in boot if k in rule_all)
    return {
        "runs": len(rs), "draws": len(rs) * N_BOOT,
        "W_draw0_uses_own_data_seed": W,
        "X_draw_uses_data_seed_of_another_run_same_cell": X,
        "Y_draws_whose_seed_is_shared_with_an_earlier_draw_same_cell": Y_draws_sharing,
        "distinct_bootstrap_seeds_per_cell": sorted(set(distinct.values())),
        "bootstrap_draws_per_cell": N_BOOT * len(seeds),
        "bootstrap_seed_equals_data_seed_of_a_different_cell": cross_cell_data,
        "bootstrap_seed_equals_a_bagging_or_rc_rule_seed": with_rule,
        "truth_seed_collides": TRUTH_SEED in data_all or TRUTH_SEED in boot or TRUTH_SEED in rule_all,
        "design_cells_checked": len(set(data_all.values())),
    }


def truth_probs(cells):
    """True merged path-cell probabilities per population, by one large simulation."""
    from wtm.moment_projection import empirical_path_counts
    from wtm.simulate import simulate_prefix
    _, pairs = alphabet(TAU)
    out = {}
    for i, (key, c) in enumerate(sorted(cells.items())):
        d = simulate_prefix(c.dist, TRUTH_N, c.T, 1, TAU, seed=TRUTH_SEED + i, keep_truth=False)
        x = merged(np.asarray(empirical_path_counts(d, TAU), float), pairs)
        out[key] = x / x.sum()
    return out


def archived_fine_draw(cnt, cid, s, d):
    seed = (int(cid, 16) + s + d) % 2 ** 32
    return np.random.default_rng(seed).multinomial(int(round(cnt.sum())), cnt / cnt.sum()).astype(float)


def test_b_d(truth):
    """B (within-run) and D (cross-run) from the archived draws."""
    _, pairs = alphabet(TAU)
    zu = defaultdict(list)          # d -> standardized bootstrap deviations
    ze = defaultdict(list)          # d -> matching standardized data errors
    dev_by_seed = defaultdict(list)  # (cid, seed) -> [(s, d, standardized deviation vector)]
    for (c, N, cid, s) in runs():
        cnt = counts_for(c, N, cid, s)
        x = merged(cnt, pairs)
        n = x.sum()
        p = truth[(c.dist_name, c.T, c.beta)]
        keep = (x > 0) & (p > 0) & (p < 1)
        e = (x[keep] / n - p[keep]) / np.sqrt(p[keep] * (1 - p[keep]) / n)
        for d in range(N_BOOT):
            X = merged(archived_fine_draw(cnt, cid, s, d), pairs)
            ph = x[keep] / n
            u = (X[keep] / n - ph) / np.sqrt(ph * (1 - ph) / n)
            zu[d].append(u)
            ze[d].append(e)
            dev_by_seed[(cid, (int(cid, 16) + s + d) % 2 ** 32)].append((s, d, u))
    corr = {}
    for d in range(N_BOOT):
        u, e = np.concatenate(zu[d]), np.concatenate(ze[d])
        r, pval = stats.pearsonr(u, e)
        corr[d] = {"r": float(r), "p": float(pval), "n": int(len(u))}
    ref = np.array([corr[d]["r"] for d in range(1, N_BOOT)])
    se = 1 / np.sqrt(corr[0]["n"])
    b = {"draw0": corr[0],
         "draws1_19_r_range": [float(ref.min()), float(ref.max())],
         "draws1_19_r_sd": float(ref.std(ddof=1)),
         "null_sd_of_r": float(se),
         "draw0_z_vs_null": float(corr[0]["r"] / se),
         "per_draw": corr}

    shared, unshared = [], []
    for (cid, seed), lst in dev_by_seed.items():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if lst[i][0] != lst[j][0] and len(lst[i][2]) == len(lst[j][2]):
                    shared.append(np.corrcoef(lst[i][2], lst[j][2])[0, 1])
    # non-sharing reference: same cell, different runs, different seeds
    by_cell = defaultdict(list)
    for (cid, seed), lst in dev_by_seed.items():
        by_cell[cid].extend((seed, *t) for t in lst)
    rng = np.random.default_rng(20261004)
    for cid, lst in by_cell.items():
        for _ in range(200):
            a, b2 = rng.choice(len(lst), 2, replace=False)
            A, B = lst[a], lst[b2]
            if A[0] != B[0] and A[1] != B[1] and len(A[3]) == len(B[3]):
                unshared.append(np.corrcoef(A[3], B[3])[0, 1])
    d = {"pairs_sharing_a_seed": len(shared),
         "mean_corr_sharing": float(np.nanmean(shared)),
         "median_corr_sharing": float(np.nanmedian(shared)),
         "pairs_reference": len(unshared),
         "mean_corr_reference": float(np.nanmean(unshared))}
    return b, d


def test_c():
    """Decision-level exchangeability of draw 0 (archived per-draw risks)."""
    from experiments.scripts import cms_table2_analyze as an
    by = defaultdict(dict)
    for r in an.draw_records():
        if r["success"]:
            by[(r["cell_id"], r["N"], r["seed_index"])][r["draw"]] = \
                {int(k): v for k, v in json.loads(r["max_risk"]).items()}
    def decide(risks):
        dates = sorted(risks[0])
        avg = {t: np.mean([x[t] for x in risks]) for t in dates}
        return min(avg, key=lambda t: (avg[t], t))
    change = np.zeros(N_BOOT)
    eligible = 0
    rank0, rank_rest = [], []
    for key, dr in by.items():
        if len(dr) != N_BOOT:
            continue
        eligible += 1
        allr = [dr[d] for d in range(N_BOOT)]
        sel = decide(allr)
        for d in range(N_BOOT):
            change[d] += decide([allr[k] for k in range(N_BOOT) if k != d]) != sel
        # rank of each draw's risk at the selected date, ties averaged
        r = stats.rankdata([allr[d][sel] for d in range(N_BOOT)], method="average")
        rank0.append(r[0])
        rank_rest.extend(r[1:])
    ref = change[1:]
    # draw 0 against draws 1..19 as two samples; permutation p-value of the
    # mean-rank difference keeps the tie structure (KS assumes continuity)
    rank0, rank_rest = np.array(rank0), np.array(rank_rest)
    obs = rank0.mean() - rank_rest.mean()
    pool = np.concatenate([rank0, rank_rest])
    g = np.random.default_rng(20261005)
    perm = []
    for _ in range(2000):
        g.shuffle(pool)
        perm.append(pool[:len(rank0)].mean() - pool[len(rank0):].mean())
    p_perm = float((np.abs(perm) >= abs(obs)).mean())
    return {"runs_with_all_20_draws": eligible,
            "leave_out_change_draw0": int(change[0]),
            "leave_out_change_draws1_19_range": [int(ref.min()), int(ref.max())],
            "leave_out_change_draws1_19_mean": float(ref.mean()),
            "draw0_mean_rank_at_selected_date": float(rank0.mean()),
            "draws1_19_mean_rank_at_selected_date": float(rank_rest.mean()),
            "mean_rank_difference_permutation_p": p_perm}


def test_e(n_rep=2000):
    """Table 3 CMS comparisons with a configuration-cluster bootstrap.

    All seed sharing is within one (configuration, N) cell, so resampling whole
    configurations (with every run of each) is valid under it."""
    import pandas as pd
    KEY = ["cell_id", "N", "prefix_t", "seed_index"]
    cms_ = pd.read_csv(RESULTS / "cms_table2_raw.csv")
    main = pd.read_csv(RESULTS / "main_benchmark_raw_all.csv", low_memory=False)
    bag = pd.read_csv(RESULTS / "bagged_full_b200_raw.csv", low_memory=False)
    em = pd.read_csv(RESULTS / "early_stopped_npmle_raw.csv", low_memory=False)
    def rule(df, r):
        d = df[(df.rule == r) & (df.prefix_t == TAU) & (df.N.isin([1000, 100000]))]
        if "success" in d:
            d = d[d.success.astype(str) == "True"]
        return d
    frames = {"CMS": cms_, "NPMLE": rule(main, "npmle"), "Bagged": rule(bag, "bagged"),
              "RC": rule(main, "regret_calibrated"), "EM55": rule(em, "early_stopped_npmle"),
              "Oracle": rule(main, "oracle"), "Act": rule(main, "act_immediately")}
    common = None
    for d in frames.values():
        k = set(map(tuple, d[KEY].values))
        common = k if common is None else common & k
    out = {"matched_runs": len(common)}
    g = np.random.default_rng(20261006)
    for N in (1000, 100000):
        keys = pd.DataFrame([k for k in common if k[1] == N], columns=KEY)
        sq = {name: d.merge(keys, on=KEY).set_index(KEY)["squared_regret"] for name, d in frames.items()}
        idx = sq["CMS"].index
        cfg = np.array([k[0] for k in idx])
        uniq = np.unique(cfg)
        rows = {}
        for other in ("NPMLE", "Bagged", "RC"):
            diff = (sq[other].loc[idx] - sq["CMS"].loc[idx]).to_numpy()   # >0 favours CMS
            per = {c: diff[cfg == c] for c in uniq}
            est = diff.mean()
            boots = []
            for _ in range(n_rep):
                pick = g.choice(uniq, len(uniq), replace=True)
                boots.append(np.concatenate([per[c] for c in pick]).mean())
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows[f"{other}_minus_CMS_MSRx1e3"] = {"estimate": 1e3 * est, "ci95": [1e3 * lo, 1e3 * hi],
                                                  "excludes_zero": bool(lo > 0 or hi < 0)}
        out[f"N={N}"] = {"runs": len(idx), "clusters": len(uniq), **rows}
    return out


def main():
    from wtm.config import load_frozen, population_cells
    t0 = time.perf_counter()
    rep = {"A_seed_map": seed_map()}
    print("A", rep["A_seed_map"], flush=True)
    cells = {(c.dist_name, c.T, c.beta): c for c in population_cells(load_frozen()) if c.gamma == 1}
    truth = truth_probs(cells)
    b, d = test_b_d(truth)
    rep["B_within_run_draw0"] = b
    print("B", {k: v for k, v in b.items() if k != "per_draw"}, flush=True)
    rep["C_decision_exchangeability"] = test_c()
    print("C", rep["C_decision_exchangeability"], flush=True)
    rep["D_cross_run_shared_streams"] = d
    print("D", d, flush=True)
    rep["E_table3_cms_cluster_inference"] = test_e()
    print("E", rep["E_table3_cms_cluster_inference"], flush=True)
    rep["truth"] = {"N": TRUTH_N, "seed": f"{TRUTH_SEED} + population index"}
    rep["runtime_s"] = time.perf_counter() - t0
    (RESULTS / "audit" / "cms_seed_audit.json").write_text(json.dumps(rep, indent=2) + "\n")


if __name__ == "__main__":
    main()
