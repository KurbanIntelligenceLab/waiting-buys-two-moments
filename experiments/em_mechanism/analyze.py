"""Analysis of the EM-mechanism study: tables and inference (figures: figures.py).

Reads results/em_mechanism/{population,finite,finite_G*}/ and writes
results/em_mechanism/analysis/.  Never reads true welfare to choose anything:
truths are used only to score already-fixed procedures.

Conventions
  regret        relative regret R(s) = 1 - W^s / W^{t*}; efficiency = 1 - R
  MSR           mean of R^2 (reported x 1e3)
  p10           tenth percentile of efficiency
  NPMLE invalid scored as immediate allocation (date 1), the main benchmark's
                convention; counts reported.  (D.3's joint-validity filter is
                reported as a sensitivity.)
  pair risk     rho = (1/30) sum_j max_i mean_s R_{j,i}(d_s)   [max after trial averaging]
                tw  = (1/30) sum_j mean_s max_i R_{j,i}(d_s)   [mean trialwise maximum]

Usage: python -m experiments.em_mechanism.analyze [--finite finite] [--population population]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.em_mechanism.common import OUT, gamma1_cells, write_csv_atomic

AN = OUT / "analysis"
B_BOOT = 2000
BOOT_SEED = 20261005
KEY_ITERS = (0, 30, 55, 80, 100, 1000, 10_000, 100_000, 1_000_000, 10_000_000)
# Figure-1 representatives, fixed by rule (not by results): every family at the
# middle budget beta = 0.002 and the longer horizon T = 20.
REP_T, REP_BETA = 20, 0.002
# Figure-3 pair, fixed by rule: the manuscript's headline Theorem-1 pair.
PAIR_DIST, PAIR_T, PAIR_BETA = "beta_1_32", 20, 0.01


def p10(x):
    return float(np.quantile(1.0 - np.asarray(x, dtype=float), 0.10))


# ------------------------------------------------------------------ population
def load_population(sub="population"):
    d = OUT / sub
    if not d.exists():
        return None
    out = {}
    for kind in ("fits", "decisions", "curves", "truth"):
        fs = sorted(glob.glob(str(d / f"*_{kind}.csv.gz")))
        out[kind] = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None
    fs = sorted(glob.glob(str(d / "*_status.csv")))
    out["status"] = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None
    return out


def population_tables(P):
    dec = P["decisions"]
    caps = P["status"].groupby(["tau", "G", "variant"]).cap.max().rename("cap").reset_index()
    dec = dec.merge(caps, on=["tau", "G", "variant"])
    rcols = [c for c in dec.columns if c.startswith("regret_")]
    em = dec[dec.iteration >= 0]
    # H1: regret at key iterations vs the capped endpoint vs NPMLE
    rows = []
    for (tau, G, var, init), g in em.groupby(["tau", "G", "variant", "init"]):
        cap = int(g.cap.iloc[0])
        r = {"tau": tau, "G": G, "variant": var, "init": init, "cap": cap, "n_configs": g[["distribution", "T", "beta"]].drop_duplicates().shape[0]}
        piv = g.pivot_table(index=["distribution", "T", "beta"], columns="iteration", values="regret_ref")
        for k in KEY_ITERS:
            if k <= cap and k in piv.columns:
                r[f"mean_regret_ref_k{k}"] = float(piv[k].mean())
        r["mean_regret_ref_cap"] = float(piv[cap].mean())
        r["mean_regret_ref_min_over_path"] = float(piv.min(axis=1).mean())
        a, b = piv[55], piv[cap]
        r["configs_cap_worse_than_55"] = int((b > a + 1e-12).sum())
        r["configs_cap_better_than_55"] = int((b < a - 1e-12).sum())
        r["configs_cap_equal_55"] = int(((b - a).abs() <= 1e-12).sum())
        later = piv.loc[:, [c for c in piv.columns if c > 55]]
        r["configs_any_later_iter_worse_than_55"] = int((later.max(axis=1) > a + 1e-12).sum())
        for rc in [c for c in rcols if g[c].notna().any()]:
            pv = g.pivot_table(index=["distribution", "T", "beta"], columns="iteration", values=rc)
            r[f"mean_{rc}_k55"] = float(pv[55].mean())
            r[f"mean_{rc}_cap"] = float(pv[cap].mean())
        rows.append(r)
    h1 = pd.DataFrame(rows)
    npm = dec[dec.iteration < 0].groupby(["tau", "G", "variant", "init"])[rcols].mean().reset_index()
    write_csv_atomic(h1, AN / "population_H1_by_init.csv")
    write_csv_atomic(npm, AN / "population_npmle.csv")

    # per-configuration detail at key iterations
    keep = em[em.iteration.isin(list(KEY_ITERS)) | (em.iteration == em.cap)]
    det = pd.concat([keep, dec[dec.iteration < 0]])
    write_csv_atomic(det.drop(columns=["cap"]), AN / "population_by_config_key_iters.csv.gz")

    # grid mismatch: residual KL of the grid optimum to the exact target
    f = P["fits"]
    mm = f[f.init.isin(["npmle", "npmle_sharp"])][["tau", "G", "variant", "distribution", "init", "ll",
                                                  "kkt", "status", "n_support", "kl_to_target",
                                                  "max_abs_cell_diff"]]
    write_csv_atomic(mm, AN / "population_grid_mismatch.csv")
    st = P["status"]
    stc = st.groupby(["tau", "G", "variant", "status"]).size().unstack(fill_value=0).reset_index()
    write_csv_atomic(stc, AN / "population_status_counts.csv")
    return h1, npm


def identified_interval_positions(sub="population"):
    """Where each fit's horizon survival E[(1-p)^(T-1)] sits between the two
    extreme members (0 = lower member, 1 = upper member), and where P sits.

    The extreme members are exactly the minimizer and maximizer of horizon
    survival over the identified set M_tau (extreme_pair_rebuild.py), so this
    locates every estimate inside the identified interval of the quantity that
    separates the members.  Uses weight snapshots saved in runs/.
    """
    from experiments.em_mechanism.common import dists
    from experiments.em_mechanism.pairs import load_pairs
    rows = []
    D = dists()
    from experiments.em_mechanism.snapshots import population_task, population_tasks
    for tag in [t for t in population_tasks(sub) if t.endswith("continuous")]:
        tau, G = int(tag.split("_")[0][3:]), int(tag.split("_")[1][1:])
        task = population_task(sub, tag)
        grid = np.linspace(0, 1, G)
        meta = task.rows
        pairs = load_pairs(tau)
        snaps = sorted(task.snaps)
        from experiments.em_mechanism.core import certified_npmle, exact_cell_probs, grid_and_psi
        _, Psi = grid_and_psi(G, tau)
        npw = {}
        for dn, dist in D.items():
            q = exact_cell_probs(dist, tau)
            npw[(dn, "npmle")] = certified_npmle(q, grid, tau, Psi).weights
            npw[(dn, "npmle_sharp")] = certified_npmle(q, grid, tau, Psi, delta=1e-11).weights
        for p in pairs:
            T = p["T"]
            s = lambda atoms, w: float(np.dot(w, (1 - np.asarray(atoms)) ** (T - 1)))
            lo = s(p["members"][0]["atoms"], p["members"][0]["weights"])
            hi = s(p["members"][1]["atoms"], p["members"][1]["weights"])
            P_ = D[p["distribution"]].mixed(0, T - 1)
            for i, (dn, init) in enumerate(meta):
                if dn != p["distribution"]:
                    continue
                for k in snaps:
                    v = float(task.snaps[k][i] @ (1 - grid) ** (T - 1))
                    rows.append({"tau": tau, "G": G, "config_id": p["config_id"], "distribution": dn,
                                 "T": T, "beta": p["beta"], "init": init, "iteration": k,
                                 "survival": v, "survival_lower": lo, "survival_upper": hi,
                                 "survival_P": P_, "position": (v - lo) / (hi - lo),
                                 "position_P": (P_ - lo) / (hi - lo)})
            for tag in ("npmle", "npmle_sharp"):
                v = float(npw[(p["distribution"], tag)] @ (1 - grid) ** (T - 1))
                rows.append({"tau": tau, "G": G, "config_id": p["config_id"], "distribution": p["distribution"],
                             "T": T, "beta": p["beta"], "init": tag, "iteration": -1,
                             "survival": v, "survival_lower": lo, "survival_upper": hi,
                             "survival_P": P_, "position": (v - lo) / (hi - lo),
                             "position_P": (P_ - lo) / (hi - lo)})
    df = pd.DataFrame(rows)
    if len(df):
        write_csv_atomic(df, AN / "population_identified_interval_position.csv.gz")
    return df


# ------------------------------------------------------------------ finite
def load_finite(sub="finite", npmle_invalid="raw"):
    """Per-run tables.  NPMLE fits that violate the support-size guard are scored,
    by default, at their raw argmax date ("raw") -- the convention of the archived
    main benchmark, whose rebuild (r15_rebuild_main_benchmark.py) recovers those
    runs by their KKT certificate.  "fallback1" scores them as immediate
    allocation instead (sensitivity)."""
    fs = sorted(glob.glob(str(OUT / sub / "cells" / "*_table.csv.gz")))
    if not fs:
        return None, None
    t = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    s = pd.concat([pd.read_csv(f.replace("_table", "_status")) for f in fs], ignore_index=True)
    cells = {c.cid: c for c in gamma1_cells()}
    from experiments.em_mechanism.common import reference_regret
    from experiments.em_mechanism.pairs import load_pairs
    pr = {p["config_id"]: p for p in load_pairs(2)}
    bad = (t.method == "npmle") & (t.npmle_status == "invalid")
    for i in np.flatnonzero(bad.to_numpy()):
        cid = t.config_id.iat[i]
        d = int(t.npmle_date.iat[i]) if npmle_invalid == "raw" else 1
        ix = t.index[i]
        t.loc[ix, "date"] = d
        t.loc[ix, "regret_ref"] = reference_regret(cells[cid])[d - 1]
        t.loc[ix, "regret_lower"] = pr[cid]["regret"][0][d - 1]
        t.loc[ix, "regret_upper"] = pr[cid]["regret"][1][d - 1]
    t.attrs["npmle_invalid"] = npmle_invalid
    return t, s


def method_frame(t):
    """One row per (kind, config, N, seed, procedure) with procedure labels."""
    cap = int(t[t.method == "em"].iteration.max())
    em = t[(t.method == "em") & t.iteration.isin([0, 30, 55, 80, cap])].copy()
    em["procedure"] = np.where(em.iteration == cap, "EM_cap", "EM" + em.iteration.astype(str))
    em.loc[em.iteration == 0, "procedure"] = "EM0_init"
    npm = t[t.method == "npmle"].copy(); npm["procedure"] = "NPMLE"
    cv = t[t.method == "em_cv"].copy(); cv["procedure"] = "EM_CV"
    out = pd.concat([em, npm, cv], ignore_index=True)
    out["regret_worst"] = out[["regret_lower", "regret_upper"]].max(axis=1)
    return out, cap


def summarize(df, truth="ref"):
    r = df[f"regret_{truth}"].to_numpy()
    return {"n": len(r), "MSR_x1e3": 1e3 * float(np.mean(r ** 2)), "mean_regret": float(np.mean(r)),
            "p10_efficiency": p10(r), "exact_date_share": float(np.mean(r == 0))}


def finite_summary(t, s, label):
    M, cap = method_frame(t)
    rows = []
    st = s[~s.init.str.startswith("cvfold")]
    for (kind, N, init, proc), g in M.groupby(["kind", "N", "init", "procedure"]):
        r = {"kind": kind, "N": N, "init": init, "procedure": proc, **summarize(g)}
        r["share_date_differs_from_npmle"] = float((g.date != g.npmle_date).mean()) if proc != "NPMLE" else 0.0
        if proc == "EM_cap":
            sc = st[(st.kind == kind) & (st.N == N) & (st.init == init)].status.value_counts()
            for k_ in ("certified", "stagnated", "cap_reached", "numerical_failure"):
                r[f"status_{k_}"] = int(sc.get(k_, 0))
            r["runtime_s_per_trajectory"] = float(st[(st.kind == kind) & (st.N == N) & (st.init == init)]
                                                  .runtime_s_per_trajectory.mean())
        if proc == "NPMLE":
            for k_ in ("certified", "valid_uncertified", "invalid"):
                r[f"status_{k_}"] = int((g.npmle_status == k_).sum())
            r["runtime_s_per_trajectory"] = float(g.runtime_s.mean())
        if proc == "EM_CV":
            r["median_cv_iteration"] = float(g.iteration.median())
        rows.append(r)
    # NPMLE rows are init-independent: label them once
    out = pd.DataFrame(rows)
    out.loc[out.procedure == "NPMLE", "init"] = "npmle"
    out = out.drop_duplicates(subset=["kind", "N", "init", "procedure"])
    # non-uniform average (each dataset weighted once per init)
    nu = M[(M.method == "em") & (M.init != "uniform")]
    for (kind, N, proc), g in nu.groupby(["kind", "N", "procedure"]):
        out = pd.concat([out, pd.DataFrame([{"kind": kind, "N": N, "init": "nonuniform_mean7",
                                             "procedure": proc, **summarize(g),
                                             "share_date_differs_from_npmle": float((g.date != g.npmle_date).mean())}])])
    write_csv_atomic(out, AN / f"{label}_summary_by_init.csv")
    return out, M, cap


def cluster_boot_diff(M, a, b, kind, N, truth="ref", stat="msr", rng=None):
    """Paired difference stat(a) - stat(b), two-stage cluster bootstrap (configs, then datasets)."""
    rng = rng or np.random.default_rng(BOOT_SEED)
    sel = lambda proc, init: M[(M.kind == kind) & (M.N == N) & (M.procedure == proc[0]) & (M.init == proc[1])]
    A, Bm = sel(a, None if False else a[1]), sel(b, b[1])
    key = ["config_id", "seed_index"]
    m = A[key + [f"regret_{truth}"]].merge(Bm[key + [f"regret_{truth}"]], on=key, suffixes=("_a", "_b"))
    f = (lambda x: x ** 2) if stat == "msr" else (lambda x: x)
    m["d"] = f(m[f"regret_{truth}_a"]) - f(m[f"regret_{truth}_b"])
    cfgs = sorted(m.config_id.unique())
    D = [m[m.config_id == c].d.to_numpy() for c in cfgs]
    S = min(len(x) for x in D)
    Dm = np.vstack([x[:S] for x in D])
    est = float(Dm.mean())
    C = len(cfgs)
    reps = np.empty(B_BOOT)
    for bb in range(B_BOOT):
        ci = rng.integers(C, size=C)
        si = rng.integers(S, size=(C, S))
        reps[bb] = Dm[ci[:, None], si].mean()
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return {"estimate": est, "ci_lo": float(lo), "ci_hi": float(hi), "n_configs": C,
            "n_per_config": S, "configs_favoring_a": int((Dm.mean(1) < 0).sum()),
            "configs_favoring_b": int((Dm.mean(1) > 0).sum())}


def headline_inference(M, cap, label):
    rows = []
    rng = np.random.default_rng(BOOT_SEED)
    for N in sorted(M.N.unique()):
        for proc in ("EM30", "EM55", "EM80", "EM_cap"):
            for init in ("uniform",):
                for stat in ("msr", "mean"):
                    r = cluster_boot_diff(M, (proc, init), ("NPMLE", "npmle"), "ref", N, stat=stat, rng=rng)
                    rows.append({"comparison": f"{proc}[{init}] - NPMLE", "N": N, "stat": stat,
                                 "scale": "x1e3" if stat == "msr" else "raw",
                                 **{k: (v * 1e3 if stat == "msr" and k in ("estimate", "ci_lo", "ci_hi") else v)
                                    for k, v in r.items()}})
        r = cluster_boot_diff(M, ("EM_CV", "uniform"), ("NPMLE", "npmle"), "ref", N, stat="msr", rng=rng)
        rows.append({"comparison": "EM_CV[uniform] - NPMLE", "N": N, "stat": "msr", "scale": "x1e3",
                     **{k: (v * 1e3 if k in ("estimate", "ci_lo", "ci_hi") else v) for k, v in r.items()}})
        r = cluster_boot_diff(M, ("EM55", "uniform"), ("EM_cap", "uniform"), "ref", N, stat="msr", rng=rng)
        rows.append({"comparison": "EM55[uniform] - EM_cap[uniform]", "N": N, "stat": "msr", "scale": "x1e3",
                     **{k: (v * 1e3 if k in ("estimate", "ci_lo", "ci_hi") else v) for k, v in r.items()}})
        # EM55 averaged over the seven non-uniform initializations vs NPMLE
        nu = M[(M.kind == "ref") & (M.N == N) & (M.procedure == "EM55") & (M.init != "uniform")]
        avg = nu.groupby(["config_id", "seed_index"]).apply(
            lambda g: np.mean(g.regret_ref ** 2), include_groups=False).rename("sq").reset_index()
        npm = M[(M.kind == "ref") & (M.N == N) & (M.procedure == "NPMLE")][["config_id", "seed_index", "regret_ref"]]
        m = avg.merge(npm, on=["config_id", "seed_index"])
        m["d"] = m.sq - m.regret_ref ** 2
        cfgs = sorted(m.config_id.unique())
        Dm = np.vstack([m[m.config_id == c].d.to_numpy()[:50] for c in cfgs])
        reps = []
        for bb in range(B_BOOT):
            ci = rng.integers(len(cfgs), size=len(cfgs)); si = rng.integers(Dm.shape[1], size=(len(cfgs), Dm.shape[1]))
            reps.append(Dm[ci[:, None], si].mean())
        rows.append({"comparison": "EM55[mean of 7 non-uniform inits] - NPMLE", "N": N, "stat": "msr",
                     "scale": "x1e3", "estimate": 1e3 * Dm.mean(), "ci_lo": 1e3 * np.quantile(reps, .025),
                     "ci_hi": 1e3 * np.quantile(reps, .975), "n_configs": len(cfgs), "n_per_config": Dm.shape[1],
                     "configs_favoring_a": int((Dm.mean(1) < 0).sum()),
                     "configs_favoring_b": int((Dm.mean(1) > 0).sum())})
    out = pd.DataFrame(rows)
    write_csv_atomic(out, AN / f"{label}_headline_inference.csv")
    return out


def heterogeneity(M, label):
    """Per family and per configuration: MSR(EM55) - MSR(NPMLE), every init."""
    ref = M[M.kind == "ref"]
    npm = ref[ref.procedure == "NPMLE"][["config_id", "N", "seed_index", "regret_ref"]].rename(
        columns={"regret_ref": "r_np"})
    rows = []
    for proc in ("EM30", "EM55", "EM80", "EM_cap"):
        x = ref[ref.procedure == proc].merge(npm, on=["config_id", "N", "seed_index"])
        x["d"] = x.regret_ref ** 2 - x.r_np ** 2
        g = x.groupby(["N", "init", "distribution", "T", "beta", "config_id"]).agg(
            msr_em=("regret_ref", lambda v: 1e3 * np.mean(v ** 2)),
            msr_npmle=("r_np", lambda v: 1e3 * np.mean(v ** 2)),
            diff_x1e3=("d", lambda v: 1e3 * np.mean(v))).reset_index()
        g["procedure"] = proc
        rows.append(g)
    per_cfg = pd.concat(rows, ignore_index=True)
    write_csv_atomic(per_cfg, AN / f"{label}_by_config.csv")
    fam = per_cfg.groupby(["procedure", "N", "init", "distribution"]).agg(
        msr_em=("msr_em", "mean"), msr_npmle=("msr_npmle", "mean"), diff_x1e3=("diff_x1e3", "mean"),
        configs_em_worse=("diff_x1e3", lambda v: int((v > 1e-12).sum())),
        configs_em_better=("diff_x1e3", lambda v: int((v < -1e-12).sum()))).reset_index()
    write_csv_atomic(fam, AN / f"{label}_by_family.csv")
    rev = per_cfg.groupby(["procedure", "N", "init"]).agg(
        configs_em_worse=("diff_x1e3", lambda v: int((v > 1e-12).sum())),
        configs_tie=("diff_x1e3", lambda v: int((v.abs() <= 1e-12).sum())),
        configs_em_better=("diff_x1e3", lambda v: int((v < -1e-12).sum()))).reset_index()
    write_csv_atomic(rev, AN / f"{label}_reversal_counts.csv")
    return per_cfg, fam, rev


def pair_risk(M, label, kind="pair"):
    """Worst-member risks on the compatible pairs, per procedure and init, with
    a within-pair trial bootstrap for EM - NPMLE differences."""
    P = M[M.kind == kind]
    rows, rng = [], np.random.default_rng(BOOT_SEED)
    groups = list(P.groupby(["N", "init", "procedure"]))
    for (N, init, proc), g in groups:
        pm = g.groupby("config_id").agg(lo=("regret_lower", "mean"), hi=("regret_upper", "mean"),
                                        tw=("regret_worst", "mean"), ref=("regret_ref", "mean"))
        rows.append({"kind": kind, "N": N, "init": init, "procedure": proc,
                     "rho_max_of_member_means": float(pm[["lo", "hi"]].max(axis=1).mean()),
                     "mean_trialwise_max": float(pm.tw.mean()),
                     "mean_regret_lower": float(pm.lo.mean()), "mean_regret_upper": float(pm.hi.mean()),
                     "mean_regret_ref": float(pm.ref.mean()),
                     "pairs_where_lower_is_worst": int((pm.lo > pm.hi).sum()),
                     "n_pairs": len(pm), "trials_per_pair": int(g.groupby("config_id").size().min())})
    risk = pd.DataFrame(rows)
    risk.loc[risk.procedure == "NPMLE", "init"] = "npmle"
    risk = risk.drop_duplicates(subset=["kind", "N", "init", "procedure"])
    write_csv_atomic(risk, AN / f"{label}_{kind}_risk.csv")

    # bootstrap of rho differences vs NPMLE (resample trials within each pair)
    diffs = []
    for N in sorted(P.N.unique()):
        npm = P[(P.N == N) & (P.procedure == "NPMLE")]
        for proc in ("EM30", "EM55", "EM80", "EM_cap", "EM_CV"):
            for init in sorted(P[P.procedure == proc].init.unique()):
                a = P[(P.N == N) & (P.procedure == proc) & (P.init == init)]
                m = a.merge(npm, on=["config_id", "seed_index"], suffixes=("", "_np"))
                cfgs = sorted(m.config_id.unique())
                arr = {c: m[m.config_id == c][["regret_lower", "regret_upper", "regret_lower_np",
                                               "regret_upper_np"]].to_numpy() for c in cfgs}
                def rho(sel):
                    va = np.mean([max(arr[c][s, 0].mean(), arr[c][s, 1].mean()) for c, s in zip(cfgs, sel)])
                    vb = np.mean([max(arr[c][s, 2].mean(), arr[c][s, 3].mean()) for c, s in zip(cfgs, sel)])
                    return va - vb
                full = [np.arange(len(arr[c])) for c in cfgs]
                est = rho(full)
                reps = [rho([rng.integers(len(arr[c]), size=len(arr[c])) for c in cfgs]) for _ in range(B_BOOT)]
                tw_a = np.mean([np.maximum(arr[c][:, 0], arr[c][:, 1]).mean() for c in cfgs])
                tw_b = np.mean([np.maximum(arr[c][:, 2], arr[c][:, 3]).mean() for c in cfgs])
                per_pair = np.array([max(arr[c][:, 0].mean(), arr[c][:, 1].mean())
                                     - max(arr[c][:, 2].mean(), arr[c][:, 3].mean()) for c in cfgs])
                diffs.append({"kind": kind, "N": N, "comparison": f"{proc}[{init}] - NPMLE",
                              "rho_diff": est, "ci_lo": float(np.quantile(reps, .025)),
                              "ci_hi": float(np.quantile(reps, .975)), "trialwise_max_diff": tw_a - tw_b,
                              "pairs_em_better": int((per_pair < -1e-12).sum()),
                              "pairs_em_worse": int((per_pair > 1e-12).sum()),
                              "pairs_tied": int((np.abs(per_pair) <= 1e-12).sum())})
    d = pd.DataFrame(diffs)
    write_csv_atomic(d, AN / f"{label}_{kind}_risk_diff.csv")
    return risk, d


def path_aggregates(sub, label):
    """MSR/mean regret/worst-member regret along the whole recorded path, per init
    (from the committed dense arrays)."""
    cells = {c.cid: c for c in gamma1_cells()}
    from experiments.em_mechanism.common import reference_regret
    from experiments.em_mechanism.pairs import load_pairs
    pr = {p["config_id"]: p for p in load_pairs(2)}
    rows = []
    for f in sorted(glob.glob(str(OUT / sub / "cells" / "*_dense.npz"))):
        name = Path(f).name
        kind, cid, Ns = name.split("_")[:3]
        N = int(Ns[1:])
        z = np.load(f)
        Rref = reference_regret(cells[cid]); Rlo = np.asarray(pr[cid]["regret"][0]); Rhi = np.asarray(pr[cid]["regret"][1])
        D = z["dates"].astype(int)
        inits = z["inits"]
        npd = z["npmle_dates"].astype(int)
        seeds = z["seeds"]
        seed_pos = {s: i for i, s in enumerate(np.unique(seeds))}
        npd_rows = npd[[seed_pos[s] for s in seeds]]
        rr, rl, rh = Rref[D - 1], Rlo[D - 1], Rhi[D - 1]
        for init in np.unique(inits):
            m = inits == init
            agg = pd.DataFrame({
                "iteration": z["iters"],
                "msr_ref": (rr[:, m] ** 2).mean(1), "mean_regret_ref": rr[:, m].mean(1),
                "mean_regret_lower": rl[:, m].mean(1), "mean_regret_upper": rh[:, m].mean(1),
                "mean_trialwise_max": np.maximum(rl, rh)[:, m].mean(1),
                "share_same_date_as_npmle": (D[:, m] == npd_rows[None, m]).mean(1),
                "mean_gap": z["gap"][:, m].mean(1), "median_kkt": np.median(z["kkt"][:, m], axis=1),
                "share_kkt_le_1e8": (z["kkt"][:, m] <= 1e-8).mean(1)})
            agg["kind"], agg["config_id"], agg["N"], agg["init"] = kind, cid, N, init
            agg["distribution"], agg["T"], agg["beta"] = cells[cid].dist_name, cells[cid].T, cells[cid].beta
            rows.append(agg)
        # date changes along the path, per trajectory
    if not rows:
        return None
    df = pd.concat(rows, ignore_index=True)
    write_csv_atomic(df, AN / f"{label}_path_by_config.csv.gz")
    return df


def path_change_counts(sub, label):
    rows = []
    for f in sorted(glob.glob(str(OUT / sub / "cells" / "*_dense.npz"))):
        kind, cid, Ns = Path(f).name.split("_")[:3]
        z = np.load(f)
        D = z["dates"].astype(int)
        it = z["iters"]
        npd = z["npmle_dates"].astype(int)
        seeds = z["seeds"]
        pos = {s: i for i, s in enumerate(np.unique(seeds))}
        for j in range(D.shape[1]):
            d = D[:, j]
            ch = np.flatnonzero(np.diff(d) != 0)
            rows.append({"kind": kind, "config_id": cid, "N": int(Ns[1:]), "seed_index": int(seeds[j]),
                         "init": z["inits"][j], "n_date_changes": len(ch),
                         "n_changes_after_55": int((it[ch + 1] > 55).sum()),
                         "last_change_iter": int(it[ch[-1] + 1]) if len(ch) else 0,
                         "date_55": int(d[np.searchsorted(it, 55)]), "date_cap": int(d[-1]),
                         "npmle_date": int(npd[pos[seeds[j]]])})
    df = pd.DataFrame(rows)
    if len(df):
        write_csv_atomic(df, AN / f"{label}_path_changes.csv.gz")
    return df


def grid_refinement(base="finite"):
    """EM55/cap and NPMLE across G on the refinement subset (uniform init)."""
    out = []
    for sub, G in ((base, 200), (f"{base}_G400", 400), (f"{base}_G800", 800)):
        t, _ = load_finite(sub)
        if t is None:
            continue
        out.append((G, t))
    if len(out) < 2:
        return None
    subset = set(out[-1][1].config_id.unique())
    seeds = set(out[-1][1].seed_index.unique())
    rows = []
    for G, t in out:
        t = t[t.config_id.isin(subset) & t.seed_index.isin(seeds) & (t.kind == "ref")]
        M, cap = method_frame(t)
        M = M[(M.init == "uniform") | (M.procedure == "NPMLE")]
        for (N, proc), g in M.groupby(["N", "procedure"]):
            rows.append({"G": G, "N": N, "procedure": proc, **summarize(g),
                         "mean_worst_member": float(g.regret_worst.mean())})
    df = pd.DataFrame(rows)
    write_csv_atomic(df, AN / "grid_refinement_finite.csv")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--finite", default="finite")
    ap.add_argument("--population", default="population")
    a = ap.parse_args(argv)
    AN.mkdir(parents=True, exist_ok=True)
    report = {}
    P = load_population(a.population)
    if P is not None and P["decisions"] is not None:
        h1, npm = population_tables(P)
        identified_interval_positions(a.population)
        report["population_tasks"] = int(P["status"][["tau", "G", "variant"]].drop_duplicates().shape[0])
    t, s = load_finite(a.finite)
    if t is not None:
        summ, M, cap = finite_summary(t, s, a.finite)
        write_csv_atomic(M, AN / f"{a.finite}_procedures.csv.gz")
        headline_inference(M, cap, a.finite)
        heterogeneity(M, a.finite)
        if (M.kind == "pair").any():
            pair_risk(M, a.finite, "pair")
        pair_risk(M, a.finite, "ref")       # ref datasets evaluated under the pair members
        path_aggregates(a.finite, a.finite)
        path_change_counts(a.finite, a.finite)
        report["finite_cells"] = int(t[["kind", "config_id", "N"]].drop_duplicates().shape[0])
        report["finite_cap"] = cap
        # sensitivity: invalid NPMLE fits scored as immediate allocation
        t1, _ = load_finite(a.finite, npmle_invalid="fallback1")
        M1, _ = method_frame(t1)
        sens = []
        for (kind, N), g in M1[M1.procedure == "NPMLE"].groupby(["kind", "N"]):
            g0 = M[(M.procedure == "NPMLE") & (M.kind == kind) & (M.N == N)]
            sens.append({"kind": kind, "N": N, "n_invalid": int((g.npmle_status == "invalid").sum()),
                         "MSR_x1e3_raw": summarize(g0)["MSR_x1e3"], "MSR_x1e3_fallback1": summarize(g)["MSR_x1e3"],
                         "rho_note": "see *_pair_risk.csv for the raw convention"})
        write_csv_atomic(pd.DataFrame(sens), AN / f"{a.finite}_npmle_invalid_sensitivity.csv")
    grid_refinement(a.finite)
    (AN / "analysis_manifest.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
