"""Recompute every mechanism and trajectory claim from saved outputs, and check it.

Uses only saved results (no EM is re-run): results/em_mechanism/{finite,
analysis,mechanism,population,validation}/ and the frozen pairs.  Writes

  results/em_mechanism/verification/claims.csv          one row per claim: id, value, source
  results/em_mechanism/verification/claims_check.md     the same, plus pass/fail of every
                                                         check and of every report token
  results/em_mechanism/mechanism/intervention_accounting.csv
  results/em_mechanism/mechanism/regret_reduction.csv
  results/em_mechanism/mechanism/m3_error_by_init.csv
  results/em_mechanism/mechanism/event_counts.csv

Exit status 1 if any check or report token fails.

Conventions (see experiments/em_mechanism/README.md, "Definitions"):
  EM55        EM after exactly 55 updates from uniform weights
  endpoint    EM after 1e6 updates (the iteration cap); certification status separate
  regret      relative regret under the reference population P, 1 - W^d / W^{t*}
  worse event reference dataset (uniform start) with regret(endpoint) > regret(EM55) + 1e-12
  improve     regret(endpoint) < regret(EM55) - 1e-12

Usage: python -m experiments.em_mechanism.verify_claims
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np
import pandas as pd

from experiments.em_mechanism.common import OUT, gamma1_cells, write_csv_atomic

V = OUT / "verification"
M = OUT / "mechanism"
A = OUT / "analysis"
CAP = 1_000_000
CLAIMS, CHECKS = [], []


def claim(cid, value, source, desc):
    CLAIMS.append({"id": cid, "value": value, "source": source, "description": desc})
    return value


def check(cid, ok, detail):
    CHECKS.append({"id": cid, "pass": bool(ok), "detail": detail})


def pct(x):
    return f"{100 * x:.1f}%"


# ------------------------------------------------------------------ events
def events():
    t = pd.concat([pd.read_csv(f, usecols=["config_id", "N", "seed_index", "method", "init", "iteration",
                                            "date", "regret_ref"])
                   for f in glob.glob(str(OUT / "finite" / "cells" / "ref_*_table.csv.gz"))])
    e = t[(t.method == "em") & (t.init == "uniform") & t.iteration.isin([55, CAP])]
    w = e.pivot_table(index=["config_id", "N", "seed_index"], columns="iteration",
                      values=["date", "regret_ref"]).reset_index()
    w.columns = ["config_id", "N", "seed", "d55", "dcap", "r55", "rcap"]
    d = w.rcap - w.r55
    w["cls"] = np.where(d > 1e-12, "worse", np.where(d < -1e-12, "improve", "same"))
    cnt = w.groupby(["N", "cls"]).size().unstack(fill_value=0)
    write_csv_atomic(cnt.reset_index(), M / "event_counts.csv")
    for N in (1000, 100000):
        tot = int(cnt.loc[N].sum())
        claim(f"events_total_N{N}", tot, "finite/cells/ref_*_table.csv.gz", "reference datasets, uniform start")
        for c in ("worse", "improve", "same"):
            claim(f"events_{c}_N{N}", int(cnt.loc[N, c]), "finite/cells/ref_*_table.csv.gz",
                  f"{c} datasets (EM55 vs endpoint)")
    check("events_660_110", cnt.loc[1000, "worse"] == 660 and cnt.loc[100000, "worse"] == 110,
          f"worse = {cnt.loc[1000, 'worse']}, {cnt.loc[100000, 'worse']}")
    check("events_32_150", cnt.loc[1000, "improve"] == 32 and cnt.loc[100000, "improve"] == 150,
          f"improve = {cnt.loc[1000, 'improve']}, {cnt.loc[100000, 'improve']}")
    check("worse_all_change_date", bool((w[w.cls != "same"].d55 != w[w.cls != "same"].dcap).all()),
          "every worse/improve event has a different date at EM55 and at the endpoint")
    ev = pd.read_csv(M / "events.csv.gz")
    m = ev.merge(w, left_on=["config_id", "N", "seed_index"], right_on=["config_id", "N", "seed"],
                 suffixes=("", "_w"))
    check("events_file_matches_tables", len(m) == len(ev) and (m.cls == m.cls_w).all()
          and (m.d55 == m.d55_w).all() and (m.dcap == m.dcap_w).all(),
          f"mechanism/events.csv.gz rows {len(ev)} agree with the finite tables")
    return w, ev


# ------------------------------------------------------------------ interventions
TESTS = [
    # name, file, target ("d55" keep EM55 date | "dcap" move to endpoint date), prespecified?, label
    ("T1_minimal_change", "constraint_tests", "d55", True,
     "endpoint fit; L1-closest to EM55 (one LP projection)"),
    ("T2_freeze_m4_m6", "constraint_tests", "d55", True, "endpoint fit + m4..m6 fixed at EM55; L1-closest to endpoint"),
    ("T2_freeze_m4_m8", "constraint_tests", "d55", True, "endpoint fit + m4..m8 fixed at EM55; L1-closest to endpoint"),
    ("T3_freeze_decision_bins", "constraint_tests", "d55", True,
     "endpoint fit + masses of [.05,.1),[.1,.2) fixed at EM55; L1-closest to endpoint"),
    ("T4_freeze_irrelevant_bins", "constraint_tests", "d55", True,
     "endpoint fit + masses of [0,.02),[.4,1] fixed at EM55; L1-closest to endpoint (negative control)"),
    ("T5_import_decision_bins", "constraint_tests", "dcap", True,
     "EM55 fit + decision-bin masses set to endpoint values; L1-closest to EM55"),
    ("T1r_late_structure_55_fit", "posthoc_moment_tests", "d55", False,
     "EM55's m1..m3; L1-closest to endpoint"),
    ("Tm1_only_m1", "posthoc_moment_tests", "dcap", False, "EM55 with m1 set to endpoint value; L1-closest to EM55"),
    ("Tm2_only_m2", "posthoc_moment_tests", "dcap", False, "EM55 with m2 set to endpoint value; L1-closest to EM55"),
    ("Tm3_only_m3", "posthoc_moment_tests", "dcap", False, "EM55 with m3 set to endpoint value; L1-closest to EM55"),
    ("Tno3_all_but_m3", "posthoc_moment_tests", "d55", False,
     "endpoint m1,m2 + EM55 m3; L1-closest to EM55"),
    ("Ttrue3_late_with_true_m3", "posthoc_moment_tests", "d55", False,
     "endpoint m1,m2 + true m3; L1-closest to endpoint"),
]


def interventions(ev):
    data = {"constraint_tests": pd.read_csv(M / "constraint_tests.csv.gz"),
            "posthoc_moment_tests": pd.read_csv(M / "posthoc_moment_tests.csv.gz")}
    rows = []
    for name, f, tgt, pre, label in TESTS:
        df = data[f]
        for (N, cls), g in df.groupby(["N", "cls"]):
            att = len(g)
            feas = g[f"{name}_feasible"].fillna(False).astype(bool)
            x = g[feas]
            succ = int((x[f"{name}_date"] == x[tgt]).sum()) if len(x) else 0
            r_col = f"{name}_regret_ref" if f"{name}_regret_ref" in x else None
            r55 = x.r55_ref if "r55_ref" in x else x.r55
            rcap = x.rcap_ref if "rcap_ref" in x else x.rcap
            rows.append({"test": name, "prespecified": pre, "construction": label, "N": N, "class": cls,
                         "success_means": "keeps EM55 date" if tgt == "d55" else "moves to endpoint date",
                         "attempted": att, "feasible": int(feas.sum()), "excluded_infeasible": int(att - feas.sum()),
                         "successful": succ,
                         "share_of_feasible": succ / feas.sum() if feas.sum() else np.nan,
                         "share_of_attempted": succ / att,
                         "mean_regret_test_feasible": float(x[r_col].mean()) if r_col and len(x) else np.nan,
                         "mean_regret_em55_feasible": float(r55.mean()) if len(x) else np.nan,
                         "mean_regret_endpoint_feasible": float(rcap.mean()) if len(x) else np.nan})
    acc = pd.DataFrame(rows)
    write_csv_atomic(acc, M / "intervention_accounting.csv")
    n_ev = ev.groupby(["N", "cls"]).size()
    for _, r in acc.iterrows():
        check(f"attempted_{r.test}_N{r.N}_{r['class']}", r.attempted == n_ev.loc[(r.N, r["class"])],
              f"{r.test} attempted {r.attempted} = events {n_ev.loc[(r.N, r['class'])]}")
    get = lambda t, N, c: acc[(acc.test == t) & (acc.N == N) & (acc["class"] == c)].iloc[0]
    for N in (1000, 100000):
        for t in [x[0] for x in TESTS]:
            r = get(t, N, "worse")
            claim(f"{t}_worse_N{N}_feasible", int(r.feasible), "mechanism/intervention_accounting.csv",
                  f"{t}: feasible of {r.attempted} worse events")
            if r.feasible:
                claim(f"{t}_worse_N{N}_share", round(r.share_of_feasible, 4),
                      "mechanism/intervention_accounting.csv", f"{t}: share of feasible worse events ({r.success_means})")
            r = get(t, N, "improve")
            if r.feasible:
                claim(f"{t}_improve_N{N}_share", round(r.share_of_feasible, 4),
                      "mechanism/intervention_accounting.csv", f"{t}: share of feasible improve events ({r.success_means})")
    k = get("Tno3_all_but_m3", 1000, "worse")
    check("hold_m3_77pct", abs(k.share_of_feasible - 0.77) < 0.005,
          f"N=1e3: {k.successful}/{k.feasible} feasible = {pct(k.share_of_feasible)}; of attempted {pct(k.share_of_attempted)}")
    k = get("Tno3_all_but_m3", 100000, "worse")
    check("hold_m3_70pct", abs(k.share_of_feasible - 0.70) < 0.005,
          f"N=1e5: {k.successful}/{k.feasible} feasible = {pct(k.share_of_feasible)}; of attempted {pct(k.share_of_attempted)}")
    k = get("T1_minimal_change", 1000, "worse")
    check("T1_keeps_1.2pct", abs(k.share_of_feasible - 0.012) < 0.001, f"{k.successful}/{k.feasible}")
    return acc, data


def regret_reduction(data, ev):
    """The 'about 89%' figure: (mean r_end - mean r_test) / (mean r_end - mean r_55),
    means over FEASIBLE worse events; plus two alternatives that keep infeasible events."""
    ph = data["posthoc_moment_tests"]
    rows = []
    for N in (1000, 100000):
        g = ph[(ph.N == N) & (ph.cls == "worse")]
        feas = g.Tno3_all_but_m3_feasible.fillna(False).astype(bool)
        x = g[feas]
        r55, rc, rt = x.r55.mean(), x.rcap.mean(), x.Tno3_all_but_m3_regret_ref.mean()
        primary = (rc - rt) / (rc - r55)
        # alternative A: infeasible events count with no intervention (regret stays at endpoint)
        rt_all = np.where(feas, g.Tno3_all_but_m3_regret_ref.fillna(0), g.rcap)
        altA = (g.rcap.mean() - rt_all.mean()) / (g.rcap.mean() - g.r55.mean())
        # alternative B: mean of per-event shares over feasible events with r_end > r_55
        per = (x.rcap - x.Tno3_all_but_m3_regret_ref) / (x.rcap - x.r55)
        rows.append({"N": N, "worse_events": len(g), "feasible": int(feas.sum()),
                     "excluded_infeasible": int((~feas).sum()),
                     "mean_regret_em55_feasible": r55, "mean_regret_endpoint_feasible": rc,
                     "mean_regret_hold_m3_feasible": rt,
                     "reduction_primary_ratio_of_means_feasible": primary,
                     "reduction_altA_ratio_of_means_all_infeasible_as_endpoint": altA,
                     "reduction_altB_mean_of_event_ratios_feasible": per.mean(),
                     "reduction_altB_median": per.median()})
        claim(f"reduction_primary_N{N}", round(primary, 4), "mechanism/regret_reduction.csv",
              f"regret-increase reduction, ratio of means over {int(feas.sum())} feasible of {len(g)} worse events")
        claim(f"reduction_altA_N{N}", round(altA, 4), "mechanism/regret_reduction.csv",
              "same, all worse events, infeasible counted as no intervention")
    df = pd.DataFrame(rows)
    write_csv_atomic(df, M / "regret_reduction.csv")
    p = df[df.N == 1000].iloc[0]
    check("reduction_89pct", abs(p.reduction_primary_ratio_of_means_feasible - 0.89) < 0.01,
          f"N=1e3 primary {pct(p.reduction_primary_ratio_of_means_feasible)} over {p.feasible}/{p.worse_events} "
          f"feasible (r55 {p.mean_regret_em55_feasible:.4f}, endpoint {p.mean_regret_endpoint_feasible:.4f}, "
          f"held m3 {p.mean_regret_hold_m3_feasible:.4f}); alt A {pct(p.reduction_altA_ratio_of_means_all_infeasible_as_endpoint)}")
    return df


# ------------------------------------------------------------------ m3 error by initialization
def m3_by_init():
    mt = {c.dist_name: c.dist.raw_moment(3) for c in gamma1_cells()}
    t = pd.concat([pd.read_csv(f, usecols=["distribution", "N", "method", "init", "iteration", "m3"])
                   for f in glob.glob(str(OUT / "finite" / "cells" / "ref_*_table.csv.gz"))])
    t["relerr"] = (t.m3 - t.distribution.map(mt)) / t.distribution.map(mt)
    em = t[(t.method == "em") & t.iteration.isin([0, 55, CAP])]
    g = em.groupby(["N", "iteration", "init"]).relerr.agg(median="median", share_under=lambda v: (v < 0).mean(),
                                                          n="size").reset_index()
    npm = t[t.method == "npmle"].groupby("N").relerr.agg(median="median", share_under=lambda v: (v < 0).mean(),
                                                         n="size").reset_index()
    npm["iteration"], npm["init"] = -1, "npmle"
    out = pd.concat([g, npm], ignore_index=True)
    write_csv_atomic(out, M / "m3_error_by_init.csv")
    for N in (1000, 100000):
        x = g[(g.N == N) & (g.iteration == 55)]
        claim(f"m3_relerr_em55_min_N{N}", round(x["median"].min(), 3), "mechanism/m3_error_by_init.csv",
              "smallest median relative m3 error at EM55 across 8 initializations")
        claim(f"m3_relerr_em55_max_N{N}", round(x["median"].max(), 3), "mechanism/m3_error_by_init.csv",
              "largest median relative m3 error at EM55 across 8 initializations")
        check(f"m3_em55_positive_all_inits_N{N}", (x["median"] > 0).all() and len(x) == 8,
              f"EM55 median m3 error > 0 for all 8 inits: {x.set_index('init')['median'].round(3).to_dict()}")
        y = g[(g.N == N) & (g.iteration == CAP)]
        claim(f"m3_relerr_endpoint_median_range_N{N}", f"{y['median'].min():.3f}..{y['median'].max():.3f}",
              "mechanism/m3_error_by_init.csv", "median relative m3 error at the endpoint, across inits")
        z = npm[npm.N == N].iloc[0]
        claim(f"m3_relerr_npmle_N{N}", round(z["median"], 3), "mechanism/m3_error_by_init.csv", "NPMLE median")
    low0 = g[(g.N == 1000) & (g.iteration == 0) & (g.init == "low")]["median"].iloc[0]
    claim("m3_relerr_low_init_iter0", round(low0, 3), "mechanism/m3_error_by_init.csv", "'low' start, iteration 0")
    check("low_init_starts_below", low0 < -0.5, f"low start median relative error at k=0: {low0:.3f}")
    return out


# ------------------------------------------------------------------ steps 1-2 descriptives
def descriptives(ev):
    for N in (1000, 100000):
        x = ev[(ev.N == N) & (ev.cls == "worse")]
        rel = lambda j: float(((x[f"wcap_m{j}"] - x[f"w55_m{j}"]) / x[f"w55_m{j}"]).median())
        for j in (1, 2, 3, 4, 6, 10):
            claim(f"worse_N{N}_median_relchange_m{j}", round(rel(j), 4), "mechanism/events.csv.gz",
                  f"median relative change of m{j}, EM55 -> endpoint, worse events")
        claim(f"worse_N{N}_effsupport_55", round(x.w55_eff_support.median(), 1), "mechanism/events.csv.gz",
              "median effective support at EM55")
        claim(f"worse_N{N}_effsupport_end", round(x.wcap_eff_support.median(), 1), "mechanism/events.csv.gz",
              "median effective support at the endpoint")
        claim(f"worse_N{N}_n99_55", float(x.w55_n99.median()), "mechanism/events.csv.gz", "median n99 at EM55")
        claim(f"worse_N{N}_n99_end", float(x.wcap_n99.median()), "mechanism/events.csv.gz", "median n99 at endpoint")
        claim(f"worse_N{N}_share_kl_true_rises", round((x.wcap_kl_true > x.w55_kl_true).mean(), 3),
              "mechanism/events.csv.gz", "share of worse events with larger KL(true || fitted) at the endpoint")
        claim(f"worse_N{N}_median_max_abs_dq", float(f"{x.max_abs_dq.median():.2g}"), "mechanism/events.csv.gz",
              "median max change of a fitted cell probability")
        y = x[x.dC.notna()]
        claim(f"worse_N{N}_mean_dC", round(y.dC.mean(), 3), "mechanism/events.csv.gz", "mean contrast change")
        claim(f"worse_N{N}_max_attr_residual", float(f"{y.attr_residual.abs().max():.2g}"),
              "mechanism/events.csv.gz", "largest |path-attribution residual|")
        claim(f"worse_N{N}_mean_obs_part", round(y.C_obs_part.mean(), 4), "mechanism/events.csv.gz",
              "mean contrast change of the T1 projection (EM55 -> projection)")
        claim(f"worse_N{N}_mean_fiber_part", round(y.C_fiber_part.mean(), 4), "mechanism/events.csv.gz",
              "mean contrast change projection -> endpoint")
        for b in range(6):
            claim(f"worse_N{N}_attr_bin{b}", round(y[f"attr{b}"].mean(), 4), "mechanism/events.csv.gz",
                  f"mean attribution to risk bin {b}")
            claim(f"worse_N{N}_massshift_bin{b}", round((x[f"wcap_mass{b}"] - x[f"w55_mass{b}"]).mean(), 3),
                  "mechanism/events.csv.gz", f"mean mass shift in risk bin {b}")
        s = x[x.k_star.notna()]
        claim(f"worse_N{N}_kstar_median", int(s.k_star.median()), "mechanism/events.csv.gz", "median switch iteration")
        claim(f"worse_N{N}_kstar_q25", int(s.k_star.quantile(.25)), "mechanism/events.csv.gz", "switch iteration q25")
        claim(f"worse_N{N}_kstar_q75", int(s.k_star.quantile(.75)), "mechanism/events.csv.gz", "switch iteration q75")
        check(f"switch_found_all_N{N}", len(s) == len(x) and (s.date_at_lo_check == s.d55).all(),
              f"{len(s)}/{len(x)} switches found; bracket-start date = EM55 date in all")
        dp = x.groupby(["d55", "dcap"]).size().sort_values(ascending=False)
        for (a, b), n in dp.head(3).items():
            claim(f"worse_N{N}_datepair_{a}to{b}", int(n), "mechanism/events.csv.gz", f"worse events moving {a}->{b}")
        early = (x.dcap < x.d55).mean()
        claim(f"worse_N{N}_share_earlier", round(early, 3), "mechanism/events.csv.gz",
              "share of worse events whose endpoint date is earlier than EM55's")


# ------------------------------------------------------------------ asymmetric loss, bias by estimator
def asymmetry():
    g = pd.read_csv(M / "regret_by_m3_error.csv")
    for _, r in g.iterrows():
        claim(f"asym_N{r.N}_{r.m3_err_bin}", round(r.mean_regret, 4), "mechanism/regret_by_m3_error.csv",
              f"mean regret, m3 error bin {r.m3_err_bin} (all estimators pooled; n = {r.n})")
        claim(f"asym_early_N{r.N}_{r.m3_err_bin}", round(r.share_too_early, 3), "mechanism/regret_by_m3_error.csv",
              "share of decisions earlier than t*")
    h = pd.read_csv(M / "m3_bias_by_estimator.csv")
    for _, r in h.iterrows():
        claim(f"bias_{r.estimator}_N{r.N}_share_under", round(r.share_m3_under, 4), "mechanism/m3_bias_by_estimator.csv",
              "share of datasets with m3 underestimated")
        claim(f"bias_{r.estimator}_N{r.N}_too_early", round(r.share_too_early, 4), "mechanism/m3_bias_by_estimator.csv",
              "share of decisions earlier than t*")


# ------------------------------------------------------------------ noise-free control and pairs
def controls():
    pc = pd.read_csv(M / "population_control.csv.gz")
    claim("noisefree_date_changes", len(pc), "mechanism/population_control.csv.gz",
          "exact probabilities, tau=2, G=200: (config, init) with a date change EM55 -> endpoint")
    for c, n in pc.cls.value_counts().items():
        claim(f"noisefree_{c}", int(n), "mechanism/population_control.csv.gz", f"{c} among those")
    imp = pc[pc.cls == "improve"]
    claim("noisefree_improve_median_relchange_m3",
          round(float(((imp.wcap_m3 - imp.w55_m3) / imp.w55_m3).median()), 4),
          "mechanism/population_control.csv.gz", "median relative m3 change, improve cases")
    claim("noisefree_improve_effsupport", f"{imp.w55_eff_support.median():.1f}->{imp.wcap_eff_support.median():.1f}",
          "mechanism/population_control.csv.gz", "median effective support EM55 -> endpoint")
    st = pd.read_csv(OUT / "population" / "tau2_G200_continuous_status.csv")
    claim("noisefree_tau2_G200_certified", int((st.status == "certified").sum()), "population/tau2_G200_continuous_status.csv",
          "of 40 trajectories certified (KKT <= 1e-8) at the 1e7 cap")
    h1 = pd.read_csv(A / "population_H1_by_init.csv")
    x = h1[(h1.tau == 2) & (h1.G == 200) & (h1.variant == "continuous")]
    claim("noisefree_configs_endpoint_worse_than_55", int(x.configs_cap_worse_than_55.sum()),
          "analysis/population_H1_by_init.csv", "sum over 8 inits of configs where endpoint regret > EM55 regret")
    pm = pd.read_csv(M / "pair_member_optimal_dates.csv")
    claim("pairs_upper_earlier_than_P", int((pm.tstar_upper < pm.tstar_P).sum()), "mechanism/pair_member_optimal_dates.csv",
          "of 30 pairs, upper member's earliest optimal date earlier than P's")
    claim("pairs_lower_equal_P", int((pm.tstar_lower == pm.tstar_P).sum()), "mechanism/pair_member_optimal_dates.csv",
          "of 30 pairs, lower member's earliest optimal date equal to P's")
    claim("pairs_lower_geq_P", int((pm.tstar_lower >= pm.tstar_P).sum()), "mechanism/pair_member_optimal_dates.csv",
          "of 30 pairs, lower member's earliest optimal date >= P's")
    claim("pairs_upper_larger_m4", int((pm.m4_upper > pm.m4_lower).sum()), "mechanism/pair_member_optimal_dates.csv",
          "of 30 pairs, upper member has the larger m4")
    r = pd.read_csv(A / "finite_pair_risk.csv")
    r = r[r.init.isin(["uniform", "npmle"])]
    for N in (1000, 100000):
        for p in ("EM55", "NPMLE", "EM_cap"):
            row = r[(r.N == N) & (r.procedure == p)].iloc[0]
            for col in ("rho_max_of_member_means", "mean_regret_lower", "mean_regret_upper", "mean_regret_ref"):
                claim(f"pair_{p}_N{N}_{col}", round(row[col], 4), "analysis/finite_pair_risk.csv", f"{p} {col}")
    d = pd.read_csv(A / "finite_pair_risk_diff.csv")
    for N in (1000, 100000):
        row = d[(d.N == N) & (d.comparison == "EM55[uniform] - NPMLE")].iloc[0]
        claim(f"pair_rho_diff_EM55_N{N}", f"{row.rho_diff:.4f} [{row.ci_lo:.4f}, {row.ci_hi:.4f}]",
              "analysis/finite_pair_risk_diff.csv", "rho(EM55) - rho(NPMLE), 95% within-pair bootstrap")


# ------------------------------------------------------------------ headline trajectory numbers
def headline():
    h = pd.read_csv(A / "finite_headline_inference.csv")
    for _, r in h[h.stat == "msr"].iterrows():
        claim(f"msr_diff_{r.comparison}_N{r.N}", f"{r.estimate:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]",
              "analysis/finite_headline_inference.csv", "MSR x1e3 difference, cluster bootstrap 95%")
    s = pd.read_csv(A / "finite_summary_by_init.csv")
    for _, r in s[(s.kind == "ref") & s.init.isin(["uniform", "npmle"])].iterrows():
        claim(f"msr_{r.procedure}_N{r.N}", round(r.MSR_x1e3, 3), "analysis/finite_summary_by_init.csv", "MSR x1e3")
    v = json.loads((OUT / "validation" / "validation_report.json").read_text())
    check("validation_V1", v["V1"]["all_report_exactly_K_updates"], "EM55 = exactly 55 updates")
    check("validation_V2", v["V2"]["date_mismatches"] == 0, f"V2 date mismatches {v['V2']['date_mismatches']}")
    def v3_ok(r):
        if "npmle" in r["source"] and "main_benchmark" in r["source"]:
            return r["raw_date_mismatches"] == 0          # 31 scored-date differences = documented convention
        if r["source"].startswith("matched_info_pairs"):
            return r["date_mismatches_among_valid"] == 0 and r["sim_seed_mismatches"] == 0
        return r["date_mismatches"] == 0
    check("validation_V3", all(v3_ok(r) for r in v["V3"]),
          "archived EM30/55/80, NPMLE (raw date) and reconstruction-v1 decisions reproduce exactly")
    check("validation_V4", v["V4_finite"]["trajectories_with_violations"] == 0, "no monotonicity violations")


# ------------------------------------------------------------------ remaining REPORT.md numbers
def report_misc():
    h1 = pd.read_csv(A / "population_H1_by_init.csv")
    x = h1[(h1.tau == 2) & (h1.G == 200) & (h1.variant == "continuous")]
    for col, lab in (("mean_regret_ref_k0", "init"), ("mean_regret_ref_k55", "EM55"), ("mean_regret_ref_cap", "endpoint")):
        claim(f"pop_tau2G200_{lab}_range", f"{x[col].min():.4f}..{x[col].max():.4f}", "analysis/population_H1_by_init.csv",
              f"mean regret under P across 8 inits ({lab}), tau=2 G=200 continuous")
    for (tau, G), g in h1[h1.variant == "continuous"].groupby(["tau", "G"]):
        claim(f"pop_tau{tau}G{G}_configs_endpoint_worse_by_init", f"{g.configs_cap_worse_than_55.min()}..{g.configs_cap_worse_than_55.max()}",
              "analysis/population_H1_by_init.csv", "configs (of 30) where endpoint regret > EM55 regret, per init")
        claim(f"pop_tau{tau}G{G}_endpoint_regret_range", f"{g.mean_regret_ref_cap.min():.4f}..{g.mean_regret_ref_cap.max():.4f}",
              "analysis/population_H1_by_init.csv", "endpoint mean regret across inits")
    npm = pd.read_csv(A / "population_npmle.csv")
    for _, r in npm[(npm.variant == "continuous") & (npm.init == "npmle")].iterrows():
        claim(f"pop_npmle_tau{r.tau}G{r.G}", round(r.regret_ref, 4), "analysis/population_npmle.csv",
              "NPMLE regret under P (delta 1e-8), mean over 30 configs")
    sc = pd.read_csv(A / "population_status_counts.csv")
    for _, r in sc[sc.variant == "continuous"].iterrows():
        claim(f"pop_certified_tau{r.tau}G{r.G}", int(r.get("certified", 0)), "analysis/population_status_counts.csv",
              "certified of 40 at the cap")
    gm = pd.read_csv(A / "population_grid_mismatch.csv")
    g2 = gm[(gm.tau == 2) & (gm.variant == "continuous")]
    claim("pop_tau2_max_kl_grid_optimum", float(f"{g2.kl_to_target.max():.2g}"), "analysis/population_grid_mismatch.csv",
          "max KL(target || NPMLE fit), tau=2, all grids")
    mz = json.loads((A / "maximizers_summary.json").read_text())
    for k in ("population", "finite"):
        for f in ("rows", "median_kkt_em", "median_kkt_npmle_tightest", "share_npmle_tightest_certified",
                  "share_npmle_date_changes_1e-8_to_tightest", "median_max_abs_cell_diff_em_vs_npmle",
                  "max_max_abs_cell_diff_em_vs_npmle", "rev_max_feas_resid", "rev_max_abs_ll_minus_em"):
            claim(f"maximizers_{k}_{f}", mz[k][f], "analysis/maximizers_summary.json", f)
    for k, fn in (("population", "maximizers_population.csv.gz"), ("finite", "maximizers_finite.csv.gz")):
        m = pd.read_csv(A / fn)
        c = m[m.kkt_em <= 1e-8]
        dis = c.date_em != c.date_npmle
        claim(f"witness_{k}_certified_rows", len(c), f"analysis/{fn}", "rows with endpoint KKT <= 1e-8")
        claim(f"witness_{k}_disagreements", int(dis.sum()), f"analysis/{fn}", "certified endpoint date != NPMLE date")
        claim(f"witness_{k}_rev_selects_npmle_date", int((dis & (c.date_rev == c.date_npmle)).sum()), f"analysis/{fn}",
              "reverse witness (endpoint's fitted cells, L1-closest to NPMLE) selects NPMLE date")
        check(f"witness_{k}_all", int((dis & (c.date_rev == c.date_npmle)).sum()) == int(dis.sum()),
              f"{int((dis & (c.date_rev == c.date_npmle)).sum())}/{int(dis.sum())}")
    pd_ = pd.read_csv(OUT / "population" / "tau2_G200_continuous_decisions.csv.gz")
    st = pd.read_csv(OUT / "population" / "tau2_G200_continuous_status.csv")
    cap = pd_.iteration.max()
    e = pd_[pd_.iteration == cap].merge(st[["distribution", "init", "status"]], on=["distribution", "init"])
    e = e[e.status == "certified"]
    n = int((e.groupby(["distribution", "T", "beta"]).date.nunique() > 1).sum())
    claim("pop_certified_endpoints_init_dependent_dates", n, "population/tau2_G200_continuous_*",
          "configs (of 30) where certified endpoints from different inits select different dates")
    path = pd.read_csv(A / "finite_path_by_config.csv.gz")
    path = path[path.kind == "ref"]
    gp = path.groupby(["N", "init", "iteration"]).msr_ref.mean().reset_index()
    for N in (1000, 100000):
        am = gp[gp.N == N].loc[lambda d: d.groupby("init").msr_ref.idxmin()]
        claim(f"path_argmin_iter_range_N{N}", f"{am.iteration.min()}..{am.iteration.max()}", "analysis/finite_path_by_config.csv.gz",
              "iteration minimizing pooled MSR, range across inits")
    ch = pd.read_csv(A / "finite_path_changes.csv.gz")
    ch = ch[ch.kind == "ref"]
    for N, g in ch.groupby("N"):
        claim(f"share_change_after55_N{N}", round((g.n_changes_after_55 > 0).mean(), 3), "analysis/finite_path_changes.csv.gz",
              "share of trajectories (all inits) changing date after k=55")
    fam = pd.read_csv(A / "finite_by_family.csv")
    f55 = fam[(fam.procedure == "EM55") & (fam.init == "uniform")]
    for N, g in f55.groupby("N"):
        b32 = g[g.distribution == "beta_1_32"].diff_x1e3.iloc[0]
        claim(f"family_beta132_share_N{N}", round(b32 / g.diff_x1e3.sum(), 3), "analysis/finite_by_family.csv",
              "Beta(1,32) share of the sum of per-family EM55-NPMLE MSR differences")
    rev = pd.read_csv(A / "finite_reversal_counts.csv")
    for _, r in rev[(rev.procedure == "EM55")].iterrows():
        claim(f"reversal_EM55_{r.init}_N{r.N}", f"{r.configs_em_better}/{r.configs_tie}/{r.configs_em_worse}",
              "analysis/finite_reversal_counts.csv", "configs EM55 better / tie / worse than NPMLE")
    gr = pd.read_csv(A / "grid_refinement_finite.csv")
    for _, r in gr[gr.procedure.isin(["EM55", "EM_cap", "NPMLE"])].iterrows():
        claim(f"grid_{r.procedure}_G{r.G}_N{r.N}", round(r.MSR_x1e3, 4), "analysis/grid_refinement_finite.csv", "MSR x1e3")
    s = pd.read_csv(A / "finite_summary_by_init.csv")
    for _, r in s[(s.kind == "ref") & (s.procedure == "EM_CV")].iterrows():
        claim(f"cv_median_iter_{r.init}_N{r.N}", r.median_cv_iteration, "analysis/finite_summary_by_init.csv", "CV median k")


# ------------------------------------------------------------------ report tokens
REPORT_TOKENS = {
    "MECHANISM_REPORT.md": ["**660**", "**110**", "470 / 609 / 660 (77.2%)", "77 / 110 / 110 (70.0%)", "**89.3%**",
                            "77.4%", "84.8%", "8 / 660 / 660 (1.2%)", "2 / 110 / 110 (1.8%)", "59.5 -> 5.1",
                            "17.9 -> 16.8", "**-27.0%**", "94.4%", "20.9%", "28 of 30", "24 of 30", "30 of 30",
                            "+5.3% to +16.8%", "+3.0% to +12.6%", "76.7% / 90.7%", "33 cases: 32 improve",
                            "Median 1,725", "0.0092 (0.0019, 0.0702)", "1,042 of 1,500", "1,366 of 1,500"],
    "REPORT.md": ["660", "110", "77.2%", "70.0%", "0.0004-0.0019", "89.3%"],
    "suggested_latex.tex": ["470/609 (77.2\\%)", "77/110 (70.0\\%)", "8/660 (1.2\\%)", "2/110 (1.8\\%)",
                            "7/128 (5.5\\%)", "3/75 (4.0\\%)", "8/136 (5.9\\%)", "10/96 (10.4\\%)",
                            "148/484 (30.6\\%)", "24/110 (21.8\\%)", "133/206 (64.6\\%)", "28/54 (51.9\\%)",
                            "89.3\\%", "84.8\\%", "77.4\\%", "1{,}042", "1{,}366", "660 of", "110 of",
                            "470 of the 609", "51 cases are infeasible", "77 of 110", "32 of these",
                            "28 of 30", "$+5.3\\%$", "$+16.8\\%$", "$+3.0\\%$", "$+12.6\\%$",
                            "98.8\\%", "98.2\\%", "76.7\\%", "90.7\\%", "$0.086$", "$0.022$",
                            "94.4\\%", "20.9\\%"],
}


# Phrases that must NOT appear (overstatements corrected in review).
FORBIDDEN = {
    "suggested_latex.tex": ["preserves the earlier date", "preserve the earlier date", "no such deterioration",
                            "helps because", "overestimation is not", "costs little", "more accurately",
                            "converged EM", "every distribution with"],
    "MECHANISM_REPORT.md": ["costs little", "Any distribution with the late", "useful, not accurate"],
    "REPORT.md": ["costs little", "converged EM", "Any distribution with the late"],
}


def tokens():
    for f, bad in FORBIDDEN.items():
        flat = " ".join((OUT / f).read_text().split())
        for b in bad:
            check(f"forbidden_{f}_{b}", b not in flat, f"'{b}' absent from {f}")
    for f, toks in REPORT_TOKENS.items():
        txt = (OUT / f).read_text()
        flat = " ".join(txt.split())
        for tk in toks:
            ok = " ".join(tk.split()) in flat
            check(f"token_{f}_{tk.strip()}", ok, f"'{tk.strip()}' present in {f}")


def md_table(df, fmt="{:.4f}"):
    """Markdown table without optional dependencies."""
    def cell(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else fmt.format(v)
        return str(v).replace("|", "/")
    rows = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    rows += ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


# Saved outputs this script reads, with the number of files expected.  The
# finite-sample and population tables ship in the EM data archive
# (wbtm-supplement-2026.10-em-data.zip); the rest in the main archive.
REQUIRED = {
    "finite/cells/ref_*_table.csv.gz": 60,
    "finite/cells/*.done.json": 120,
    "population/tau2_G200_continuous_status.csv": 1,
    "population/tau2_G200_continuous_decisions.csv.gz": 1,
    "mechanism/events.csv.gz": 1,
    "mechanism/constraint_tests.csv.gz": 1,
    "mechanism/posthoc_moment_tests.csv.gz": 1,
    "mechanism/population_control.csv.gz": 1,
    "analysis/finite_summary_by_init.csv": 1,
    "analysis/population_H1_by_init.csv": 1,
    "analysis/maximizers_summary.json": 1,
    "validation/validation_report.json": 1,
    "frozen/pairs_tau2.json": 1,
}


def missing_inputs() -> list[str]:
    out = []
    for pat, n in REQUIRED.items():
        k = len(glob.glob(str(OUT / pat)))
        if k != n:
            out.append(f"results/em_mechanism/{pat}: found {k}, expected {n}")
    return out


def main():
    miss = missing_inputs()
    if miss:
        print("EM STUDY OUTPUTS MISSING -- cannot verify the EM claims:\n  " + "\n  ".join(miss) +
              f"\nResults root: {OUT.parent.parent} (set WBTM_ROOT to a source checkout or extracted release)."
              "\nIn a release, extract wbtm-supplement-2026.10-em-data.zip into the same directory as "
              "the main archive (see experiments/em_mechanism/README.md, 'Data').", file=sys.stderr)
        sys.exit(2)
    V.mkdir(parents=True, exist_ok=True)
    w, ev = events()
    acc, data = interventions(ev)
    red = regret_reduction(data, ev)
    m3_by_init()
    descriptives(ev)
    asymmetry()
    controls()
    headline()
    report_misc()
    tokens()
    cl = pd.DataFrame(CLAIMS)
    ck = pd.DataFrame(CHECKS)
    write_csv_atomic(cl, V / "claims.csv")
    write_csv_atomic(ck, V / "checks.csv")
    lines = ["# Claim verification (generated by `python -m experiments.em_mechanism.verify_claims`)", "",
             f"Checks: {int(ck['pass'].sum())}/{len(ck)} pass.", "", "## Checks", "",
             "| id | pass | detail |", "|---|---|---|"]
    lines += [f"| {r.id} | {'PASS' if r['pass'] else '**FAIL**'} | {r.detail} |" for _, r in ck.iterrows()]
    lines += ["", "## Intervention accounting (attempted / feasible / successful / excluded)", "",
              "Success = keeps EM55's date (T1-T4, T1r, Tno3, Ttrue3) or moves to the endpoint's date "
              "(T5, Tm1-Tm3).  Shares are of feasible events; `share_of_attempted` counts infeasible as failures.",
              "", md_table(acc), "",
              "## Regret-reduction definitions", "", md_table(red), "",
              "## All claims", "", md_table(cl)]
    (V / "claims_check.md").write_text("\n".join(lines) + "\n")
    print(f"{len(cl)} claims; checks {int(ck['pass'].sum())}/{len(ck)} pass")
    for _, r in ck[~ck["pass"]].iterrows():
        print("FAIL", r.id, r.detail)
    sys.exit(0 if ck["pass"].all() else 1)


if __name__ == "__main__":
    main()
