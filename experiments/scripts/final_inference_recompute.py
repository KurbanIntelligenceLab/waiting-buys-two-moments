"""Final RC-vs-certified-NPMLE paired inference, complete-case analysis, and
date-error metrics, recomputed cleanly from the current (certified NPMLE + v3)
main_benchmark_raw_all.csv.

Written as a one-off audit-verification script (not part of the main pipeline).
Source data: results/main_benchmark_raw_all.csv (57,598 rows, HEAD b97a236).

Sign convention (explicit, never flipped internally):
    every reported difference is literal RC - NPMLE.
    For p10/median/exact_acc/... : positive = RC better.
    For MSR: negative = RC better (lower mean squared regret).

Bootstrap: cluster bootstrap by config_id (all seeds carried together per the
frozen configuration), matching experiments/scripts/paired_inference.py's
documented resampling unit. N_BOOT=2000 replicates, 95% CI (ALPHA=0.05).
Holm axis: across the 6 rule-comparisons (COMPETITORS) within each (N,t) cell,
per metric -- matching the paper's documented "Holm correction across rule
comparisons within each combination of population size and prefix length."
Only the regret_calibrated-vs-npmle row of that corrected family is reported
here, but the correction itself is computed over the full competitor set so
the reported Holm p-values are the real, paper-protocol-consistent ones, not
a vacuous single-comparison correction.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.scripts.table4 import apply_failure_convention  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"
ALPHA = 0.05
N_BOOT = 2000
SEED = 20260906
REFERENCE = "regret_calibrated"
COMPETITORS = ["npmle", "nnls", "parametric_beta", "worst_case_lp",
               "worst_case_envelope", "act_immediately"]
METRICS = ["p10", "median", "msr", "exact_acc", "signed_err", "abs_err"]


def _stat(a, b, metric):
    """a, b: dict of arrays for rule_a (RC) and rule_b (competitor).
    Returns literal a - b for every metric, no sign flip, ever."""
    if metric == "p10":
        return float(np.quantile(a["we"], 0.10) - np.quantile(b["we"], 0.10))
    if metric == "median":
        return float(np.median(a["we"]) - np.median(b["we"]))
    if metric == "msr":
        return float(np.mean(a["reg"] ** 2) - np.mean(b["reg"] ** 2))
    if metric == "exact_acc":
        return float(np.mean(a["exact"]) - np.mean(b["exact"]))
    if metric == "signed_err":
        return float(np.mean(a["err"]) - np.mean(b["err"]))
    if metric == "abs_err":
        return float(np.mean(np.abs(a["err"])) - np.mean(np.abs(b["err"])))
    raise ValueError(metric)


def compare(df, rule_a, rule_b, rng):
    piv = {}
    for col, name in [("welfare_efficiency", "we"), ("relative_regret", "reg"),
                       ("exact_argmax_match", "exact")]:
        piv[name] = df.pivot_table(index=["config_id", "seed_index"],
                                    columns="rule", values=col)
    df["date_err"] = df["scored_date"] - df["true_optimal_date"]
    piv["err"] = df.pivot_table(index=["config_id", "seed_index"],
                                 columns="rule", values="date_err")

    keep = piv["we"][[rule_a, rule_b]].dropna().index
    for k in piv:
        piv[k] = piv[k].loc[keep]
    cfgs = piv["we"].index.get_level_values(0).to_numpy()
    uniq = np.unique(cfgs)
    by_cfg = {c: np.where(cfgs == c)[0] for c in uniq}

    A = {k: piv[k][rule_a].to_numpy() for k in piv}
    B = {k: piv[k][rule_b].to_numpy() for k in piv}

    out = {"rule_A": rule_a, "rule_B": rule_b, "n_runs": len(A["we"]),
           "n_configs": len(uniq), "bootstrap_replicates": N_BOOT,
           "resampling_unit": "configuration (cluster, all seeds carried)"}

    boots = {m: np.empty(N_BOOT) for m in METRICS}
    for bi in range(N_BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_cfg[c] for c in pick])
        a_bi = {k: A[k][idx] for k in A}
        b_bi = {k: B[k][idx] for k in B}
        for m in METRICS:
            boots[m][bi] = _stat(a_bi, b_bi, m)

    for m in METRICS:
        obs = _stat(A, B, m)
        bs = boots[m]
        lo, hi = np.quantile(bs, [ALPHA / 2, 1 - ALPHA / 2])
        p = 2.0 * min((bs <= 0).mean(), (bs >= 0).mean())
        out[f"{m}_A"] = float(np.median(A["we"])) if False else None  # placeholder, filled below
        out[f"{m}_diff"] = obs
        out[f"{m}_ci_low"] = float(lo)
        out[f"{m}_ci_high"] = float(hi)
        out[f"{m}_p_raw"] = float(min(1.0, max(p, 1.0 / N_BOOT)))
    # also store raw point values for reporting
    out["RC_p10"], out["B_p10"] = float(np.quantile(A["we"], .10)), float(np.quantile(B["we"], .10))
    out["RC_median"], out["B_median"] = float(np.median(A["we"])), float(np.median(B["we"]))
    out["RC_msr"], out["B_msr"] = float(np.mean(A["reg"] ** 2)), float(np.mean(B["reg"] ** 2))
    out["RC_exact_acc"], out["B_exact_acc"] = float(np.mean(A["exact"])), float(np.mean(B["exact"]))
    out["RC_signed_err"], out["B_signed_err"] = float(np.mean(A["err"])), float(np.mean(B["err"]))
    out["RC_abs_err"], out["B_abs_err"] = float(np.mean(np.abs(A["err"]))), float(np.mean(np.abs(B["err"])))
    return out


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def main():
    raw = pd.read_csv(RESULTS / "main_benchmark_raw_all.csv", low_memory=False)
    d = apply_failure_convention(raw)
    rng = np.random.default_rng(SEED)

    rows = []
    for (N, t), fam in d.groupby(["N", "prefix_t"]):
        fam_rows = []
        for comp in COMPETITORS:
            if comp not in fam.rule.unique():
                continue
            r = compare(fam.copy(), REFERENCE, comp, rng)
            r.update({"N": int(N), "prefix_t": int(t)})
            fam_rows.append(r)
        for m in METRICS:
            adj = holm([r[f"{m}_p_raw"] for r in fam_rows])
            for r, a in zip(fam_rows, adj):
                r[f"{m}_p_holm"] = float(a)
                r[f"{m}_significant"] = bool(a < ALPHA)
        rows.extend(fam_rows)

    out = pd.DataFrame(rows)
    npmle_only = out[out.rule_B == "npmle"].sort_values(["N", "prefix_t"]).reset_index(drop=True)
    npmle_only.to_csv(RESULTS / "final_rc_vs_npmle_inference.csv", index=False)

    # markdown report
    lines = []
    lines.append("# Final RC vs certified-NPMLE paired inference (recomputed)\n")
    lines.append(f"Source: `results/main_benchmark_raw_all.csv` ({len(raw)} rows). "
                 f"Script: `experiments/scripts/final_inference_recompute.py`.\n")
    lines.append("Bootstrap unit: **configuration** (cluster, all seeds carried together). "
                 f"Replicates: **{N_BOOT}**. CI level: **95%**. "
                 "Holm axis: across the 6 rule-comparisons (RC vs each of "
                 f"{COMPETITORS}) within each (N,t) cell, per metric -- the family "
                 "the paper's Inference paragraph describes. Sign convention: "
                 "every diff column is literal RC - NPMLE, never flipped "
                 "(negative on MSR = RC better; positive elsewhere = RC better).\n")
    lines.append("| N | t | metric | RC | NPMLE | RC-NPMLE | 95% CI | raw p | Holm p | sig |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in npmle_only.iterrows():
        for m, label in [("p10", "p10"), ("median", "median"), ("msr", "MSR"),
                          ("exact_acc", "exact-date acc"), ("signed_err", "signed date err"),
                          ("abs_err", "abs date err")]:
            lines.append(f"| {int(r.N)} | {int(r.prefix_t)} | {label} | "
                          f"{r[f'RC_{m}']:.6f} | {r[f'B_{m}']:.6f} | "
                          f"{r[f'{m}_diff']:.6f} | "
                          f"[{r[f'{m}_ci_low']:.6f}, {r[f'{m}_ci_high']:.6f}] | "
                          f"{r[f'{m}_p_raw']:.4f} | {r[f'{m}_p_holm']:.4f} | "
                          f"{'YES' if r[f'{m}_significant'] else 'no'} |")
    (RESULTS / "final_rc_vs_npmle_inference.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", RESULTS / "final_rc_vs_npmle_inference.csv")
    print("wrote", RESULTS / "final_rc_vs_npmle_inference.md")


if __name__ == "__main__":
    main()
