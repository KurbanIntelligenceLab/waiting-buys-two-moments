"""Build Table 4 from a frozen raw benchmark file.

FAILURE-SCORING CONVENTION, frozen 2026-08-28
---------------------------------------------
A run on which a rule returns no estimate is scored as the planner falling back
to acting immediately:

    no estimate  =>  that = 1,  welfare scored normally,  n.e. incremented

Rationale.  The draft's Table 4 caption says failures are "scored rather than
discarded".  Excluding them from the welfare quantiles would contradict that
sentence and, worse, would flatter exactly the rules that fail most: a method
could improve its p10 by failing on its hard cases.  Under this convention every
rule has the same denominator on every metric, and `n.e.` becomes an additional
reliability measure rather than a mechanism for dropping bad runs.

The fallback is `that = 1` because acting immediately is the one date a planner
can always choose with no estimate at all -- it needs no fitted distribution, no
identified set and no data beyond knowing the programme has started.

This changes the present Table 4 only in the fourth decimal, but it is frozen
now, before the supplementary grid, so that every prefix is scored identically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"

RULE_ORDER = ["oracle", "act_immediately", "parametric_beta", "nnls", "npmle",
              "worst_case_envelope", "worst_case_lp", "regret_calibrated"]
RULE_LABEL = {
    "oracle": "Oracle, act at t*",
    "act_immediately": "Act immediately, t=1",
    "parametric_beta": "Parametric, Beta moments",
    "nnls": "Least squares (ablation)",
    "npmle": "Nonparametric MLE",
    "worst_case_envelope": "Worst-case envelope",
    "worst_case_lp": "Worst-case LP",
    "regret_calibrated": "Regret-calibrated (ours)",
}
FALLBACK_DATE = 1          # frozen: no estimate => act immediately


def apply_failure_convention(raw: pd.DataFrame) -> pd.DataFrame:
    """Score no-estimate runs at the fallback date; keep `success` for n.e."""
    d = raw.copy()
    bad = ~d["success"].astype(bool)
    if not bad.any():
        d["scored_date"] = d["selected_date"]
        return d

    # welfare of the fallback date, taken from the act-immediately rows of the
    # same (cell, seed) -- identical data, so this is the true mean-field value
    imm = (d[d.rule == "act_immediately"]
           .set_index(["cell_id", "seed_index"])[
               ["welfare_efficiency", "relative_regret", "squared_regret",
                "true_welfare_at_selected", "exact_argmax_match"]])
    idx = pd.MultiIndex.from_frame(d.loc[bad, ["cell_id", "seed_index"]])
    for col in imm.columns:
        d.loc[bad, col] = imm[col].reindex(idx).to_numpy()
    d["scored_date"] = np.where(bad, FALLBACK_DATE, d["selected_date"])
    return d


def build(raw_path: Path, sizes=None) -> pd.DataFrame:
    raw = pd.read_csv(raw_path)
    d = apply_failure_convention(raw)
    sizes = sizes or sorted(d.N.unique())

    rows = []
    for r in RULE_ORDER:
        row = {"rule": RULE_LABEL.get(r, r), "rule_key": r}
        for N in sizes:
            s = d[(d.rule == r) & (d.N == N)]
            row[f"MSR_N{N}"] = s.squared_regret.mean()
            row[f"med_N{N}"] = s.welfare_efficiency.median()
            row[f"p10_N{N}"] = s.welfare_efficiency.quantile(0.10)
            row[f"min_N{N}"] = s.welfare_efficiency.min()
            row[f"ne_N{N}"] = int((~s["success"].astype(bool)).sum())
            row[f"argmax_N{N}"] = s.exact_argmax_match.mean()
            row[f"n_N{N}"] = len(s)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    # NOTE: the incremental main_benchmark_raw_N*.csv slices this originally
    # assembled from were removed as superseded once the certified NPMLE
    # rerun replaced their npmle/regret_calibrated rows (they remain in git
    # history). The final results/table4*.csv were regenerated once, directly,
    # by combining the 6 untouched rules from main_benchmark_raw_all.csv with
    # npmle_certified and regret_v3 from v3_full_benchmark_raw.csv -- see the
    # commit that overwrote main_benchmark_raw_all.csv with that combination.
    # This main() is kept for reference but will no longer find its inputs.
    raws = sorted(p for p in RESULTS.glob("main_benchmark_raw_N*.csv"))
    if not raws:
        print("no main_benchmark_raw_N*.csv slices found -- these were removed"
              " as superseded; main_benchmark_raw_all.csv and table4*.csv are"
              " already final and should not be regenerated this way. See the"
              " note above main().")
        return 1
    # Each sweep's output file collects the WHOLE raw directory, so later files
    # already contain earlier cells.  Deduplicate on the run identity rather
    # than trusting the file boundaries.
    raw = pd.concat([pd.read_csv(p) for p in raws], ignore_index=True)
    before = len(raw)
    raw = raw.drop_duplicates(subset=["cell_id", "seed_index", "rule"],
                              keep="last").reset_index(drop=True)
    if before != len(raw):
        print(f"deduplicated {before} -> {len(raw)} rows "
              f"({raw.cell_id.nunique()} cells)")
    combined = RESULTS / "main_benchmark_raw_all.csv"
    raw.to_csv(combined, index=False)

    for t in sorted(raw.prefix_t.unique()):
        sub = raw[raw.prefix_t == t]
        tmp = RESULTS / f"_tmp_t{t}.csv"
        sub.to_csv(tmp, index=False)
        tab = build(tmp)
        tmp.unlink()
        out = RESULTS / (f"table4.csv" if t == 2 else f"table4_t{t}.csv")
        tab.to_csv(out, index=False)
        pd.set_option("display.width", 250)
        cols = [c for c in tab.columns if c != "rule_key"
                and not c.startswith(("min_", "argmax_", "n_"))]
        print(f"\n=== prefix t={t} ===")
        print(tab[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
