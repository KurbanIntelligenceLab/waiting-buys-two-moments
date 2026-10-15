"""Rebuild main_benchmark_raw_all.csv with the Window-A NPMLE and RC rows.

`main_benchmark_raw_all.csv` is the paper-facing per-run table that the paired
inference, complete-case and date-error analyses all read. No script has ever
written it -- it was assembled historically from incremental slices -- so the
R15 correction cannot propagate to those analyses without an explicit rebuild.

What this does:

  * the six window-independent rules (oracle, act_immediately, parametric_beta,
    nnls, worst_case_envelope, worst_case_lp) are carried through untouched;
    they are computed from cohort counts or are definitional, and the
    observation-window correction does not reach them
  * the two window-dependent rules are replaced wholesale:
        npmle             <- bagged_full_b200_raw.csv, rule "single"
        regret_calibrated <- bagged_full_b200_raw.csv, rule "rc"
    topped up with results/r15_recovered_skipped_runs.csv, the 53 runs the
    b200 script's `n_support > n_atoms` guard rejected and which were
    recomputed judging validity by the KKT certificate instead

Columns present in the main table but not in the b200 output are reconstructed
rather than dropped: `stakes` is a population quantity carried over from the
matching config, and `success`/`failure_reason` are set from whether the run
produced a date.

The pre-correction file is preserved at results/window_b/ and at tag
window-b-frozen. See results/r15_window_correction_report.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"

KEY = ["cell_id", "N", "prefix_t", "seed_index"]
WINDOW_DEPENDENT = {"npmle": "single", "regret_calibrated": "rc"}


def main() -> None:
    main_tbl = pd.read_csv(RESULTS / "main_benchmark_raw_all.csv", low_memory=False)
    b200 = pd.read_csv(RESULTS / "bagged_full_b200_raw.csv")
    b200 = b200[b200.SKIP.isna()]
    recovered = pd.read_csv(RESULTS / "r15_recovered_skipped_runs.csv")

    carried = main_tbl[~main_tbl.rule.isin(WINDOW_DEPENDENT)].copy()
    print(f"carrying through {len(carried)} rows across "
          f"{carried.rule.nunique()} window-independent rules")

    # `stakes` and the config descriptors are per (cell_id, N, prefix_t); take
    # them from any surviving row of that cell rather than recomputing.
    ref_cols = ["cell_id", "N", "prefix_t", "config_id", "distribution",
                "gamma", "T", "beta", "stakes"]
    ref = (main_tbl[ref_cols].drop_duplicates(subset=["cell_id", "N", "prefix_t"])
           .set_index(["cell_id", "N", "prefix_t"]))

    out_cols = list(main_tbl.columns)
    rebuilt = [carried]

    for target_rule, source_rule in WINDOW_DEPENDENT.items():
        src = pd.concat(
            [b200[b200.rule == source_rule],
             recovered[recovered.rule == source_rule]],
            ignore_index=True)
        src = src.drop_duplicates(subset=KEY)

        rows = src.set_index(["cell_id", "N", "prefix_t"]).join(
            ref[["config_id", "distribution", "gamma", "T", "beta", "stakes"]],
            rsuffix="_ref").reset_index()
        for c in ("config_id", "distribution", "gamma", "T", "beta"):
            if f"{c}_ref" in rows:
                rows[c] = rows[c].fillna(rows[f"{c}_ref"])

        rows["rule"] = target_rule
        rows["success"] = rows.selected_date.notna() & (rows.selected_date > 0)
        rows["failure_reason"] = ""
        for c in out_cols:
            if c not in rows:
                rows[c] = np.nan
        rebuilt.append(rows[out_cols])
        print(f"  {target_rule:18s} <- {source_rule:8s}  {len(rows)} runs "
              f"({len(src)} unique keys)")

    out = pd.concat(rebuilt, ignore_index=True)
    out.to_csv(RESULTS / "main_benchmark_raw_all.csv", index=False)
    print(f"\nwrote {RESULTS / 'main_benchmark_raw_all.csv'} ({len(out)} rows)")
    print(out.rule.value_counts().to_string())


if __name__ == "__main__":
    main()
