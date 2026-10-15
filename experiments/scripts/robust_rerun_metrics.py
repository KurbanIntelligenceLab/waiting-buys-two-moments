"""Score the recomputed robust baselines against the published ones.

Every row is computed on ONE sample: the 7,189 runs of the 7,200-run grid in
which the identified-set construction is feasible. The 11 infeasible runs
(empty estimated identified set) are the same runs for the envelope rule, the
achieved-welfare LP and maximin, and they are excluded rather than imputed.

Published decisions come from results/main_benchmark_raw_all.csv (and
results/r15_maximin_welfare_raw.csv for maximin). Corrected decisions come
from the monotone-combination rerun, results/robust_rerun_*_raw.csv. For
worst_case_lp only the 801 runs whose test fired were rerun; the monotone
construction (see experiments/scripts/robust_rerun.py) implies no other run
can change, so the remaining runs carry their published decision.

Decisions are scored against the true population welfare trajectory, which is
an evaluation reference and is not an input to any rule.
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


def metrics(g: pd.DataFrame) -> dict:
    return {
        "n": len(g),
        "p10": np.percentile(g.welfare_efficiency, 10),
        "median": g.welfare_efficiency.median(),
        "mean": g.welfare_efficiency.mean(),
        "msr": g.squared_regret.mean(),
        "exact_pct": 100.0 * g.exact_argmax_match.mean(),
    }


def main() -> None:
    bench = pd.read_csv(RESULTS / "main_benchmark_raw_all.csv")
    mm_pub = pd.read_csv(RESULTS / "r15_maximin_welfare_raw.csv")
    wc_new = pd.read_csv(RESULTS / "robust_rerun_worst_case_lp_raw.csv")
    mm_new = pd.read_csv(RESULTS / "robust_rerun_maximin_raw.csv")

    lp_pub = bench[bench.rule == "worst_case_lp"]
    valid = lp_pub[lp_pub.success][KEY]
    assert len(valid) == 7189, len(valid)

    # the infeasible runs are the same ones for every identified-set rule
    inf_lp = set(map(tuple, lp_pub[~lp_pub.success][KEY].values))
    for name, df in [("envelope", bench[bench.rule == "worst_case_envelope"]),
                     ("maximin published", mm_pub),
                     ("maximin rerun", mm_new)]:
        assert set(map(tuple, df[~df.success][KEY].values)) == inf_lp, name

    rows = []

    def add(label, g, note=""):
        rows.append({"row": label, **metrics(g), "note": note})

    add("act immediately", bench[bench.rule == "act_immediately"].merge(valid, on=KEY))
    add("worst-case envelope, published",
        bench[(bench.rule == "worst_case_envelope") & bench.success])

    add("achieved-welfare LP, published", lp_pub[lp_pub.success])
    # corrected: published rows, overwritten on the runs that were rerun
    rerun_ok = wc_new[wc_new.success]
    lp_corr = lp_pub[lp_pub.success].merge(
        rerun_ok[KEY + ["welfare_efficiency", "squared_regret",
                        "exact_argmax_match", "date_combined"]],
        on=KEY, how="left", suffixes=("", "_new"))
    hit = lp_corr.welfare_efficiency_new.notna()
    assert int(hit.sum()) == len(rerun_ok) == 801, (int(hit.sum()), len(rerun_ok))
    for c in ["welfare_efficiency", "squared_regret", "exact_argmax_match"]:
        lp_corr.loc[hit, c] = lp_corr.loc[hit, c + "_new"]
    lp_corr["exact_argmax_match"] = lp_corr.exact_argmax_match.astype(float)
    add("achieved-welfare LP, corrected", lp_corr)

    add("maximin welfare, published", mm_pub[mm_pub.success])
    add("maximin welfare, corrected", mm_new[mm_new.success])

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "robust_rules_recomputed.csv", index=False)

    # decision-change accounting, reported against both denominators
    fired = int((lp_pub[lp_pub.success].selected_date > 1).sum())
    lp_changed = int(rerun_ok.changed.sum())
    mm_ok = mm_new[mm_new.success]
    mm_changed = int(mm_ok.changed.sum())
    later = int((mm_ok[mm_ok.changed].date_combined
                 > mm_ok[mm_ok.changed].date_default).sum())
    earlier_lp = int((rerun_ok[rerun_ok.changed].date_combined
                      < rerun_ok[rerun_ok.changed].date_default).sum())
    env = bench[(bench.rule == "worst_case_envelope") & bench.success]

    print(out.to_string(index=False))
    print()
    print(f"LP: fired {fired}/7189, rerun {len(rerun_ok)}, changed {lp_changed} "
          f"({100*lp_changed/7189:.2f}% of valid runs, "
          f"{100*lp_changed/fired:.1f}% of firing runs); "
          f"{earlier_lp}/{lp_changed} move earlier")
    print(f"maximin: changed {mm_changed}/{len(mm_ok)} "
          f"({100*mm_changed/len(mm_ok):.2f}%); {later}/{mm_changed} move later")
    print(f"widening held: LP {int(rerun_ok.widening_held.sum())}/{len(rerun_ok)}, "
          f"maximin {int(mm_ok.widening_held.sum())}/{len(mm_ok)}")
    print(f"envelope selected date 1 in {int((env.selected_date == 1).sum())}"
          f"/{len(env)} valid runs")
    print("maximin published dates:",
          mm_pub[mm_pub.success].selected_date.value_counts().sort_index().tolist())
    print("maximin corrected dates:",
          mm_ok.date_combined.value_counts().sort_index().tolist())


if __name__ == "__main__":
    main()


def corrected_frames():
    """Return (lp_published, lp_corrected, mm_published, mm_corrected) frames
    over the 7,189 feasible runs, with a common set of scoring columns."""
    bench = pd.read_csv(RESULTS / "main_benchmark_raw_all.csv")
    mm_pub = pd.read_csv(RESULTS / "r15_maximin_welfare_raw.csv")
    wc_new = pd.read_csv(RESULTS / "robust_rerun_worst_case_lp_raw.csv")
    mm_new = pd.read_csv(RESULTS / "robust_rerun_maximin_raw.csv")
    cols = KEY + ["welfare_efficiency", "squared_regret", "exact_argmax_match"]

    lp_pub = bench[(bench.rule == "worst_case_lp") & bench.success]
    rerun_ok = wc_new[wc_new.success]
    lp_corr = lp_pub.merge(rerun_ok[cols + ["date_combined"]], on=KEY,
                           how="left", suffixes=("", "_new"))
    hit = lp_corr.welfare_efficiency_new.notna()
    for c in cols[4:] + ["welfare_efficiency"]:
        lp_corr.loc[hit, c] = lp_corr.loc[hit, c + "_new"]
    lp_corr["exact_argmax_match"] = lp_corr.exact_argmax_match.astype(float)
    return (lp_pub[cols], lp_corr[cols],
            mm_pub[mm_pub.success][cols], mm_new[mm_new.success][cols])
