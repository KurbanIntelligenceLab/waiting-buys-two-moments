# Compute inventory: recomputing the paper's total core-hours

Written 2026-09-03, in response to the abstract's `\TODO{recompute final CPU
core-hours}`. Every number below is either read directly from a `runtime_s` (or
equivalent) column already on disk, or -- where no such column was recorded --
reconstructed from the launch-commit and completion-commit/log timestamps at
the known worker count (10). The two kinds are kept in separate totals and
never blended silently.

## Strictly measured (per-row `runtime_s` / `secs_*`, summed once)

| Core-h | File | Experiment |
| ---: | --- | --- |
| 99.17 | `main_benchmark_raw_all.csv` | Exp 9/10/16/27/28/29/30 -- the 6 non-recomputed rules' final numbers, plus the now-superseded EM-fit npmle/regret rows |
| 72.83 | `identification_welfare_bounds.csv` | Exp 4, 16,200 rows, G=201/501/1001/2001 -- matches the figure already stated in the draft |
| 6.21 | `reachable_optimal_dates.csv` | Exp 5/6 |
| 0.06 | `lower_bound_summary.csv` | Exp 8 frontier |
| 0.40 | `spec_test_null.csv` | Exp 21 |
| 1.66 | `spec_test_drift.csv` | Exp 22 |
| 2.51 | `spec_test_group_correlation.csv` | Exp 23 |
| 4.86 | `npmle_ablation_raw.csv` | Exp 17 |
| 192.37 | `regret_rule_ablation_raw.csv` | Exp 18, E/lambda sweep -- pre-certification EM, about v1 |
| 0.34 | `worst_case_lp_pilot.csv` | D8 pilot |
| 2.53 | `worst_case_lp_pilot_confidence.csv` | D8d |
| 9.07 | `worst_case_lp_pilot_expanded.csv` | D8 grid refinement |
| 20.88 | `nels88_semisynthetic_raw.csv` | Exp 24 |
| 21.85 | `nels88_shirali_budget_raw.csv` | Exp 24 diagnostic arm |
| 20.24 | `els2002_semisynthetic_raw.csv` | Exp 25 |
| 23.11 | `els2002_shirali_budget_raw.csv` | Exp 25 diagnostic arm |
| 0.02 | `certified_npmle_study.csv` | 187-run stratified pre-check |
| 292.24 | `results/window_b/certified_benchmark_raw.csv` | Certified rerun: NPMLE plug-in, v1, v2, 360 cells. Window B; archived 2026-09-18, not regenerated because nothing reads it |
| 17.46 | `warm_gate_raw.csv` | Warm-start solver equivalence gate, 20 cells (rejected) |
| 376.87 | `v3_full_benchmark_raw.csv` | v3 full 30-configuration benchmark (dev + holdout) |
| **1164.69** | | **TOTAL, strictly measured** |

## Wall-clock-reconstructed (no per-row runtime was stored)

Derived from launch-commit timestamp to completion-commit/log timestamp,
multiplied by the fixed worker count (10). Labelled separately because it is
not a per-row measurement.

| Core-h (approx.) | Job | Note |
| ---: | --- | --- |
| ~111.0 | `regret_v2` full sweep (v1+v2, EM-era) | 22:42 -> 09:48 next day, 11.1h wall; superseded by the certified rerun |
| ~5.2 | `regret_v2_instrument.py` | 1048-run disagreement diagnostic, EM-era |
| ~4.3 | `em_regularization_path.py` | 720-run EM path-tracing study |
| ~32.5 | `misspecification.py` Exp 19+20 chain | `runtime_s` column is a vestigial placeholder fixed at 0 after the worst-case rules were dropped; wall-clock used instead |
| **~153.0** | | **TOTAL, wall-clock-reconstructed** |

## Grand total

**1164.69 + 153.0 ~= 1317.7 core-hours**, at most 10 parallel workers, no GPU, no
paid API calls.

## Explicitly excluded (superseded or duplicate representations)

Excluding these is what keeps the total from double-counting, not a judgement
that the compute wasn't real:

- `main_benchmark_raw_N1000_100000_t2.csv`, `_t2_3_4.csv`, `_t3_4.csv`,
  `N5000_20000_t2_3_4.csv` -- incremental slices superseded by
  `main_benchmark_raw_all.csv`.
- `identification_welfare_bounds_G1001_backup.csv` (14.45 core-h) -- the
  preserved LP-infeasible run from the Beta(1,32) t=4 incident, kept only as a
  documentation record; the draft's own "72.8 core-hours / 16,200 rows" already
  excludes it.
- `dev_v3_raw.csv` + `holdout_v3_raw.csv` -- merged into, and numerically
  identical in total to, `v3_full_benchmark_raw.csv`.
- `npmle_grid_ablation.csv`, `npmle_tolerance_ablation.csv`,
  `regret_rule_ablation.csv` -- summary tables computed from raw files already
  counted.

## Treated as negligible

No new NPMLE fits or LP solves; seconds of arithmetic, not summed:
`v3_bootstrap_ablation.csv` (reused saved bootstrap contrasts), `embodied_pilot.csv`,
`subsample_instability*.csv`, `current_subsample_sensitivity.csv`,
`certified_gate.csv`.

## The judgement call this leaves open

`regret_rule_ablation_raw.csv` (192.4 core-h) and the `regret_v2` full sweep
(~111 core-h) are genuine completed compute, but both concern the pre-v3
regret-calibrated rule and are now superseded by the certified v3 benchmark.
If the final paper drops the v1/v2 ablation and diagnostic material entirely
(rather than keeping it as an appendix record of the design process), the
defensible total shrinks by roughly 300 core-hours, to **~1017 core-h measured
/ ~1170 core-h including reconstructed wall-clock**. This is a decision about
what the paper keeps, not a compute-accounting question, so it is left open
here rather than resolved unilaterally.

---

## Revision 2026-09-04: "experiments reported in this paper"

The abstract's wording changed from "the completed experiments" to "the
experiments reported in this paper" -- this resolves the judgement call left
open above in the direction of excluding superseded pre-certification EM-era
work, since its numbers do not appear in any final table.

**Excluded, newly, because superseded and not reported:**

| Core-h | Item | Why excluded |
| ---: | --- | --- |
| 192.37 | `regret_rule_ablation_raw.csv` (Exp 18) | v1's E/lambda ablation; v1 is superseded by v3, which has its own (near-zero-cost) bootstrap-size ablation |
| 51.66 | `main_benchmark_raw_all.csv`, npmle + regret_calibrated rows only | EM-era fits; Table 4's NPMLE and regret-calibrated numbers now come from the certified rerun. The other 6 rules' rows in this file (47.51 core-h) are unaffected and still counted, since those rules were never recomputed. |
| 111.0 | `regret_v2` full sweep (wall-clock) | EM-era v1+v2 evaluation on the full E9 design; superseded by `results/window_b/certified_benchmark_raw.csv` (Window B, archived) |
| 5.2 | `regret_v2_instrument.py` (wall-clock) | EM-era disagreement diagnostic; the early/late-stopping mechanism it found is now independently confirmed on certified data (computed directly from `v3_full_benchmark_raw.csv`, already counted there) |

**Retained:** `em_regularization_path.py` (~4.3 core-h) -- unlike the v1/v2-specific
items above, this motivates *why the certified solver correction was necessary
at all* (the frozen-EM-as-implicit-regularizer finding), which is a claim about
the numerical method rather than about the superseded rule, and is likely to be
referenced in the paper's discussion of the solver.

## Total under "experiments reported in this paper"

    920.65 core-h  strictly measured
     36.83 core-h  wall-clock-reconstructed (em_regularization_path + misspecification Exp19/20)
    -------
    957.48 core-h  TOTAL

Rounds to **~957 core-hours**.
