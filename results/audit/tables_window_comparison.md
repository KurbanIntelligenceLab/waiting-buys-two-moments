# Tables 3–5 under the two recovered observation windows (2026-10-03)

**Question.** The manuscript (Section 6.2, Table 3 notes, Appendix A.2) says
that NPMLE, early-stopped EM, bagging and regret calibration use exit records
*including* period-τ failures ("Window B"). Does any recovered implementation
with that window reproduce the printed Tables 3–5?

**Answer: no.** The printed Tables 3 and 4 equal, cell for cell, the outputs of
the implementation that uses the pre-failure prefix ("Window A"). The Window-B
implementation gives different numbers. Table 5 does not depend on the window
and matches under either.

The manuscript's stated window therefore contradicts the implementation that
produced its tables. Nothing was changed to reconcile them: no code, seeds,
exclusions, rounding or manuscript text.

## Both historical versions are kept

| Version | Code | Outputs | Window |
|---|---|---|---|
| A (current) | this tree | `results/main_benchmark_raw_all.csv`, `bagged_full_b200_raw.csv`, `early_stopped_npmle_raw.csv`, `early_stopped_robustness_raw.csv` | pre-failure prefix |
| B (pre-correction) | commit `197fd9e9` | `results/window_b/` (same file names; each byte-identical to `197fd9e9:results/…`) | includes period-τ failures |

CMS (`cms_table2_raw.csv`) and the Table 5 LP rules (`robust_rerun_*_raw.csv`)
use cohort moments and exist only once. The CMS run postdates the correction.

## Method

`python experiments/audit/tables_window_comparison.py`

1. **Verifier on both versions.** The manuscript verifier
   (`release/verify_paper.py`) is run unchanged twice: on `results/`, and on
   a mirror in which the four window-dependent inputs are replaced by their
   `results/window_b/` versions. Each Table 3–5 claim is compared with the
   manuscript text. Output: `tables_window_comparison.{json,csv}`.
2. **Every cell, by the verifier's own formulas.** For both versions the
   script computes every Table 3 and Table 4 cell, without the verifier's
   structural checks, which stop Window B early (below). The formulas are the
   verifier's own: matched sample of runs where every procedure has a
   decision, MSR, gain over NPMLE, p10. Output: `tables_window_cells.csv`.

## Result

| Table | Claims | Match the manuscript under A | Under B |
|---|---|---|---|
| Table 3 (with §5.3, §7.2 text) | 13 | 13 | 0 |
| Table 4 (with D.1 text) | 8 | 8 | 0 |
| Table 5 | 8 | 8 | 8 (window-independent) |

Under B the verifier also stops on structure:
- *Table 3:* one NPMLE run is missing; it is Window B's single
  invalid primary fit, the one Appendix C.2 reports.
- *Table 4:* the exclusions are not exactly the Beta failures.

### Selected cells (MSR ×10⁻³, p10)

| Procedure | N | Manuscript | Window A | Window B |
|---|---|---|---|---|
| Matched runs | 10³ / 10⁵ | 589 / 580 | 589 / 580 | 600 / 599 |
| Certified NPMLE | 10³ | 4.61, 0.8672 | 4.61, 0.8672 | 4.74, 0.8683 |
| Certified NPMLE | 10⁵ | 1.60, 0.9658 | 1.60, 0.9658 | 0.38, 0.9921 |
| Bagged NPMLE | 10⁵ | 0.59, 0.9878 | 0.59, 0.9878 | 0.18, 0.9952 |
| Regret-calibrated | 10³ | 3.95, 0.8681 | 3.95, 0.8681 | 3.71, 0.8683 |
| CMS | 10⁵ | 0.35, 0.9921 | 0.35, 0.9921 | 0.34, 0.9921 |
| Early-stopped EM (55) | 10³ | 0.21, 0.9964 | 0.21, 0.9964 | 0.10, 0.9964 |
| Table 4 NPMLE | 10⁵ | 1.55, 0.9876 | 1.55, 0.9876 | 0.38, 0.9921 |

The manuscript column is what the manuscript prints. The verifier confirms
every printed Table 3–4 cell under A; `tables_window_cells.csv` has all cells
for both versions.

Under B, some qualitative statements in Section 7.2 would also change. At
N = 10⁵, CMS's gain over NPMLE falls from +1.24 to +0.03, and CMS no longer
outperforms bagging and regret calibration (MSR 0.34 vs 0.18 and 0.18). The
ordering "bagging below regret calibration" holds under both versions at
N = 10³ (A: 3.49 vs 3.95; B: 3.62 vs 3.71).

## Remaining contradiction

The manuscript describes the Window-B information set for the likelihood
methods. Its tables were computed with Window A. Its Appendix C.2 counts come
from the Window-B run. No single recovered implementation reproduces both the
description and the numbers. This is disclosed as discrepancies `window` and
`c2_run` (`release/paper_artifacts.yaml`).
