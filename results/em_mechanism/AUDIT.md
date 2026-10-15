# EM-mechanism study: implementation and manuscript audit

Kept separate from the findings report on purpose.  Nothing here changes any
existing table, label, result file, code path or manuscript claim; every item
is an observation for the authors' decision, with its exact code path, the
archived provenance it was checked against, and a reproducible example
(`python -m experiments.em_mechanism.audit_examples` prints A2-A5).

## Verified as reproduced (no discrepancy)

| Check | Result | Evidence |
|---|---|---|
| "EM 55 iterations" | exactly 55 EM updates; iteration 0 = uniform weights 1/G | `wtm/npmle.py::fit_npmle` evaluates the likelihood before each update; with `tol=1e-300` its stop test never fires: 3,600/3,600 fits report `iterations == K` (V1) |
| New batch engine vs `fit_npmle` | max relative weight difference 4.0e-15; 0/3,600 date mismatches (K = 30, 55, 80) | V2, `validation/validation_report.json` |
| Archived EM30/55/80, tau = 2 | 0 date mismatches on 1,200 datasets each; log-likelihood within 2.5e-10 | `results/early_stopped_robustness_raw.csv`, `early_stopped_npmle_raw.csv` (V3) |
| Archived certified NPMLE, tau = 2 | 0 mismatches on 1,169 datasets; the other 31 differ only by scoring convention (A1) | `results/main_benchmark_raw_all.csv`, rule `npmle` |
| reconstruction-v1 (D.3) | identical simulation seeds; 0 date mismatches for EM30/55/80 and certified NPMLE; identical validity | `results/matched_info_pairs_raw.csv`, seeds 1001-1005 |
| Weight floor 1e-300 (this study only) | no recorded diagnostic changes in 846 trajectories, bitwise; floor binds in 134 | `validation/floor_check.json` |

## A1. NPMLE validity convention differs between the main benchmark and D.3 (documentation)

* **Code paths.**  D.3 reconstruction (`experiments/scripts/matched_info_pairs.py`,
  `npmle()`) and the bagging generator (`bagged_full_b200.py`) treat a fit with
  `n_support > #cells` as invalid.  The main benchmark table was rebuilt by
  `experiments/scripts/r15_rebuild_main_benchmark.py`, which re-admits the 53
  runs that guard rejected "judging validity by the KKT certificate instead"
  (`results/r15_recovered_skipped_runs.csv`).
* **Observation.**  On the 1,200 archived tau = 2 datasets, `fit_npmle_certified`
  returns 31 support-violating fits; the archive scores all 31 at the fit's raw
  argmax date (31/31 identical to ours).  Table 3 therefore applies a different
  validity rule from D.3 (Table 6), where such fits are excluded by joint
  validity.  The manuscript describes neither difference.
* **This study** scores those fits at their raw date (the main-benchmark
  convention) and reports the immediate-allocation fallback as a sensitivity
  (`analysis/finite_npmle_invalid_sensitivity.csv`).

## A2. "Certified" NPMLE fits are accepted without meeting the 1e-8 certificate (manuscript wording)

* **Manuscript.**  `paper/main.tex` l.1286-1288: "Certified NPMLE checks the
  directional-derivative optimality condition on that grid with tolerance 1e-8."
* **Code.**  `wtm/npmle.py::fit_npmle_certified` returns the certificate; the
  callers (`rerun_certified.py` ll.100-110, `matched_info_pairs.py::npmle`)
  accept any fit with finite likelihood and Lindsay-bounded support and record
  the residual only as a flag (`kkt_flagged`).
* **Observation.**  The check is performed but not enforced: on the 1,200
  archived tau = 2 benchmark datasets, 450 scored NPMLE fits (37.5%) have
  residual > 1e-8 and 3 have residual > 1e-6 (median 6.6e-9).
* **Example (A2).**  Beta(1,32), T = 20, beta = 0.0005, N = 1e5, seed 1015
  (sim_seed 4096100428): residual 1.53e-6, support 5, scored (date 4).

## A3. Code comment: "the optimum is unique and the starting point cannot change where EM converges" is false for the maximizer (code documentation)

* **Code.**  `wtm/npmle.py::fit_npmle` docstring, ll.47-52 (used to justify warm
  starts).  The sibling docstring of `fit_npmle_fast` states the correct fact:
  `Psi` has low rank, the likelihood is concave but not strictly concave in `w`,
  and its maximizers form a polytope.
* **Observation.**  The optimal *value* and the optimal fitted cell
  probabilities are unique (strict concavity in `Psi w`); the optimal *weights*
  are not, and neither is the selected date.  On exact probabilities (tau = 2,
  G = 200), EM endpoints certified to KKT <= 2e-15 from different
  initializations select different dates in 5 of 30 configurations.
* **Example (A3).**  Beta(1,32), T = 20, beta = 0.01, exact probabilities, 1e7
  updates: EM[uniform] and EM[rand1] both have ll = -0.362321065244892 and
  KKT = 1.55e-15, fitted cells equal to 1.1e-16, weight TV 0.34, and select
  dates 5 and 1.
* **Impact on reported results.**  The manuscript states that neither reported
  procedure uses a warm start, so no reported number depends on the comment.

## A4. Support-reduction solver: tighter tolerances are not reached, and the date can move with the tolerance (numerical caveat)

* **Code.**  `fit_npmle_certified` (SLSQP inner solve with `ftol=1e-14`; the
  outer loop stops when the most-violating grid point is already active).
* **Observation.**  On the exact-probability targets (14 tasks, 3,360
  comparison rows) no NPMLE solve requested at 1e-13 is certified (median
  achieved residual 1.6e-8), while EM run to its cap reaches a median residual
  of 3.1e-15.  Changing the requested tolerance from 1e-8 to 1e-13 changes the
  NPMLE's selected date in 6.7% of rows.
* **Example (A4).**  Same target as A3: requested 1e-8 -> residual 3.5e-10;
  requested 1e-11 or 1e-13 -> residual 1.77e-8 (worse).
* **Consequence.**  "Certified NPMLE" picks one atomic member of a non-singleton
  maximizer set; which member is picked depends on solver path and tolerance.

## A5. Witness construction (supporting A3)

For every exact-probability comparison where certified EM and the NPMLE select
different dates (601 rows with EM residual <= 1e-8), the LP witness of
`maximizers.py` finds weights reproducing EM's fitted cell probabilities to
<= 1.1e-16 (hence the same objective, to 4.4e-16, and the same KKT residual)
that select the NPMLE's date in 601 of 601 rows.  Example (A5): the A3 target,
witness residual 2.1e-17, ll = -0.362321065244892, KKT = 1.33e-15, date 1
(EM: date 5).

## Already documented elsewhere (not re-audited)

* Manuscript says NPMLE/EM/bagging observe period-tau failures; the code and
  archive that produced Tables 3-4 use the pre-failure window
  (`results/audit/tables_window_comparison.md`).  This study uses the code's
  window.
* Appendix D.3 Tables 6-7 are unrecovered; reconstruction-v1 is separate
  (`results/audit/d3_provenance_search.md`).
