# NELS:88 provenance audit (Experiment 24)

Audit run 2026-08-30, before any rule was evaluated.

## 1. Source paper

Ali Shirali, Ariel Procaccia, Rediet Abebe, *The Hidden Cost of Waiting for
Accurate Predictions*, arXiv:2503.00650 (ICLR 2025 / EAAMO).

Dataset, quoted: **NELS:88**, "In this data, failure corresponds to student
dropout and is recorded." Cohort tracked at four time points.

Calibration, quoted verbatim:

> Let m0 and m1 be the proportion of the initial pool of individuals who failed
> right before the first and second steps, respectively. Assuming there has been
> no intervention at the first step, it follows from the central limit theorem
> that m0 -> a/(a+b) and m1 -> a(a+1)/[(a+b)(a+b+1)] at a fast rate of
> O(1/sqrt(N)). By solving for a and b, we can accurately estimate the initial
> distribution as a Beta distribution.

Result, quoted: **"Our estimation gives alpha = 0.028 and beta = 0.35 for the
case of NELS data."** The paper states the implied dropout rate "aligns with
NELS-provided estimates at approximately 7%".

## 2. Repository audit

`https://github.com/alishiraliGit/hidden-costs-of-waiting-for-accurate-predictions`

Contents: MATLAB only --
`find_optimal_over_time_alloc.m`, `main_one_time.m`, `main_over_time.m`,
`posterior_at_k_t.m`, `simulate_traj.m`, `simulate_untreated_traj.m`,
`value_at_k_t.m`, plus `figs/`.

**There is no `data/` directory, no NELS file of any kind, and no script that
performs the NELS calibration.** The repository is pure simulation. The only
reproducible artefact of the NELS analysis anywhere in the published record is
the parameter pair (0.028, 0.35).

## 3. Answers to the required provenance questions

| Question | Answer |
| --- | --- |
| Raw NELS data included? | **No** |
| Raw records or fitted parameters used downstream? | Only the fitted `(alpha, beta)`; the repo never reads data |
| Exact `m0, m1` values | **Not published.** They are recoverable *by inversion* from the published pair: `m0 = 0.074074`, `m1 = 0.055260` (verified below) |
| Interpretation of "failed right before first/second steps" | **Ambiguous in the source** -- see section 4 |
| Weights used? | Not stated |
| Cohort filtering | Not stated |
| Missing-data handling | Not stated |
| Horizon `T` and budget in their NELS experiment | **Not stated.** Their figures use budget ratios B/N of 5%, 10%, 20%, but no `T` is given for the NELS run |

## 4. An unresolved ambiguity in the source, recorded not resolved

The stated limits are `m0 -> E[p]` and `m1 -> E[p^2]`. The verbal description
of `m1` -- "the proportion of the initial pool who failed right before the
second step" -- does not obviously equal `E[p^2]` under an absorbing-failure
reading, where that proportion would be `E[p(1-p)] = E[p] - E[p^2]`. Either the
verbal gloss or the reading is loose in the source.

This does **not** affect what we do here: Level B takes the fitted distribution
`Beta(0.028, 0.35)` as given and never re-derives it from `m0, m1`. The
ambiguity is recorded because it would matter to anyone attempting Level A.

## 5. Calibration validation (independent, ours)

Solving the paper's own moment equations in closed form,
`s = (m0 - m1)/(m1 - m0^2)`, `alpha = m0 * s`, `beta = s - alpha`:

| Check | Value |
| --- | --- |
| `E[p] = m0` implied by (0.028, 0.35) | **0.074074** |
| `E[p^2] = m1` implied | **0.055260** |
| Paper's stated dropout rate | ~7% -- **agrees** |
| Round-trip: solve (m0, m1) back for (alpha, beta) | **(0.028000, 0.350000)** -- exact |

The published calibration is internally consistent and we reproduce their
inversion exactly.

## 6. Level determination: **Level B**

Level A (raw-data reproduction) requires the NELS:88 public-use file. NCES
distributes it only through the **interactive Online Codebook** or **DataLab
(PowerStats/QuickStats)**; there is no flat-file download. The probed EDAT zip
endpoints that work for ELS:2002 return 404 for NELS. Per the standing
instruction, we stop rather than substitute a different dataset.

**To attempt Level A, the following must be obtained manually:**

> NELS:88/2000 **public-use** student file (base year 1988 through fourth
> follow-up 2000), exported from the NCES Online Codebook at
> <https://nces.ed.gov/surveys/nels88/data_products.asp>, containing the
> longitudinal dropout/enrolment status variables for the base-year cohort.
> Any of SPSS/SAS/Stata/ASCII export is fine.

Level A is *not required* for this experiment to be valid, because the audit
established that the published analysis itself only ever uses the fitted pair.

**Level B is therefore what we run, and it is labelled as such throughout:
"published NELS calibration", never "raw-data reanalysis".**

## 7. Consequence discovered during validation (reported, not hidden)

Under our frozen budget grid, `Beta(0.028, 0.35)` produces very low stakes, and
at any budget ratio at or above 1% the optimal date is `t* = 1`:

| budget ratio | T=12: t*, stakes | T=20: t*, stakes |
| --- | --- | --- |
| 0.0005 | 3, 0.0111 | 3, 0.0127 |
| 0.002 | 2, 0.0071 | 2, 0.0090 |
| 0.01 | 1, 0.0000 | 1, 0.0000 |
| 0.02 / 0.05 / 0.10 / 0.20 | 1, 0.0000 | 1, 0.0000 |

The last row covers **Shirali et al.'s own budget ratios (5%, 10%, 20%)**. On
their calibration at their budgets, immediate allocation is exactly optimal and
the timing question is vacuous. Running those arms would measure a constant, so
they are not run; the frozen small-budget grid is the only regime in which this
calibration has nonzero stakes at all, and even there the maximum is 1.3%.

This is recorded before rule evaluation so that it cannot be mistaken for a
post-hoc explanation of whatever the rules do.
