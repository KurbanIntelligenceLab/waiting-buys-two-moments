# What does early EM preserve? Mechanism follow-up

**Conclusion.**  On the reference populations at tau = 2, early stopping
retains an upward bias in the estimated third moment m3 = E[p^3].  Continued
fitting often lowers that estimate and shifts decisions toward costly earlier
intervention.  Moment interventions identify this change as a major channel of
the deterioration from EM55 to the endpoint at the iteration cap.  A biased
estimate can improve a decision because the observed welfare loss is
asymmetric in the m3 error.  Underestimating m3 is associated with much larger
regret than moderate overestimation (up to +50%).  Regret rises again for
larger overestimates, so not every overestimate is harmless.  This concerns
the direction of the m3 error.  The study does not compare the overall
accuracy of the EM55 and endpoint estimates of the risk distribution.

Qualifications (section 6): the single-moment interventions are **post hoc**;
this is a population-dependent explanation for this benchmark, not a universal
benefit of early stopping; holding m3 also blocks many improvements;
distribution shape still matters; and the nonuniqueness witnesses of
`REPORT.md` section 3.2 remain valid but do not explain the EM55-to-endpoint
deterioration.

Every number below is recomputed from saved outputs by
`python -m experiments.em_mechanism.verify_claims`
(`verification/claims.csv`, `verification/claims_check.md`).  Protocol:
`experiments/em_mechanism/MECHANISM_PROTOCOL.md`, frozen before any mechanism
quantity was computed.  Tests T1-T5 are prespecified; tests marked * were added
after the prespecified results were seen.  No new data: the inputs are the
completed trajectories of the main study.

## Terms

| term | meaning |
|---|---|
| EM55 | EM after exactly 55 updates from uniform grid weights |
| endpoint | EM after 1e6 updates, the iteration cap.  Certification is reported separately: at the cap, 1,042 of 1,500 uniform-start reference trajectories at N = 1e3 and 1,366 of 1,500 at N = 1e5 meet the KKT certificate (residual <= 1e-8). |
| m_j | raw moment E[p^j] of the fitted grid distribution; at tau = 2, m1-m3 are identified, m4 and above are not |
| regret | relative regret under the reference population P, 1 - W^d / W^{t*} |
| worse / improve event | a dataset whose regret under P is higher / lower at the endpoint than at EM55 (difference > 1e-12) |
| conditional result | computed over worse (or improve) events only, not over all datasets |

## 1. Sample

Reference datasets, uniform start, tau = 2, 200-point grid: 1,500 datasets per
N (30 configurations x 50).

| N | worse | improve | same regret | all |
|---|---|---|---|---|
| 1e3 | **660** | 32 | 808 | 1,500 |
| 1e5 | **110** | 150 | 1,240 | 1,500 |

Every worse and improve event changes the selected date.  Among worse events,
the endpoint's date is earlier than EM55's in 92.4% (N = 1e3) and 92.7%
(N = 1e5).  The most common moves at N = 1e3 are 2->1 (191), 4->1 (172) and
3->1 (123).

**Switch point.**  The first iteration after 55 at which the date changes is
found exactly by re-running EM from the saved EM55 weights.  It was found for
every worse event, and the date at the start of each recorded bracket equals
EM55's in all of them.  Median 1,725 (IQR 214-13,630) at N = 1e3; 14,663
(2,181-98,695) at N = 1e5.  One EM step at the switch changes every moment by
less than 1e-4 relative, so the switch is a threshold crossing by a contrast
that drifts continuously.

## 2. What changes between EM55 and the endpoint (worse events, medians)

| quantity | N = 1e3 | N = 1e5 |
|---|---|---|
| relative change in m1 / m2 | -0.35% / -4.1% | -0.56% / -6.9% |
| relative change in **m3** | **-27.0%** | **-26.7%** |
| relative change in m4 / m6 / m10 | -52% / -85% / -99.2% | -52% / -86% / -99.5% |
| effective support (grid points) | 59.5 -> 5.1 | 17.9 -> 16.8 |
| points holding 99% of mass | 90.5 -> 7 | 30 -> 22 |
| max change in a fitted cell probability | 2.5e-3 | 5.1e-4 |
| share with larger KL(true cells \|\| fitted cells) at the endpoint | 94.4% | 20.9% |

At tau = 2 the pre-failure cells span exactly the polynomials of degree <= 3,
so the fitted cell probabilities are a one-to-one function of (m1, m2, m3).
At N = 1e3 the endpoint fits the empirical frequencies more closely and the
true probabilities less closely, the distribution atomizes, and the higher
moments collapse.  At N = 1e5 the moment changes are similar, but the
estimate does not atomize, and in most events it moves toward the true cell
probabilities.  At N = 1e5 the change is therefore not mainly noise fitting.

## 3. How the change moves the welfare contrast

`C = [W^{d_end} - W^{d_55}] / W*_P`, computed from estimated welfare.  Mean
change over worse events: +0.110 (N = 1e3) and +0.071 (N = 1e5).  A path
integral of the envelope gradient along the straight segment EM55 -> endpoint
reproduces the exact change to within 1.7e-4.

* **By risk region**, the change is spread over all six bins.  At N = 1e3 the
  largest contributions are [.1,.2) (+0.036) and [.05,.1) (+0.021); mass leaves
  [.05,.2) (-0.15) and moves to [0,.02) (+0.07) and [.2,.4) (+0.10).
* **Observable-driven versus within-fit.**  We split the change at one
  projection: the L1-closest distribution to EM55 that has the endpoint's
  fitted cell probabilities (the T1 projection below).  Moving from EM55 to
  this projection carries +0.1102 of the change (N = 1e3); moving from the
  projection to the endpoint carries -0.0002.  At N = 1e5: +0.0723 and
  -0.0009.  This is one decomposition along one projection, not a statement
  about every distribution with those fitted probabilities.

## 4. Constraint tests

Each test solves one linear program: minimize the L1 distance to a stated
target distribution, over nonnegative grid weights, subject to stated equality
constraints.  The selected date of the solution is then recorded.

* *Attempted*: all worse or improve events.
* *Feasible*: the LP has a solution.
* *Excluded*: infeasible events, which have no constrained estimate.
* *Success*: the constrained estimate keeps EM55's date (keep tests) or moves
  to the endpoint's date (move tests).

Shares are of feasible events; the accounting table also gives shares of
attempted events, counting infeasible ones as failures.  Mean regret under P
is over feasible events, with EM55 and endpoint regret over the **same**
feasible events.  Full table: `mechanism/intervention_accounting.csv`.

**Worse events (conditional on deterioration)**

| test | target; constraints | success | N = 1e3: succ / feas / att | N = 1e5: succ / feas / att | N = 1e3 mean regret: test (EM55, endpoint) |
|---|---|---|---|---|---|
| T1 | closest to EM55; endpoint's fitted cells | keep | 8 / 660 / 660 (1.2%) | 2 / 110 / 110 (1.8%) | 0.0735 (0.0020, 0.0747) |
| T2 | closest to endpoint; endpoint's cells + m4..m6 of EM55 | keep | 1 / 2 / 660 | 0 / 0 / 110 | - |
| T2' | same with m4..m8 | keep | 0 / 0 / 660 | 0 / 0 / 110 | - |
| T3 | closest to endpoint; endpoint's cells + EM55 masses in [.05,.1), [.1,.2) | keep | 7 / 128 / 660 (5.5%) | 3 / 75 / 110 (4.0%) | 0.0335 (0.0011, 0.0378) |
| T4 (negative control) | closest to endpoint; endpoint's cells + EM55 masses in [0,.02), [.4,1] | keep | 8 / 136 / 660 (5.9%) | 10 / 96 / 110 (10.4%) | 0.0362 (0.0012, 0.0386) |
| T5 | closest to EM55; EM55's cells + endpoint masses in the T3 bins | move | 21 / 569 / 660 (3.7%) | 3 / 110 / 110 (2.7%) | 0.0033 (0.0019, 0.0673) |
| T1r* | closest to endpoint; EM55's m1, m2, m3 | keep | 288 / 660 / 660 (43.6%) | 49 / 110 / 110 (44.5%) | 0.0323 (0.0020, 0.0747) |
| Tm1* | closest to EM55; m1 of endpoint, m2, m3 of EM55 | move | 4 / 660 / 660 (0.6%) | 0 / 110 / 110 (0%) | - |
| Tm2* | closest to EM55; m2 of endpoint, m1, m3 of EM55 | move | 45 / 603 / 660 (7.5%) | 28 / 110 / 110 (25.5%) | - |
| Tm3* | closest to EM55; m3 of endpoint, m1, m2 of EM55 | move | 133 / 206 / 660 (64.6%) | 28 / 54 / 110 (51.9%) | - |
| **Tno3*** | closest to EM55; **m1, m2 of endpoint, m3 of EM55** | keep | **470 / 609 / 660 (77.2%)** | **77 / 110 / 110 (70.0%)** | **0.0092 (0.0019, 0.0702)** |
| Ttrue3* | closest to endpoint; m1, m2 of endpoint, true m3 | keep | 148 / 484 / 660 (30.6%) | 24 / 110 / 110 (21.8%) | 0.0455 (0.0014, 0.0655) |

Bin selection for T3/T4 used estimated-welfare attributions only (largest and
smallest mean |attribution| over worse events, per N); the selected bins were
the same at both N.

**Regret reduction from holding m3 (Tno3), the "about 89%" figure.**

| definition | N = 1e3 | N = 1e5 |
|---|---|---|
| ratio of means over feasible worse events: (mean r_end - mean r_test) / (mean r_end - mean r_55) | **89.3%** (609 of 660; 51 infeasible excluded) | 84.8% (110 of 110) |
| same over all worse events, infeasible counted as no intervention (regret stays at the endpoint) | 77.4% | 84.8% |
| mean of per-event ratios, feasible events | 89.7% | 79.2% |

The 89% figure is therefore conditional on deterioration events **and** on
feasibility of the intervention.  With infeasible events counted as
unrepaired it is 77%.

**Improve events (control)**

| test | N = 1e3: succ / feas / att | N = 1e5: succ / feas / att |
|---|---|---|
| T1 (keep) | 2 / 32 / 32 (6.3%) | 5 / 150 / 150 (3.3%) |
| Tno3* (keep EM55's date, i.e. block the improvement) | 23 / 30 / 32 (76.7%) | 136 / 150 / 150 (90.7%) |

**Reading.**

* **The prespecified hypotheses fail.**  Freezing unidentified moments (T2) is
  almost never feasible under the endpoint's fit.  Freezing risk-bin masses
  (T3) is often infeasible and, where feasible, no better than the negative
  control (T4).  Importing the bin masses into EM55 (T5) rarely reproduces the
  endpoint's date.  By the protocol's criteria, a risk-bin or
  unidentified-moment mechanism is not supported.
* **What T1 shows.**  For one particular distribution, the L1-closest to EM55
  among those with the endpoint's fitted cell probabilities, the endpoint's
  date is selected in 98.8% (N = 1e3) and 98.2% (N = 1e5) of worse events.
  The change in fitted probabilities is enough to move the date even with
  minimal change elsewhere.  T1 does not show that *every* distribution with
  those probabilities selects the endpoint's date.  The nonuniqueness
  witnesses of `REPORT.md` section 3.2 show that some such distributions
  select different dates.
* **Within the fitted moments, m3 is the main lever (post hoc).**  Holding m3
  at EM55's value while taking the endpoint's m1 and m2 keeps EM55's date in
  77.2% (N = 1e3) and 70.0% (N = 1e5) of feasible worse events.  Changing m1
  alone almost never moves the date.  Changing m2 alone rarely does (7.5% /
  25.5%).  Changing m3 alone often does where feasible (64.6% / 51.9%), but
  that change is infeasible for 454 of 660 events at N = 1e3.
* **m3 is a lever, not a protective rule.**  Holding m3 also blocks 76.7% /
  90.7% of the improvements.  It is the coordinate through which the date
  responds to further fitting, in either direction.

## 5. Why an upward bias in m3 helps on these populations

* **Early EM overestimates m3 for every initialization.**  The median relative
  error at EM55 across the eight starts is +5.3% to +16.8% at N = 1e3 and
  +3.0% to +12.6% at N = 1e5.  That includes the "low" start, which begins
  80% below the truth (median -0.803 at iteration 0).  The endpoint
  (+1.6% to +1.8% / 0.0%) and the NPMLE (+1.5% / 0.0%) are approximately
  median-unbiased (`mechanism/m3_error_by_init.csv`).
* **The loss is asymmetric in the m3 error** (Figure 5a).  This is
  associational: EM55, endpoint and NPMLE fits of all initializations are
  pooled.  At N = 1e3, mean regret is 0.086 when m3 is underestimated by more
  than 50%, 0.035 for 25-50%, and 0.0084-0.0091 for errors between -10% and
  +50%.  At N = 1e5 it is 0.048 for 25-50% under against 0.0024 for 25-50%
  over.  Underestimates mostly produce dates earlier than the optimum (56%
  and 77% of decisions in the two largest underestimation bins at N = 1e3).
* **Who underestimates.**  Share of datasets with m3 below the truth: EM55
  36.5% / 12.7%, endpoint 48.4% / 49.9%, NPMLE 48.7% / 49.9% (N = 1e3 / 1e5).
  Share of decisions earlier than the optimum at N = 1e3: EM55 14.8%,
  endpoint 47.9%, NPMLE 54.9%.
* **The direction of the error matters, not only its size.**  Replacing the
  endpoint's m3 by the *true* m3 (Ttrue3) keeps EM55's date in only 30.6% /
  21.8% of feasible worse events.  EM55's advantage is consistent with the
  direction of its m3 error, which avoids the costly side of the loss on these
  populations.  No claim is made about overall estimation accuracy.

## 6. Controls and limits

* **Noise-free control** (exact probabilities, tau = 2, 200-point grid; all 40
  trajectories certified at the 1e7 cap).  Across 30 configurations x 8
  starts, the date changes between EM55 and the endpoint in 33 cases: 32
  improve and 1 worsens.  In the improving cases m3 also falls (median
  -15.4%), but toward the truth, and the estimate does not atomize (effective
  support 58.4 -> 57.4).
* **Compatible-population reversal.**  The extreme pair members match P's
  m1-m3 exactly and differ in the tail: the upper member has the larger m4 in
  30 of 30 pairs.  The upper member's earliest optimal date is earlier than
  P's in 28 of 30 pairs.  The lower member's equals P's in 24 of 30, and is
  equal or later in 30 of 30 (`mechanism/pair_member_optimal_dates.csv`).
  EM55's lean toward later dates suits P and the lower member and costs it
  under the upper member.  Mean regret under the upper member on pair
  datasets: EM55 0.0343 vs NPMLE 0.0142 (N = 1e3) and 0.0325 vs 0.0242
  (N = 1e5).
* **Population dependence.**  This explains the benchmark's reference
  families at tau = 2; it is not a general property of early stopping.  A
  population whose optimum is earlier, like the upper members, is hurt by the
  same bias.
* **Post hoc.**  The m1/m2/m3 interventions were designed after T1 showed that
  the fitted cells carry the change.
* **Shape matters.**  The endpoint's shape combined with EM55's fitted moments
  (T1r) still keeps EM55's date in only about 44% of worse events, so m3 is a
  major channel, not the only one.
* **Nonuniqueness witnesses.**  The witnesses in `REPORT.md` section 3.2
  remain valid: different weight vectors with the same fitted probabilities
  can select different dates.  They do not establish that drift within an
  unchanged fit explains the EM55-to-endpoint deterioration.  Here that
  within-fit part of the contrast change is near zero (section 3).
* **Scope.**  Events come from the 30 reference configurations at tau = 2,
  uniform start.  The m3-bias table covers all eight starts; the interventions
  use the uniform start only.

## Reproduce

All from saved outputs (no EM study is re-run):

```bash
export OMP_NUM_THREADS=1
python -m experiments.em_mechanism.mechanism --workers 4          # ~13 min; re-runs EM only from saved EM55 weights to find switch points
python -m experiments.em_mechanism.mechanism_posthoc --workers 4  # post-hoc moment interventions (~2 min)
python -m experiments.em_mechanism.mechanism_tables               # m3 bias, asymmetric loss, pair-member dates
python -m experiments.em_mechanism.verify_claims                  # recomputes and checks every number above
python -m experiments.em_mechanism.mechanism_figure               # Figure 5
```
