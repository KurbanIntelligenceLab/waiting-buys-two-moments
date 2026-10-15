# Why does early-stopped EM beat the certified NPMLE? Results of the trajectory study

All numbers below are from **completed** runs (finished 2026-10-06), computed by
the frozen protocol (`experiments/em_mechanism/PROTOCOL.md`, committed before
any main-stage result existed).  Pilot outputs (`pilot_*`) and the interim
validation (`validation/interim/`) are kept only as records and are not
evidence.  The manuscript, existing results and existing code are unchanged;
suspected discrepancies are in the separate `AUDIT.md`.  Every number in this
report and in `MECHANISM_REPORT.md` is recomputed from saved outputs by
`python -m experiments.em_mechanism.verify_claims` (`verification/`).

Terminology: **EM55** is EM after exactly 55 updates from uniform weights; the
**endpoint** is EM after the iteration cap (1e6 updates for finite samples,
1e7 or 1e6 for exact probabilities).  An endpoint is not necessarily
converged: whether it meets the KKT certificate (residual <= 1e-8) is reported
separately wherever endpoints are compared.

## 1. What was run (planned vs completed)

| Stage | Planned | Completed | Content |
|---|---|---|---|
| A. Exact observation probabilities | 14 tasks | **14** | tau = 2 (G = 200, 400, 800), tau = 3, 4 (G = 200, 400); continuous and grid-realizable truth; 5 families x 8 initializations; 560 trajectories to 1e7 (tau = 2, G = 200) or 1e6 updates |
| B. Finite samples, tau = 2, G = 200 | 120 cells | **120** | 30 configurations x N in {1e3, 1e5} x {reference datasets, pair datasets} x 50 datasets = 6,000 datasets; per dataset 8 EM initializations to 1e6 updates, certified NPMLE, 2-fold CV EM: 48,000 + 12,000 trajectories, 6,000 NPMLE fits |
| Grid refinement (finite) | 24 cells | **24** | G = 400, 800; 6 prespecified configurations x 2 N x 20 datasets, uniform start |
| Weight-floor check | 3 cases | **3** | 846 trajectories with and without the floor |
| Validation V1-V7 | all | **all** | `validation/validation_report.json` |
| Maximizer / witness analysis | stages A, B | **both** | `analysis/maximizers_*` |
| Deferred (optional) | regularized-likelihood comparator | not run | deferred as instructed |

Failures: no EM trajectory failed numerically; 0 likelihood decreases beyond
1e-13 in 60,560 trajectories.  Iteration-cap hits are an outcome, reported in
every table (section 3).  Compute: 1.5 s median per EM trajectory to 1e6
updates; ~30 core-hours for stage B.  The container was reclaimed twice while
idle; both times the run resumed from per-cell checkpoints without recomputing
completed cells.

## 2. Faithful reproduction of the original methods

| Check | Result |
|---|---|
| "EM55" | exactly 55 EM updates from uniform weights (iteration 0): 3,600/3,600 `fit_npmle` fits report `iterations == K` |
| New batch engine vs `fit_npmle` | max relative weight difference 4.0e-15; 0/3,600 date mismatches at K = 30, 55, 80 |
| Archived EM30 / EM55 / EM80 (tau = 2, seeds 1001-1020) | 0 date mismatches on 1,200 datasets each; log likelihood within 2.5e-10 |
| Archived certified NPMLE (`main_benchmark_raw_all.csv`) | 0 raw-date mismatches on 1,200 datasets (31 support-violating fits are scored at their raw date, as the archive does; AUDIT A1) |
| reconstruction-v1 (D.3), seeds 1001-1005 | identical simulation seeds and decisions on all 300 datasets for EM30/55/80 and NPMLE |
| Weight floor 1e-300 (only numerical modification) | added after each update to avoid subnormal arithmetic; in 846 trajectories (134 where it binds) every recorded diagnostic is **bitwise identical** with and without it: dates, log likelihood, KKT residual, cell probabilities, moments, first-certified iteration, status; weights differ by <= 7e-290 |
| Batch welfare evaluator | equals `_trajectory` and `PopulationConfig` to 1.4e-15 (same greedy allocation convention for truth and estimate); near-ties re-decided with `_trajectory` |

## 3. Results

### 3.1 Exact probabilities (no sampling noise)

* **Regret improves, then plateaus; it does not deteriorate.**  At tau = 2
  (G = 200) every one of the 40 trajectories is certified (KKT <= 1e-8) by
  1e7 updates.  Mean regret under the reference population falls from
  0.028-0.053 at initialization to 0.0004-0.0019 at EM55 and 0.0000-0.0014 at
  the certified endpoint.  The endpoint is worse than EM55 in 1 of 30 configurations for
  one initialization (rand1); otherwise equal or better.  The same holds on
  G = 400, 800 and at tau = 3, 4, where one configuration per initialization
  is worse at the endpoint.
* **The EM endpoint beats the NPMLE without any noise.**  NPMLE mean regret
  under P: 0.0033 (tau = 2, G = 200), rising to 0.0246 (G = 400) and 0.0271
  (G = 800).  The EM endpoint (certified in 40/40, 38/40 and 38/40
  trajectories) stays at 0.0000-0.0014 (G = 200) and 0.0000-0.0002
  (G = 400, 800) across initializations.  At
  tau = 3, 4 the NPMLE's regret is also small (0.0015 and 0.0000 at G = 200).
* **Grid misspecification is negligible.**  The grid optimum reproduces the
  continuous population's cell probabilities to KL <= 3e-15 at tau = 2; the
  grid-realizable truth gives the same trajectories.
* **Iteration-cap hits.**  Certified by the cap: 40/40 (tau = 2, G = 200),
  38/40 (G = 400, 800), 20/40 (tau = 3), 8/40 (tau = 4, G = 200).

### 3.2 Same likelihood, different decisions (approximate optimality vs demonstrated nonuniqueness)

* The objective is a function of the fitted cell probabilities only, and is
  strictly concave in them; the weights, and the date, are not determined.
* **Demonstrated, exact probabilities.**  EM endpoints certified to
  KKT <= 2e-15 from different initializations select different dates in 5 of
  30 configurations (tau = 2, G = 200).  Example (AUDIT A3): Beta(1,32),
  T = 20, beta = 0.01; EM[uniform] and EM[rand1] have identical log
  likelihood, KKT 1.55e-15, fitted cells equal to 1.1e-16, weight TV 0.34, and
  select dates 5 and 1.
* **Witness.**  In every exact-probability case where certified EM and the
  NPMLE disagree (601 rows), an LP finds weights reproducing EM's fitted cell
  probabilities to <= 1.1e-16 (objective equal to 4.4e-16, same KKT residual)
  that select the NPMLE's date: 601/601.  In finite samples (4,003 datasets
  with certified EM and a valid NPMLE), 1,182 disagree and the witness
  (residual <= 1.9e-12, objective within 3.6e-12) selects the NPMLE's date in
  1,182/1,182.
* **The NPMLE is only approximately optimal.**  No support-reduction solve
  requested at 1e-13 is certified (median achieved residual 1.6e-8 on exact
  targets, 1.3e-8 on finite data); EM endpoints at the cap reach a median
  3e-15 (exact) / 7e-16 (finite).  Tightening the requested NPMLE tolerance from
  1e-8 to 1e-13 changes its selected date in 6.7% (exact) and 1.9% (finite)
  of cases.  Differences in predicted observation probabilities between
  the EM endpoint and the NPMLE are small but nonzero (median 1.2e-8 exact,
  6.9e-9 finite; max 9.3e-6 and 9.3e-5).  We therefore speak of
  nonuniqueness only where the witness or the certified-EM comparison
  demonstrates it, not because two fits pass a tolerance.
* **What the witnesses do not show.**  They establish nonuniqueness at the
  endpoint.  They do not establish that drift within an unchanged fit explains
  the deterioration from EM55 to the endpoint.  The mechanism follow-up finds
  that part of the welfare-contrast change near zero
  (`MECHANISM_REPORT.md` section 3).

### 3.3 Finite samples: reference populations (paired, shared datasets)

Reference datasets, 30 configurations x 50 datasets per N, scored under P.
MSR x 1e3; p10 = tenth percentile of efficiency.  Status counts are for the
EM endpoint (uniform start) and for the NPMLE.

| N | Procedure | MSR | mean regret | p10 | date != NPMLE | status |
|---|---|---|---|---|---|---|
| 1e3 | NPMLE | 4.466 | 0.0383 | 0.8672 | - | 897 cert., 581 valid-uncert., 22 invalid |
| 1e3 | EM30 / EM55 / EM80 (uniform) | 0.180 / 0.196 / 0.248 | 0.0025 / 0.0026 / 0.0031 | 0.9964 / 0.9964 / 0.9935 | 61% / 59% / 57% | |
| 1e3 | EM55, mean of 7 non-uniform starts | 0.224 | 0.0029 | 0.9964 | 58% | |
| 1e3 | EM at cap 1e6 (uniform) | 4.043 | 0.0343 | 0.8683 | 20% | 1,042 cert., 458 cap |
| 1e3 | CV-stopped EM (label-free; median k = 31) | 1.293 | 0.0118 | 0.9622 | 48% | |
| 1e5 | NPMLE | 1.415 | 0.0120 | 0.9750 | - | 896 cert., 559 valid-uncert., 45 invalid |
| 1e5 | EM30 / EM55 / EM80 (uniform) | 0.047 / 0.004 / 0.003 | 0.0020 / 0.0006 / 0.0005 | 0.9961 / 0.9969 / 0.9969 | 44% / 45% / 45% | |
| 1e5 | EM55, mean of 7 non-uniform starts | 0.018 | 0.0011 | 0.9969 | 43% | |
| 1e5 | EM at cap 1e6 (uniform) | 0.261 | 0.0029 | 0.9990 | 37% | 1,366 cert., 134 cap |
| 1e5 | CV-stopped EM (median k = 355) | 0.073 | 0.0010 | 1.0000 | 39% | |

Prespecified paired differences in MSR x 1e3 (95% two-stage cluster bootstrap
over configurations then datasets, 2,000 replicates; conditional on this fixed
benchmark, not a statement about arbitrary populations):

| Comparison | N = 1e3 | N = 1e5 |
|---|---|---|
| EM55 - NPMLE | -4.27 [-7.00, -1.99]; 28 configs favour EM, 2 NPMLE | -1.41 [-2.93, -0.23]; 18 / 2 (10 tied) |
| EM30 - NPMLE | -4.29 [-7.34, -1.87] | -1.37 [-2.83, -0.26] |
| EM80 - NPMLE | -4.22 [-6.99, -1.86] | -1.41 [-2.85, -0.25] |
| EM at cap - NPMLE | -0.42 [-0.91, -0.08] | -1.15 [-2.39, -0.18] |
| EM55 - EM at cap | -3.85 [-6.63, -1.64] | -0.26 [-0.59, -0.04] |
| EM55 (7 non-uniform starts) - NPMLE | -4.24 [-7.02, -1.99] | -1.40 [-2.79, -0.27] |
| CV-stopped EM - NPMLE | -3.17 [-5.31, -1.53] | -1.34 [-2.77, -0.28] |

* **The path.**  At N = 1e3, MSR is minimal at iterations 16-37 (by
  initialization), rises steadily from about k = 100, and reaches the NPMLE's
  level at the cap; 48% of trajectories change date after k = 55.  At N = 1e5
  the minimum is at k = 25-1,800 and the rise near the cap is small.
* **Where the gain sits.**  Beta(1,32) contributes about two-thirds of the
  pooled EM55 gain at N = 1e3 (-14.3 of a per-family mean of -4.3) and about
  98% at N = 1e5 (-6.9; the other four families contribute between -0.05 and
  0).  EM55 is worse than the NPMLE in 2 configurations at each N (uniform
  start); the counts are the same for every initialization within one.

### 3.4 Compatible populations (pair datasets; both members scored)

One dataset per (pair, N, seed), drawn from the shared pre-failure law (lower
member); every procedure scored under both members.  rho = max over members of
the trial-mean regret, averaged over 30 pairs; "trialwise" = mean of trialwise
maximum regret.  Uniform start.

| N | Procedure | rho | trialwise | mean regret lower | mean regret upper | mean regret P |
|---|---|---|---|---|---|---|
| 1e3 | NPMLE | 0.0633 | 0.0718 | 0.0615 | **0.0142** | 0.0364 |
| 1e3 | EM55 | **0.0355** | 0.0370 | 0.0058 | 0.0343 | 0.0020 |
| 1e3 | EM at cap | 0.0555 | 0.0671 | 0.0536 | 0.0162 | 0.0329 |
| 1e3 | CV-stopped EM | 0.0347 | 0.0473 | 0.0207 | 0.0305 | 0.0113 |
| 1e5 | NPMLE | 0.0365 | 0.0434 | 0.0233 | **0.0242** | 0.0127 |
| 1e5 | EM55 | 0.0344 | 0.0344 | 0.0044 | 0.0325 | 0.0007 |
| 1e5 | EM at cap | **0.0292** | 0.0335 | 0.0077 | 0.0277 | 0.0032 |
| 1e5 | CV-stopped EM | 0.0314 | 0.0341 | 0.0053 | 0.0308 | 0.0018 |

Paired rho differences (bootstrap of trials within pairs, 2,000 replicates):
EM55 - NPMLE = -0.0278 [-0.0308, -0.0248] at N = 1e3 (26 pairs better,
4 worse) and -0.0021 [-0.0040, -0.0006] at N = 1e5 (10 better, 8 worse,
12 tied); EM30 - NPMLE at N = 1e5 = +0.0018 [-0.0002, +0.0032]; EM at cap -
NPMLE = -0.0079 and -0.0072.  The reference datasets scored under the same
members give the same picture (`analysis/finite_ref_risk*.csv`).

* **The advantage reverses on one member.**  Under the upper member, early
  EM's regret (0.034 at N = 1e3, 0.033 at N = 1e5) is higher than the
  NPMLE's (0.014, 0.024).  EM's gains are under P and the lower member.  The
  worst-member risk still favours EM55 at N = 1e3 because the NPMLE's
  lower-member regret is large; at N = 1e5 the worst-member advantage is
  small, and the EM endpoint (1,366 of 1,500 certified at N = 1e5 on reference
  data), not EM55, has the lowest rho.
* Exact probabilities give the same pattern: Beta(1,32), T = 20,
  beta = 0.01 (Fig. 3a), and pooled tau = 2: EM55 0.0005 / 0.0038 / 0.0327
  (P / lower / upper) against NPMLE 0.0033 / 0.0103 / 0.0256.

### 3.5 Initialization

Initialization changes where EM is at k = 30-80 and, at exact
probabilities, which certified maximizer it reaches (5/30 configurations),
but not the finite-sample conclusions: EM55 MSR ranges 0.14-0.33 (N = 1e3)
and 0.002-0.05 (N = 1e5) across the eight starts, every start beats the NPMLE
in 28 (N = 1e3) and 17-18 (N = 1e5) configurations, and all starts move
toward the NPMLE's level at N = 1e3.  *Hindsight-best start (diagnostic only,
not deployable): "high" at N = 1e3 (0.144), "low" at N = 1e5 (0.0023).*

### 3.6 Grid resolution (prespecified subset, uniform start)

| MSR x 1e3 | N = 1e3: G = 200 / 400 / 800 | N = 1e5: G = 200 / 400 / 800 |
|---|---|---|
| EM55 | 0.036 / 0.034 / 0.032 | 0.011 / 0.0002 / 0.0002 |
| EM at cap | 2.60 / 2.60 / 2.56 | 0.27 / 0.10 / 0.10 |
| NPMLE | 2.97 / 9.29 / 6.07 | 1.08 / 5.99 / 6.64 |

The EM path is nearly grid-invariant; the NPMLE's selected atomic solution
is not, and refinement makes it worse.

## 4. What the evidence supports

1. **Does regret deteriorate during optimization with exact probabilities?**
   No.  It falls and then plateaus, and the certified endpoint is better than
   the NPMLE.  Finite-sample overfitting alone therefore cannot explain the
   EM-NPMLE gap.
2. **Does early stopping help across initializations?**  Yes, on this
   benchmark, for every start: EM55 beats the NPMLE at both N.  At N = 1e3
   most of the gain is lost by iterating to the cap (EM55 - endpoint = -3.85;
   endpoint - NPMLE = -0.42).  At N = 1e5 most of the gain is kept at the
   endpoint (endpoint - NPMLE = -1.15 of -1.41).
3. **Do fitted observations become similar while welfare predictions stay
   different?**  Yes, and demonstrably so: weight vectors with identical
   fitted cell probabilities (to 1e-16 exact, 1e-12 finite), hence identical
   likelihood and KKT residual, select different dates.  The likelihood does
   not determine the decision.
4. **Does the advantage persist on compatible alternatives?**  Only partly.
   It reverses under the upper member.  Worst-member risk favours EM55
   clearly at N = 1e3 and only marginally at N = 1e5.
5. **Sample size, observation length, grid.**  The effect of early stopping
   is large at N = 1e3 and small at N = 1e5.  At tau = 3, 4 the NPMLE's
   population regret is already near zero.  Grid refinement leaves EM
   unchanged and degrades the NPMLE.
6. **Can a fixed stopping count be justified without true welfare?**  Not by
   the likelihood: it improves monotonically while regret rises.  The
   label-free CV rule recovers much of the gain (MSR 1.29 vs 4.47 at
   N = 1e3), but less than EM55, and it stops much later at N = 1e5
   (median k = 355).  The 30-80 window is good on these reference families,
   but this study gives no data-driven reason for it.

**Mechanism** (details and full accounting: `MECHANISM_REPORT.md`).  On the
reference populations at tau = 2, early stopping retains an upward bias in the
estimated third moment m3 = E[p^3] (median +5.3% to +16.8% at N = 1e3 and
+3.0% to +12.6% at N = 1e5 across all eight initializations).  Continued
fitting often lowers that estimate and shifts decisions toward costly earlier
intervention.  Among datasets whose regret rises from EM55 to the endpoint
(660 of 1,500 at N = 1e3; 110 of 1,500 at N = 1e5), a post-hoc intervention
is informative.  It holds m3 at EM55's value and takes the endpoint's m1 and
m2, choosing the L1-closest distribution to EM55.  It keeps EM55's date in
77.2% of feasible events (470/609; 51 infeasible) and 70.0% (77/110).  At
N = 1e3 it removes 89.3% of the extra regret: a ratio of means over the 609
feasible events, or 77.4% if infeasible events count as unrepaired.  The
interventions support this explanation and identify the m3 change as a major
channel.  A biased estimate can help here because the observed loss is
asymmetric in the m3 error.  Underestimating m3 goes with too-early dates and
large regret, while moderate overestimation (up to +50%) goes with much
smaller regret; larger overestimates are costly again.  The study does not
compare overall estimation accuracy.  At N = 1e5 only about 21% of
deteriorating datasets move farther from the true observation probabilities
(94% at N = 1e3), so sampling-noise overfitting alone does not explain the
deterioration.

Qualifications: the single-moment interventions are post hoc.  This is a
population-dependent explanation for this benchmark, not a universal benefit
of early stopping; it also explains the reversal under the upper pair members,
whose optimum is earlier.  Holding m3 also blocks most improvements (76.7% /
90.7%).  Distribution shape still matters (the endpoint's shape with EM55's
fitted moments keeps EM55's date in only about 44% of events).  The
nonuniqueness witnesses (section 3.2) remain valid but do not explain the
EM55-to-endpoint deterioration.

## 5. Limitations

* Results are conditional on the 30 frozen configurations; the pooled gain is
  concentrated in Beta(1,32).
* Compatible-pair evidence uses the extreme pairs only: two members of an
  infinite identified set.
* EM is capped at 1e6 (finite) / 1e6-1e7 (exact) updates; 19% of finite
  endpoints and 20-32 of 40 trajectories at tau = 3, 4 are not certified.
* The NPMLE solver itself is only approximately optimal (AUDIT A4).
* Finite samples were run at tau = 2 only; tau = 3, 4 are population-level.

## 6. Reproduce

See `experiments/em_mechanism/README.md` for every command, which commands
use saved results, and the claim-to-file map.  Verification of every reported
number from saved outputs (seconds):

```bash
python -m experiments.em_mechanism.verify_claims
```

Figures (`figures/`, PDF and PNG): 1a exact trajectories (gap, regret under
three compatible populations, date; one configuration per family at T = 20,
beta = 0.002); 1b finite-sample pooled path; 2 initialization sensitivity
(aggregate, per configuration); 3 compatible populations (one trajectory
under three populations, welfare curves, pooled worst-member risk, position
in the identified interval); 4 grid, observation length and realizable truth;
5 mechanism (asymmetric loss in the m3 error, m3 bias by estimator,
interventions on deterioration events, mass shift).
