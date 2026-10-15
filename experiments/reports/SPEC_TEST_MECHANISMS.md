# Specification test: frozen definitions (Experiments 21-23)

**Frozen 2026-08-21, before any specification-test result was computed.**
The brief requires the departure mechanisms be recovered from an existing plan
if one exists, and otherwise "explicitly documented before being treated as
frozen". No prior definition exists (see AUDIT.md), so the mechanisms below are
implementation choices, fixed here in advance and not revised afterwards.

Nothing in this file may change once `results/spec_test_*.csv` exist.

---

## 1. The test itself -- fully specified by the draft

Remark 2 of the draft: `M_t` is a truncated Hausdorff moment set, nonempty
exactly when the observed moment vector lies in the truncated moment cone of
`[0,1]`, which is a semidefinite condition on the Hankel matrix **and** the
localizing matrices for `p >= 0` and `1-p >= 0`. The draft is explicit that
Hankel positivity alone is insufficient, and gives the counterexample
`(1, 2, 4, 8, 16)` -- Hankel PSD as the moments of a point mass at `p = 2`, with
no representing measure on `[0,1]`.

A length-`t` prefix identifies moments `m_0..m_{2t-1}`, an **odd** number of
steps (`n = 2t-1 = 2k+1`, `k = t-1`). The Karlin-Studden conditions for that case
are that both of the following are positive semidefinite:

    A_k = ( m_{i+j+1} )_{i,j=0..k}                 localizing for p >= 0
    C_k = ( m_{i+j} - m_{i+j+1} )_{i,j=0..k}       localizing for 1-p >= 0

Both are used. Neither alone is the test. The draft's counterexample is included
as a unit test of the implementation.

## 2. Recovering moments from observables

By Lemma 1 the cohort counts span `P_{2t-1}`, so the moments are a linear
function of them. Writing `M` for the monomial coefficient matrix of the
functionals `phi_{s,k}` (the same matrix Experiment 1 takes the rank of), the
population relation is `n = M m`. Moments are recovered as
`mhat = pinv(M) nhat`, which is exact in population and least-squares in sample.

## 3. Test statistic and calibration

**Statistic.** `S = min( lambda_min(Ahat_k) / scale_A , lambda_min(Chat_k) / scale_C )`,
each matrix scaled by its own trace so the statistic is invariant to the overall
magnitude of the moments. `S > 0` means the estimate lies inside the cone.

**Why a raw sign test will not do.** Sampling noise pushes `S` below zero some of
the time even under a correctly specified model, especially when the true
distribution sits near the cone boundary. The brief is explicit: tiny numerical
PSD violations must not be classified as model rejection.

**Calibration: null-projected parametric bootstrap.** For each dataset:

1. fit a distribution on the support grid to the observed record counts by
   nonnegative least squares (weights `>= 0`, summing to 1). Any grid
   distribution automatically has moments inside the cone, so this is a bona
   fide null;
2. draw `B` bootstrap datasets from `Multinomial(N, q0)` where `q0` is that
   fitted null's record law;
3. recompute `S` on each, giving the null distribution of the statistic at this
   sample size;
4. **reject** the model when `S_observed < quantile_alpha(S_bootstrap)`.

This calibrates against the actual sampling noise at each `N` rather than against
an arbitrary numerical threshold.

**Frozen test parameters**

| Parameter | Value |
|---|---|
| level `alpha` | 0.05 |
| bootstrap draws `B` | 299 |
| replications per cell | 400 |
| NNLS support grid | 101 uniform points on [0,1] |
| prefixes | t = 2, 3, 4 |
| population sizes | N = 1e3, 5e3, 2e4, 1e5 |
| risk distributions | Beta(1,32), Beta(1,8), Beta(1,4) |
| observation model | gamma = 1 |

## 4. Experiment 22 -- drifting risk (FROZEN)

**Mechanism.** Each individual's per-period risk escalates geometrically from
its own baseline:

    p_i^t = min( p_i * (1 + delta)^(t-1), 1 )

At `delta = 0` this is exactly the null model, so the alternative nests the null
and `delta` is an interpretable departure strength. The drifted `p_i^t` governs
**both** the flag probability and the exit probability in period `t`, since
`ptilde = p` at `gamma = 1`.

**Frozen strengths.** `delta in {0.05, 0.10, 0.20, 0.40}` (5% to 40% escalation
per period).

**Why this violates the model.** The paper's `phi_{s,k}` assumes one fixed `p`
per individual across all periods. Under drift the observable counts are no
longer expectations of `phi_{s,k}(p)` for any single distribution of `p`, so the
recovered moment vector need not lie in the cone at all.

## 5. Experiment 23 -- within-group correlated failures (FROZEN)

**Mechanism.** Individuals are partitioned into groups of fixed size `g`. In each
period, each group draws one shared uniform `U_group`. Each individual `i` in the
group fails that period if `V_i < p_i`, where

    V_i = U_group        with probability rho      (shared shock)
        = U_i            with probability 1 - rho  (own independent draw)

drawn independently per individual per period.

**Marginals are preserved exactly.** `P(V_i < p_i) = rho * p_i + (1-rho) * p_i = p_i`
for every individual and period, so each individual's marginal failure process is
*identical* to the null model. Only the dependence across individuals changes.

**This is deliberate, and it is the interesting case.** Because the marginals are
untouched, the *expected* cohort counts are unchanged, so the expected moment
vector still lies in the cone. Any power the test has against this alternative
must come from **variance inflation**: correlated failures make the empirical
moment vector noisier than the independence bootstrap expects, pushing `S` below
a critical value calibrated under independence more often than `alpha`.

Recording this prediction in advance matters. If the measured power against
correlation is near the false-positive rate, that is not a failure of the
implementation -- it is the correct answer for a test built on the mean moment
vector, and it should be reported as a genuine limitation of the diagnostic
rather than explained away afterwards.

**Frozen parameters.** `g in {10, 50}`, `rho in {0.25, 0.50, 1.00}`.
Flags remain independent across individuals; only failures are correlated, which
is what "failures correlated within groups" states.

## 6. What is reported

- Experiment 21: false-positive rate against `N` and `t` under correct
  specification. Should sit near `alpha = 0.05`.
- Experiment 22: detection rate against `N`, `t`, and `delta`.
- Experiment 23: detection rate against `N`, `t`, `g`, and `rho`.
- One figure with all three against `N`.

The comparison of interest is power *against the measured false-positive rate*,
not against the nominal 0.05.
