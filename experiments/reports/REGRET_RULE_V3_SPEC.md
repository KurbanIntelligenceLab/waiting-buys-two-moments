# Regret-calibrated rule, variant 3: global expected-regret minimisation

**Definition frozen 2026-09-03, before implementation and before any evaluation.**
The certified NPMLE estimator is unchanged. Only the map from the bootstrap
distribution of trajectories to a date changes.

## 0. Why the obvious form of the brief does not work

The brief proposes the two losses

    L_stop(s) = [ max_{u>s} W^u - W^s ]_+        regret from stopping at s
    L_wait(s) = [ W^s - max_{u>s} W^u ]_+        regret from continuing past s

and comparing their expectations. But for **every** trajectory, replicate by
replicate,

    L_stop(s) - L_wait(s) = max_{u>s} W^u - W^s = R_s     identically,

because `[x]_+ - [-x]_+ = x`. Hence

    E[L_stop(s)] <= E[L_wait(s)]   <=>   E[R_s] <= 0.

**The asymmetry cancels in the difference.** A pointwise comparison of the two
expected losses is exactly "threshold the bootstrap mean of `R_s` at zero" -- it
is v2 with the point estimate replaced by a bootstrap mean and `lambda -> 0`. It
is not asymmetry-aware, and it would not fix the diagnosed defect.

Asymmetry must therefore enter through the **level of regret at each candidate
date**, not through the sign of a contrast between adjacent ones. That is what
v3 does.

## 1. Definition

Let `W_b`, `b = 1..B`, be the welfare trajectories from the `B = 200` paired
bootstrap replicates (the same replicates v1 and v2 use, from the same certified
NPMLE refits). Define the **relative regret of committing to date `s` under
replicate `b`**:

    rho_b(s) = ( max_u W_b^u - W_b^s ) / max_u W_b^u        in [0, 1)

This is one minus the welfare efficiency of date `s` under replicate `b` -- the
evaluation metric itself, computed on the planner's own uncertainty. Then

    **t_hat  =  argmin_s  Rbar(s),        Rbar(s) = (1/B) * sum_b rho_b(s)**

with ties broken toward the **earliest** date (the frozen convention elsewhere:
acting sooner is the default when nothing distinguishes two dates).

That is the whole rule. It is a single global decision over `s = 1..T-1`.

## 2. Why this is asymmetry-aware

`rho_b(s)` is the actual welfare shortfall, not a signed contrast. A date whose
downside is severe in some replicates is penalised in proportion to that
severity. Because the fitted `W` rises steeply to its peak and decays slowly
after it, a date one period early carries a larger `rho` than one period late,
and averaging `rho` over replicates inherits that asymmetry automatically. No
weight, no loss ratio, and no asymmetry parameter is introduced: the asymmetry is
a property of `W`, and the rule simply stops discarding it.

## 3. Zero-uncertainty limit

If uncertainty vanishes, every replicate trajectory equals the fitted one,
`W_b = W` for all `b`. Then `rho_b(s) = (max_u W^u - W^s)/max_u W^u` for every
`b`, so

    Rbar(s) = (max_u W^u - W^s) / max_u W^u,

which is a strictly decreasing function of `W^s`. Therefore

    argmin_s Rbar(s) = argmax_s W^s ,

with the earliest-date tie-break selecting the first global argmax. **The
zero-uncertainty limit is exactly the NPMLE plug-in rule, date for date.** This
is an identity, not an approximation, and holds for every trajectory shape --
unimodal or not.

## 4. Randomisation is not needed, and would be strictly worse

v1 and v2 randomise because they convert a test statistic into a probability and
then draw. Under an **expected-loss** objective with a fixed belief distribution
(here, the bootstrap distribution over trajectories), the optimal action is the
one minimising expected loss, and it is deterministic: mixing over actions
attains a convex combination of their expected losses, which is at least the
minimum. So randomising can only increase expected regret.

Randomisation would still be justified under a *minimax* or adversarial
criterion, where a mixed strategy can improve the worst case. This rule does not
claim such a guarantee, and that limitation is stated rather than hidden.

Two consequences follow immediately:

* **No compounding.** There is no forward walk and no per-step draw, so there is
  no product of survival probabilities and no geometric leak of premature-stop
  probability in the number of dates before the optimum. The diagnosed
  compounding defect is removed by construction, not mitigated.
* **Reproducibility.** The rule is a deterministic function of the record counts
  and the bootstrap seed, so it has no walk-draw variance at all.

## 5. Tuning constants

**None are introduced.** `B = 200` and the paired multinomial bootstrap are
inherited unchanged from the frozen spec; `lambda`, the probit link, `EPS_SE` and
the walk are all *removed* rather than replaced. v3 has strictly fewer free
choices than v1 or v2. The only new convention is the tie-break, fixed to
earliest-date in advance and consistent with the existing fallback convention.

## 6. Assumptions required for consistency

1. **Bootstrap consistency of the trajectory**: `W_b -> W` in the sense that
   `max_b sup_s |W_b^s - W^s| -> 0` in probability as `N -> infinity`. This is
   the same assumption v2's argument needs (`se_s -> 0`).
2. **NPMLE consistency**: the certified fit's implied trajectory converges to the
   population trajectory. Note this is an assumption about the *trajectory*, not
   the mixing distribution -- the NPMLE is discrete and does not converge to a
   smooth mixing law, which is precisely the finding in
   `em_regularization_path.csv`. Consistency of `t_hat` for `t*` requires the
   trajectory functional to converge even where the mixing distribution does not.
   **If that fails, v3 converges to the argmax of whatever trajectory the
   certified NPMLE implies, not to `t*`.** So does the NPMLE plug-in; v3 inherits
   this and cannot repair it.
3. **Unique population argmax**, for the limit to be a single date.

Under 1-3, `Rbar(s) -> (max_u W^u - W^s)/max_u W^u` uniformly in `s`, and the
argmin of a uniformly convergent sequence with a unique minimiser converges to
it, so `P(t_hat = t*) -> 1`. **No unimodality is required.**

## 7. Expected failure mode, recorded before evaluation

**v3 will track the NPMLE plug-in more closely than v1 or v2 do, and will
therefore inherit more of its behaviour -- including its errors.** Because the
zero-uncertainty limit is the plug-in and the finite-sample rule is a smoothed
version of it, v3 should agree with NPMLE far more often than v2's 96.6% at
N=1e5. The hedge it retains is only the averaging over replicates.

Concretely, three predictions made now:

1. **v3 will select later than v2 on average**, because early stopping carries
   the larger `rho` and averaging penalises it. Signed date error should move
   toward zero from v2's negative side.
2. **v3's advantage over v2 will be largest at small N**, where the bootstrap
   spread is wide and the asymmetry has the most to act on, and will shrink at
   large N where both approach the plug-in.
3. **The main risk is that v3 becomes nearly indistinguishable from the NPMLE
   plug-in**, in which case it is a cleaner derivation of essentially the same
   rule rather than an improvement on it. If v3's held-out p10 lands within
   Monte Carlo error of NPMLE's, that is the honest conclusion and will be
   reported as such -- a rule that reduces to the plug-in is not a contribution
   merely because it was derived differently.

A fourth possibility, worth stating because it would be the most awkward: the
averaging could make v3 *worse* than v2 at large N by pulling it off the plug-in
argmax when the bootstrap spread is asymmetric but the point estimate is already
correct. Nothing in the design rules this out.

## 8. Evaluation protocol

Development/holdout split of the frozen benchmark by **configuration** (the
clustering unit of the paired analysis), created and committed **before** v3 is
evaluated. Development may inspect only the development half. The held-out half
is not inspected until this definition is frozen -- which this commit does.

v3 introduces no tuning constants, so no development-set fitting is required or
performed. The split exists to guarantee that no unstated choice was made against
the evaluation data.

Reported on the held-out half: v1, v2, v3, certified NPMLE and NNLS on p10
welfare efficiency, MSR, exact-date agreement, and paired welfare differences.

**No selection among v3 variants on held-out performance.** One definition is
frozen here and evaluated once.
