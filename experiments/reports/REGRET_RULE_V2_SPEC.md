# Regret-calibrated rule, variant 2: global-regret stopping

**Definition frozen 2026-09-01, before implementation and before any evaluation
on Experiment 9.** One change, motivated by one diagnosed failure mechanism.
The NPMLE estimator, its grid, tolerance, iteration cap, the bootstrap, the
studentization, the probit link and the randomized walk are all **unchanged**.

## 1. Mathematical definition

Let `W^s`, `s = 1, ..., T-1`, be the welfare trajectory implied by the fitted
NPMLE mixing distribution (identical to the current rule's, and to the NPMLE
plug-in's). Write `M = max_s W^s`.

### The statistic that changes

Current rule -- **adjacent** contrast:

    D_s = (W^{s+1} - W^s) / M                          s = 1, ..., T-2

Variant 2 -- **global forward regret**, the welfare still available by
continuing past `s`:

    R_s = ( max_{u > s} W^u  -  W^s ) / M              s = 1, ..., T-2

The maximum runs over the same candidate set the rule may still reach,
`u in {s+1, ..., T-1}`. Normalization by `M` is unchanged, so `R` and `D` are in
the same units and the inherited constants keep their meaning.

### Everything downstream is untouched

    se_s   = sd over bootstrap replicates b of R_s^b        (E = 200 replicates)
    z_s    = R_s / max(se_s, EPS_SE),  with 0/0 defined as 0
    pi_s   = Phi(z_s / lambda)                              (probit, lambda = 1)
    walk   : for s = 1, ..., T-2:  if u_s > pi_s, stop at s
             otherwise stop at T-1

`R_s^b` is computed from the same bootstrap replicate trajectory `W_b` as `D_s^b`
is now: draw `n_b ~ Multinomial(N, q_hat)`, refit NPMLE warm-started at the point
estimate, rebuild the trajectory, and apply the same formula. The bootstrap is
paired across dates exactly as before.

### Relation between the two statistics

Since `max_{u>s} W^u >= W^{s+1}`,

    R_s >= D_s   for every s,   with equality at s = T-2.

So variant 2 is uniformly more willing to continue, and the two rules coincide
at the last date they scan. Both reduce to the same object whenever the maximum
over `u > s` is attained at `u = s+1` -- which is exactly the unimodal case.

## 2. Behaviour as N -> infinity

Assume the fitted trajectory is consistent, `W_hat -> W` (this holds by NPMLE
consistency and continuity of `fill_greedy` in the mixing distribution), the
bootstrap standard errors satisfy `se_s -> 0`, and the population argmax
`t* = argmin{ s : W^s = M }` is unique.

Fix any `s < t*`. Then `t* > s` is in the feasible set, so
`max_{u>s} W^u = M > W^s`, hence `R_s -> r_s > 0` strictly. Therefore
`z_s = R_s / se_s -> +infinity` and `pi_s -> 1`: the walk continues past `s`
with probability tending to one.

At `s = t*`: every `u > t*` has `W^u < M = W^{t*}`, so `R_{t*} -> r < 0`
strictly, `z_{t*} -> -infinity`, `pi_{t*} -> 0`, and the walk stops with
probability tending to one.

Hence `P(t_hat = t*) -> 1`. **The rule is consistent, and consistency does not
require unimodality** -- only that some later date strictly beats `s` for every
`s < t*`, which is automatic because `t*` itself is one such date.

This is precisely what fails now. The current statistic `D_s` is negative
whenever the trajectory dips, whether or not a higher peak lies beyond, so
`pi_s -> 0` at the first dip and the walk stops there with probability tending
to one. Its limit is the first **local** maximum; variant 2's limit is the
**global** one.

## 3. Does it reduce to the NPMLE argmax when uncertainty vanishes?

**Yes, exactly.** Set `se_s = 0` for all `s`, so `pi_s = 1{R_s > 0}` (with the
frozen `0/0 -> z = 0` convention giving `pi = 1/2` only when `R_s` is exactly
zero, i.e. a genuine tie in the fitted welfare). The walk then continues while
`R_s > 0` and stops at the first `s` with `R_s <= 0`. But

    R_s <= 0  <=>  W^s >= max_{u>s} W^u  <=>  s is a global argmax of W over {s,...,T-1}

and scanning forward from `s = 1`, the first such `s` is the first global argmax
of the whole trajectory. **The zero-uncertainty limit of variant 2 is the NPMLE
plug-in rule, date for date.** The randomization is then a pure hedge layered on
top of the plug-in, which is what "regret calibration" was supposed to mean.

The current rule does **not** have this property: its zero-uncertainty limit is
the first local maximum, which differs from the plug-in whenever the trajectory
is not unimodal.

## 4. New tuning parameters

**None.** Every constant is inherited unchanged and keeps its original
justification:

| Constant | Value | Status |
| --- | --- | --- |
| `lambda` | 1.0 | inherited; the posterior anchor argument is unaffected -- at `lambda = 1`, `pi_s` is still the flat-prior posterior probability that continuing helps, now with "helps" meaning "some reachable later date is better" rather than "the next date is better" |
| `E` (bootstrap replicates) | 200 | inherited; the 1/sqrt(2(E-1)) = 5% relative-SD argument is unchanged |
| `EPS_SE` | 1e-6 | inherited, same units (`R` and `D` share the `M` normalization) |
| `MIN_BOOTSTRAP_FRACTION` | 0.8 | inherited |
| NPMLE grid / tol / max_iter | 200 / 1e-8 / 20000 | inherited, estimator untouched |
| link | probit | inherited |

The candidate set for the inner maximum is not a free parameter: it is forced to
`{s+1, ..., T-1}`, the dates the walk can still reach. Restricting it to a
shorter lookahead window would be a different rule with a new parameter, and is
deliberately **not** what is being proposed.

## 5. One genuinely new statistical object, and its predicted failure mode

`R_s` involves a **maximum** over dates, so `R_s^b` is a bootstrapped max-type
statistic. Two consequences, recorded in advance so that neither can be
presented afterwards as a discovery:

1. **`R_hat_s` is upward biased in finite samples.** `E[max_u W_hat^u] >=
   max_u E[W_hat^u]` by Jensen. So at small `N` the rule will over-estimate the
   welfare still available and **stop too late** -- the opposite of the current
   rule's bias. If variant 2 shows a positive mean date error at `N = 10^3`
   while the current rule shows a negative one, that is this effect and it is
   expected, not a surprise.
2. **The bootstrap for a max is not uniformly valid** near ties (Andrews-type
   non-differentiability): when two dates are nearly tied for the inner maximum,
   `se_s` can be poorly calibrated. This affects the width of the hedge, not the
   `N -> infinity` limit, since the argument in section 2 needs only
   `se_s -> 0`, not that `se_s` is correctly calibrated.

Neither is repaired here. Repairing them would introduce parameters, and the
instruction is one change.

## 6. Pre-declared evaluation protocol

Evaluated on the **existing Experiment 9 design**: the same 30 gamma=1
configurations, the same `N in {10^3, 5x10^3, 2x10^4, 10^5}`, the same prefixes
`t in {2,3,4}`, the same 20 seeds, and the **same simulation and rule seeds**, so
that variant 2, the current rule and the NPMLE plug-in are compared on identical
simulated data and identical bootstrap draws.

**Correctness check, run first:** recomputing the *current* statistic from the
same bootstrap pass must reproduce the frozen Experiment 9 selected dates
exactly. If it does not, the harness is wrong and no comparison is reported.

**Reported:** p10 / median / min welfare efficiency, MSR, exact-date agreement,
signed date error, and the paired win/loss against both the current rule and the
NPMLE plug-in, split by the trap / unimodal classification from
`regret_rule_diagnosis.csv`.

**No parameter will be tuned against these results.** If variant 2 underperforms,
that is reported as-is.
