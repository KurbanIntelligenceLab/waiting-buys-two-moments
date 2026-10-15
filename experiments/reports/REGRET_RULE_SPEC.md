# Regret-calibrated randomized rule: frozen specification

**Candidate A with randomized stopping.** This supersedes the exploratory
`REGRET_RULE_DESIGN.md` (which enumerated three candidates without choosing).
The design below is complete: formula, defaults, degenerate cases and tuning
protocol are all fixed here, before any benchmark run.

**Status: specified, not yet implemented, not yet evaluated.**
No benchmark performance number has been produced for this rule.

---

## 1. What the rule sees

The i.i.d. unit is one individual's observable record through the prefix. The
record alphabet `A_t` has `|A_t| = 8, 13, 19` atoms for `t = 2, 3, 4`
(exit at end of period `s <= t` with `y^s = k`, or alive at end of `t` with
`y^t = k`). Observed counts `n` are the exact sufficient statistic:

    n ~ Multinomial(N, q(Q)),      q_a(Q) = E_Q[psi_a(p)]

with `psi_a` the record functionals already in `wtm/observation.py`. Records --
not cohort counts -- are used, matching the draft's description of the plug-in
rules as estimating "from the observed exit records".

## 2. The rule, step by step

### S1. Estimate the risk distribution (NPMLE)

On a support grid `g_1..g_{G}`, with `Psi[a,i] = psi_a(g_i)`:

    what = argmax_{w >= 0, sum w = 1}  sum_a n_a log( (Psi w)_a )

by EM, whose update is (columns of `Psi` sum to 1, so this is a standard finite
mixture):

    w  <-  w * ( Psi^T ( n / (Psi w) ) ) / N

Implemented and verified in `wtm/npmle.py`.

### S2. Welfare trajectory

    What^s = W^s(what),   s = 1 .. T-1

computed exactly from `what` by the mean-field machinery (group masses, greedy
budget fill). No sampling error enters here: given `what`, `What^s` is exact.

### S3. Adjacent welfare contrast, in relative units

    What^max = max_r What^r
    Deltahat_s = ( What^{s+1} - What^s ) / What^max,      s = 1 .. T-2

`Deltahat_s` is the *gain from waiting one more period*, measured on the same
relative scale as the paper's welfare-efficiency and mean-square-regret metrics.
Absolute welfare differences are O(beta) and would make the hyperparameters
scale with the budget ratio, which is a benchmark factor. **Adapted, not
inherited** (Section 4).

### S4. Sampling uncertainty by paired bootstrap

For `b = 1 .. E`:

    n^(b) ~ Multinomial(N, n/N)          resample the observed record multinomial
    what^(b) = NPMLE(n^(b))              refit
    Delta_s^(b) = ( W^{s+1}(what^(b)) - W^s(what^(b)) ) / max_r W^r(what^(b))

    sehat_s = sd_b( Delta_s^(b) )

**The standard error is computed on the replicate-wise differences.** It must
never be assembled as `sqrt(se(W^{s+1})^2 + se(W^s)^2)`: `What^{s+1}` and
`What^s` come from the same fit and are strongly positively correlated, so the
marginal-variance formula would inflate `sehat_s`, drive every `z_s` toward zero
and silently turn the rule into near-uniform randomization.

### S5. Studentize

    z_s = Deltahat_s / max( sehat_s , eps_se )

### S6. Smooth wait probability

    pi_s = Phi( z_s / lambda )

with `Phi` the standard normal CDF. `pi_s` is the probability of **waiting** past
date `s`.

### S7. Randomized stopping

Draw `U_1 .. U_{T-2}` i.i.d. Uniform(0,1) from the **rule seed**, which is
separate from the simulation seed. Walk forward:

    for s = 1 .. T-2:
        if U_s <= pi_s:  continue      (wait)
        else:            return s      (act now)
    return T-1

The induced distribution over dates is proper by construction:

    P(that = s)   = ( prod_{r<s} pi_r ) ( 1 - pi_s ),   s = 1..T-2
    P(that = T-1) = prod_{r=1}^{T-2} pi_r

This is the only composition that (i) is built from *adjacent* contrasts as the
draft's sentence states, (ii) yields a valid distribution over dates with no
normalisation step, and (iii) has an operational meaning specific to timing:
each `pi_s` answers "is it worth waiting one more period?".

## 3. Behaviour at the limits (sanity, not tuning)

- `lambda -> 0`: `pi_s -> 1{Deltahat_s > 0}`. The rule becomes the deterministic
  plug-in "wait while the estimated trajectory is still rising".
- `lambda -> infinity`: `pi_s -> 1/2`, an unbiased coin at every step.
- `N -> infinity`: `sehat_s -> 0`, so `|z_s| -> infinity` wherever
  `Delta_s != 0`, and the rule converges to the plug-in argmax. **This rule is
  consistent**, unlike the identified-set ensemble (Candidate B), whose spread
  does not vanish with `N`. That is a deliberate commitment: it treats the
  residual ambiguity at large `N` as *statistical*, so the rule should stop
  hedging once the data can resolve the contrast.

## 4. What is inherited from Kitagawa, Lee & Qiu (2023) and what is not

The draft cites arXiv:2310.06242 for the randomized posture. **It is motivation,
not an applicable theorem**: their result concerns a single binary treatment
choice, ours is a `(T-1)`-way ordered sequential choice. No optimality claim is
made or implied for the rule above.

| Element | Status | Note |
|---|---|---|
| Criterion is regret, and mean square regret specifically | **inherited** | MSR is the draft's own Table 4 column |
| Under MSR the optimal rule is *fractional*, not bang-bang -- randomizing is not a hedge but the right shape of answer | **inherited (qualitative)** | This is the reason to randomize at all. Under max-regret or a plug-in you would commit. |
| The fraction should vary *smoothly* with how well resolved the welfare contrast is | **inherited (qualitative)** | |
| The exact optimal fractional map | **NOT inherited** | Derived for a binary choice under their loss and limit experiment. Not transferable, and not reproduced here. |
| `Phi(z/lambda)` as the specific link | **adapted** | Chosen because at `lambda = 1` it is the flat-prior posterior probability that waiting helps (Section 5). Probit is conventional in this literature; logistic is an equally defensible alternative and is an ablation axis. |
| Sequential stopping composition (S7) | **adapted** | No counterpart in KLQ. Forced by the multi-date structure. |
| Relative rather than absolute contrast scale (S3) | **adapted** | Forced by the paper's own relative-regret metric. |
| Bootstrap SE rather than an asymptotic variance (S4) | **adapted** | Forced: `W^s` is a non-smooth functional of an NPMLE, whose influence function is not standard. |
| Paired replicate-wise differencing (S4) | **adapted** | Forced by the correlation between adjacent trajectory points. |
| `eps_se`, NPMLE grid/tolerance, nonparametric vs parametric bootstrap | **optional** | Recorded and ablated; none is claimed to follow from anything. |

## 5. Hyperparameters

### Ensemble size `E` = number of bootstrap replicates

**Default: `E = 200`.**

Justification, from precision rather than performance: the relative standard
error of a standard-deviation estimate from `E` replicates is approximately
`1/sqrt(2(E-1))` -- 5.0% at `E = 200`, 10.1% at `E = 50`, 16.2% at `E = 20`.
`sehat_s` enters only through the smooth link, which damps that noise further.
5% is comfortably below the resolution at which `pi_s` changes materially.

**Measured cost.** One NPMLE fit takes **6-308 ms** (`wtm/npmle.py`, grid 200,
measured across `t = 2,3,4` and `N = 1e3, 2e4, 1e5`). So `E = 200` costs roughly
**1.2-62 s per rule evaluation**. Note the cost is *worst at the smallest N*:
at `N = 1000, t = 2` EM needed 5,000 iterations and had not converged, versus
276 iterations at `N = 1e5`. This inverts the usual intuition and means the
`N = 10^3` column of Table 4 is the expensive one, not the cheap one.

**Ablation axis (draft Section 7 requires it): `E in {25, 50, 100, 200, 400}`.**

### Regularization strength `lambda`

**Default: `lambda = 1.0`.**

Justification, again structural rather than tuned: if `Deltahat_s` is
approximately Gaussian around the true contrast with standard deviation
`sehat_s`, then under a flat prior

    P( Delta_s > 0 | data )  =  Phi( Deltahat_s / sehat_s )  =  Phi(z_s)  =  pi_s at lambda = 1.

So at the default the rule waits with exactly the probability that waiting one
more period genuinely improves welfare. That is an interpretable anchor chosen
before any evaluation and requiring no reference to how the rule performs.

`lambda` controls smoothness and shrinkage jointly: it is a temperature on the
link, and equivalently shrinks the studentized statistic toward zero by a factor
`1/lambda`.

**Ablation axis: `lambda in {0.25, 0.5, 1.0, 2.0, 4.0}`.**

### Fixed numerical settings (not hyperparameters; separately ablated per the draft)

| Setting | Default | Note |
|---|---|---|
| NPMLE support grid | 200 uniform points on [0,1] | draft ablates grid resolution: {50, 100, 200, 400, 800} |
| NPMLE tolerance | 1e-8 relative on log-likelihood | draft ablates tolerance: {1e-6, 1e-8, 1e-10} |
| NPMLE max iterations | **20000** (raised from 5000 on 2026-08-27) | 10% of gamma=1 fits hit the old cap; worst observed requirement 7,522. Raising a cap can only convert non-convergence into convergence, never change a converged answer, so this is a correctness fix and not tuning. Non-convergence is still recorded, never silently accepted. |
| `eps_se` | 1e-6, in relative-contrast units | scale-free because the contrast is normalised |
| link | probit | logistic is the ablation alternative |
| bootstrap | nonparametric, resampling the observed record multinomial | parametric (resampling the fitted `q(what)`) is the ablation alternative |

## 6. Degenerate cases -- required behaviour

| Situation | Behaviour |
|---|---|
| `sehat_s = 0` and `Deltahat_s != 0` | floor applies, `\|z_s\|` large, `pi_s -> 0 or 1`: commit |
| `sehat_s = 0` and `Deltahat_s = 0` | **define `z_s = 0`**, so `pi_s = 1/2`. Must be explicit: `0/0` yields `nan` and would poison the walk |
| `What^max <= 0` | no estimate; record `failure_reason = "degenerate trajectory"` |
| primary NPMLE does not converge | no estimate; `failure_reason = "npmle_no_convergence"` |
| a bootstrap replicate does not converge | drop that replicate, count it; if more than 20% are dropped, no estimate |
| estimated record law has an empty atom | permitted -- NPMLE handles zero counts natively |

Every no-estimate outcome is scored, never dropped from the denominator.

## 7. Tuning protocol -- fixed in advance

**Primary protocol: no tuning.** Run with `lambda = 1.0`, `E = 200` and the fixed
numerical settings above, and evaluate once on the benchmark. Both defaults were
chosen from structural arguments (posterior interpretation; standard-error
precision), not from performance, so there is nothing to leak.

**If tuning is judged necessary, it happens on a disjoint tuning grid**, defined
here in advance and sharing no cell with the benchmark:

| Factor | Tuning grid | Benchmark grid |
|---|---|---|
| distribution | Beta(1,16), Beta(1,2) | Beta(1,32), Beta(1,8), Beta(1,4), + mixture/bimodal |
| budget ratio | 0.001, 0.005 | 0.0005, 0.002, 0.01 |
| horizon | 12, 20 | 12, 20 |
| prefix | 2, 3, 4 | 2, 3, 4 |
| gamma | 1 | 1, 2, 4 |
| seeds | block 900000+ | frozen benchmark seeds (D3) |

Select `(lambda, E)` minimising mean square regret on the tuning grid, freeze,
then evaluate **once**. No further adjustment after any benchmark number is seen.

**The decision whether to tune must be taken before the first benchmark result
is looked at.** Recorded here so it cannot be taken afterwards.

## 8. Complexity

Per rule evaluation: `E + 1` NPMLE fits, `E + 1` trajectory evaluations.
Trajectory cost is `O(T^2 G)` ~ 80k flops, negligible beside the fits. So the
rule costs `(E+1)` x (6-308 ms), i.e. **1.2-62 s at `E = 200`**.

For scale: at 648 beta-only finite-sample cells and 50 seeds that is 32,400
evaluations, or roughly **11-560 core-hours** for this rule alone. `E = 50`
divides that by four at a precision cost of 10.1% vs 5.0% relative error in
`sehat`. This is a real input to the seed-count decision (D3) and should be
settled alongside it.

## 9. Deliberately not decided here

- Whether to tune at all (Section 7 offers both paths; the choice is yours).
- The seed count (D3), which multiplies every cost above.
- Whether the benchmark's other five rules run before or after this one is
  frozen -- a scientific-integrity question, not a technical one.

## 10. Reference pseudocode

```
def regret_calibrated(records_n, N, t, gamma, T, beta, grid, lam, E, rule_seed):
    Psi   = record_matrix(grid, t, gamma)
    what  = npmle(records_n, Psi)                     # S1
    if not what.converged: return NO_ESTIMATE
    W     = [welfare(what.w, s) for s in 1..T-1]      # S2
    Wmax  = max(W)
    if Wmax <= 0: return NO_ESTIMATE
    D     = [(W[s+1] - W[s]) / Wmax for s in 1..T-2]  # S3

    boot = []                                          # S4
    for b in 1..E:
        nb = multinomial(N, records_n / N)
        wb = npmle(nb, Psi)
        if not wb.converged: continue
        Wb = [welfare(wb.w, s) for s in 1..T-1]
        boot.append([(Wb[s+1] - Wb[s]) / max(Wb) for s in 1..T-2])
    if len(boot) < 0.8 * E: return NO_ESTIMATE
    se = std(boot, axis=0)                             # paired differences

    z  = [0.0 if (se[s] <= 0 and D[s] == 0)            # S5, S6
          else D[s] / max(se[s], EPS_SE) for s in ...]
    pi = Phi(z / lam)

    rng = Generator(rule_seed)                         # S7
    for s in 1..T-2:
        if rng.uniform() > pi[s]: return s
    return T-1
```
