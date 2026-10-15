# Measured results that require changes to the draft

> **REVISED 2026-08-27** after the mixture and bimodal families were frozen
> (D4/D4b) and the population grid expanded from 54 to 90 cells (18 to 30 at
> gamma=1).  Every number below is on the full grid.  The earlier 18-cell
> figures are superseded and are shown only for comparison.

Findings from the completed experiments that do not match the current prose.
None has been smoothed, reframed to fit, or omitted. Each states what the draft
says, what was measured, and what the honest replacement claim is.

---

## 1. `\RES[-single]`: uniqueness is 13.3% at t=2, not "usually"

**What the draft says** (Section 5, "The decision is pinned down far more
tightly"):

> "across the identified sets arising in our benchmark, the set of reachable
> optimal dates is a single date in `[RES-single]` of configurations"

and in Contribution 2:

> "the set of dates consistent with the same data is usually a single date"

**What was measured** (Experiment 5, gamma = 1, 30 configurations per prefix,
24 restarts and up to 400 iterations per candidate date):

| prefix | unique optimal date | superseded 18-cell figure |
|---|---|---|
| **t = 2** | **13.3%** (4 of 30) | 22.2% |
| **t = 3** | **63.3%** (19 of 30) | 66.7% |
| **t = 4** | **93.3%** (28 of 30) | 100.0% |

By family, the pattern is not uniform:

| distribution | t=2 | t=3 | t=4 |
|---|---|---|---|
| Beta(1,32) | 33.3% | 66.7% | 100% |
| Beta(1,8) | 33.3% | 100% | 100% |
| Beta(1,4) | 0% | 33.3% | 100% |
| bimodal | 0% | 50.0% | 83.3% |
| mixture | 0% | 66.7% | 83.3% |

**The problem.** t = 2 is the paper's headline prefix -- it is the prefix in
Table 4, in Figure 3, and in the Section 5 sentence about distributions agreeing
on the first four moments. At that prefix the reachable optimal date is unique in
**one configuration in eight**. "Usually a single date" is **not supported at
t = 2**. It becomes a majority statement only at t = 3, and never reaches
certainty.

**Direction of the error.** The search is constructive: it can only *under*-count
\|T_t\|. So 13.3% is an **upper bound** on the true uniqueness fraction, and a
more exhaustive search could only push it lower. The claim cannot be rescued by
searching harder.

**The replacement claim.** The right statement is about a *rate*, not a level:

> Decision ambiguity collapses rapidly with prefix length -- from a single date
> in 13% of configurations at t = 2, to 63% at t = 3, to 93% at t = 4 -- even
> though the welfare itself remains partially identified throughout.

**It does not reach zero.** Two configurations are still ambiguous at t = 4, both
in the newly added families: bimodal (T=20, beta=0.002; dates 3 and 4) and
mixture (T=20, beta=0.01; dates 1 and 2).  An earlier version of this note
claimed uniqueness reached 100% at t = 4; that was an artifact of the Beta-only
grid and does not survive the full benchmark.

This is still a defensible and interesting result, for three reasons.

1. **It is a sharper contrast.** Set against Experiment 4's widths (33.9% /
   13.5% / 4.4%), the two rows tell a coordinated story: welfare ambiguity
   shrinks roughly threefold per period while decision ambiguity shrinks faster
   still, from 87% of configurations ambiguous to 7%.
2. **It is falsifiable and dated.** "Usually" is vague; "13% -> 63% -> 93%" is a
   curve someone can reproduce.
3. **It sits naturally with Figure 3.** Figure 3 exhibits a t = 2 pair whose
   optimal dates genuinely differ. Under the old framing that figure is an
   exception to "usually a single date". Under the new framing it is a
   *typical* case at t = 2, and the paper's own illustration stops fighting its
   own summary sentence.

**Where the original intuition partly survives.** The second half of the
Section 5 sentence -- that acting on the wrong member is cheap -- holds for four
of the five families but fails badly for the fifth.

| prefix | ambiguous configs | median worst cost | p90 | max |
|---|---|---|---|---|
| t = 2 | 26 | **5.07%** | 24.52% | **45.63%** |
| t = 3 | 11 | **0.24%** | 2.26% | 4.72% |
| t = 4 | 2 | **0.36%** | 0.64% | 0.70% |

**"Usually cheap" is only true away from the mixture family.** Median worst-case
cost at t = 2, by distribution:

| distribution | median worst cost |
|---|---|
| Beta(1,32) | 2.82% |
| Beta(1,8) | 4.19% |
| bimodal | 4.62% |
| Beta(1,4) | 4.64% |
| **mixture (small high-risk subgroup)** | **24.38%** |

The three most expensive configurations in the entire benchmark are all mixture
cells, reaching 45.63%.  `\RES[-amb]` therefore cannot be filled with a single
reassuring number: the median is 5.07% but the distribution has a long right tail
concentrated entirely in the most application-realistic family.  Report the
median with p90 and max, and state the dependence on family explicitly.

**Mechanism, and why it matters for the paper's framing.**  At these budget
ratios the planner treats the extreme upper tail of risk, while the low-order
moments are dominated by the low-risk bulk.  Two distributions can match the
first four moments and still differ sharply in how much mass sits near p = 0.8 --
and that mass is exactly what gets treated.  So a **small high-risk subgroup is
the worst case for moment-based identification**, in both the value (LP widths
159% vs 13-32% for other families) and the decision (24% vs 3-5% ambiguity cost).

The applications the draft cites -- tenants facing eviction, students at risk of
dropout -- are small-high-risk-subgroup problems.  The family that is most
realistic is the one where identification fails hardest.  That strengthens
Section 5's claim rather than weakening it, but it means the pooled medians in
Table 5 average over two genuinely different regimes and should be reported with
a by-family breakdown.

**Suggested rewrite of the Contribution 2 clause** (not applied; the `.tex` is
not on this machine):

> ~~"the set of dates consistent with the same data is usually a single date"~~
> "the set of dates consistent with the same data narrows quickly with the
> prefix -- unique in 13% of configurations at t=2, 63% at t=3 and 93% at t=4 --
> and where it has not narrowed, committing to the wrong member typically costs a
> few percent of welfare, though up to 46% when the population contains a small
> high-risk subgroup"

**Cross-check.** The LP bounds independently certify 1,030 of 1,350 date-config
pairs as unreachable, and there were **zero** cases where the constructive search
found a date the bounds had excluded. The two methods agree.

Sources: `results/reachable_optimal_dates.csv`,
`results/ambiguity_welfare_cost.csv`, `results/table5.csv`.

---

## 1b. Contribution 4: NPMLE wins, not the proposed rule

**What the draft says** (Contribution 4): "We benchmark the timing rule actually
used in practice against nonparametric, worst-case, and regret-calibrated
alternatives on a common benchmark, and find `[RES]`."

**What was measured** (Table 4, gamma=1, t=2, 30 configurations x 20 seeds,
p10 of welfare efficiency -- the pre-declared primary metric):

| rule | p10, N=1e3 | p10, N=1e5 |
|---|---:|---:|
| Oracle | 1.0000 | 1.0000 |
| **Nonparametric MLE** | **0.9631** | **1.0000** |
| Regret-calibrated (ours) | 0.9253 | 0.9979 |
| Least squares | 0.8671 | 0.9750 |
| Parametric Beta | 0.8535 | 0.8536 |
| Worst-case LP | 0.8502 | 0.8502 |
| Worst-case envelope | 0.8502 | 0.8502 |
| Act immediately | 0.8502 | 0.8502 |

**NPMLE is best on the primary metric at both sample sizes**, and the draft
itself calls it "the baseline any proposed rule has to beat". It is not beaten.
Checked for artifacts: NPMLE is ahead on **all five** distributions at N=1e3,
and in a paired comparison on identical data the regret-calibrated rule wins
only 8.3% of pairs at N=1e3 and 0.8% at N=1e5 (mean paired difference -0.0101
and -0.0020).

**t = 3 and t = 4 are now in, and the point-estimate ordering is unchanged**:
NPMLE is first in all six (N, prefix) columns. But the paired inference
(Experiment 27) shows the ordering is not uniformly resolvable.

### What the paired bootstrap actually supports

Cluster bootstrap over configurations, 2,000 replicates, Holm-corrected within
each (N, prefix) family. Effect = regret-calibrated minus competitor.

**Against NPMLE, on the pre-declared primary metric (p10), the gap is
significant in only 1 of 6 families** -- the single hardest cell, N=1e3 and t=2
(-0.0378, CI [-0.0945, -0.0089], p = 0.006). In the other five the difference is
not distinguishable from zero (p = 0.09 to 1.00).

On the secondary metrics NPMLE's advantage is firmer: significant on MSR in 5 of
6 families and on mean efficiency in 4 of 6. On the median, never.

**Against the other rules the proposed rule wins clearly:**

| comparison | p10 significant | MSR significant |
|---|---|---|
| vs worst-case LP | **6/6** | 6/6 |
| vs worst-case envelope | **6/6** | 6/6 |
| vs act immediately | **6/6** | 6/6 |
| vs parametric Beta | **5/6** | 6/6 |
| vs least squares | 2/6 | 6/6 |
| vs NPMLE | **1/6** | 5/6 |

**The defensible revision:**

> We benchmark eight timing rules. Nonparametric likelihood fitting attains the
> best point estimates throughout, and is significantly better than the proposed
> rule on mean square regret; on the pre-declared lower-tail metric the two are
> statistically indistinguishable in five of six (N, prefix) families. The
> regret-calibrated rule significantly outperforms the parametric plug-in used
> in practice, both worst-case rules, and immediate allocation.

This is a stronger and more honest claim than either "our rule wins" or "NPMLE
wins": the proposed rule is not separable from the best available baseline on
the metric the paper declared primary, and it clearly beats everything else.

**Caveat on the p10 comparisons.** At N=1e5 most runs sit at efficiency exactly
1.0, so the pooled p10 is a coarse, heavily tied statistic and several bootstrap
CIs terminate exactly at 0.0000. The non-significance against NPMLE at large N
partly reflects that ceiling rather than genuine equivalence. MSR, which uses the
whole distribution, is the more discriminating metric there and does separate
them.

### Three findings that do support the draft

**Worst-case rules collapse to immediate allocation.** Both are identical to
act-immediately to four decimals in three of four columns; the LP variant differs
only slightly at N=1e5. They forfeit the entire value of waiting.

*Wording note:* say **"collapse almost entirely to immediate allocation,
forfeiting the value of waiting"**, not "actively harmful". Relative to
act-immediately they are non-beneficial, not worse -- the envelope is exactly
identical and the LP variant is marginally better in one column. They are
harmful only relative to the data-adaptive rules and the oracle, and the looser
phrase invites the misreading that they lose to acting immediately.

**Parametric Beta fails on 12.5% of runs at N=1e3** (75/600), every one
`nonpositive_variance` -- exactly the draft's "the moment problem it solves can
be infeasible on realized data". Falls to 0.5% at N=1e5.

**The lower tail is where the action is.** Parametric Beta holds a median of
0.9964 while its p10 is 0.8535. That is the Table 4 caption's own argument for
bolding p10, measured on the paper's own baseline.

---

## 2. Figure 3's caption numbers

Recorded in full in `experiments/reports/FIGURE3_DISCREPANCY.md`. Summary:
optima are **2 periods apart, not 1**; the welfare gap is **51.5%, not 34%**;
and committing to the wrong member costs **11.5%**, not "at most a few percent".
Stable across five grid resolutions. The qualitative claim survives and is
stronger; two caption sentences do not.

Note this now *coheres* with item 1: an 11.5% cost sits comfortably inside the
Experiment 6 distribution at t = 2 (median 5.07%, p90 24.52%, max 45.63%), so
Figure 3 is not an outlier -- it is an ordinary case.

---

## 3. Lemma 1 at gamma != 1: the rate is gamma+1, not 2

Section 9 states the general case is open. Experiment 1 measures it for integer
gamma: the cohort span grows by **gamma+1 dimensions per period** -- 2t at
gamma=1 (exactly Lemma 1), 3t-1 at gamma=2, and 5 per period at gamma=4. This is
a measurement over three values of gamma, not a theorem, and the draft's "we
regard the general case as open" should stand. But the paper can state the
measured rate rather than leaving the reader with no information at all.

See `experiments/reports/RANK_CONDITIONING.md` for the accompanying caveat that
the floating-point rank undercounts at high degree for conditioning reasons, and
that the exact rank is the one to read.

---

## 4. Items still unresolved that touch the prose

- **`\RES[-instab]`** (D5): the draft says "the headline *accuracy* of a rule
  moved by a factor of ...", but Section 7 elsewhere demotes exact-argmax
  agreement to "a secondary diagnostic only". If the instability figure refers to
  argmax accuracy, the sentence advertises the instability of a metric the paper
  says it does not care about. Worth confirming which quantity was meant.
- **Section 6 forward reference** (D1): "Section 7 specifies the estimator and
  its tuning" is unfulfilled -- Section 7 never defines the regret-calibrated
  rule. This needs fixing in the draft regardless of what the rule turns out to
  be.

---

## 5. Experiment 16 answers the draft's Question 8 outright

**Question 8:** "Does the parametric plug-in fail because of Beta
misspecification, because of estimation, or because plug-in decision-making
itself is fragile?"

Holding the decision template **exactly fixed** and varying only the estimator
(p10 welfare efficiency):

| N | prefix | Parametric Beta | Least squares | NPMLE |
|---|---|---:|---:|---:|
| 1e3 | 2 | 0.8535 | 0.8671 | **0.9631** |
| 1e3 | 3 | 0.8536 | 0.8683 | **0.9605** |
| 1e3 | 4 | 0.8536 | 0.8683 | **0.9750** |
| 1e5 | 2 | 0.8536 | 0.9750 | **1.0000** |
| 1e5 | 3 | 0.8536 | 0.9907 | **1.0000** |
| 1e5 | 4 | 0.8536 | 0.9969 | **1.0000** |

**The answer is the estimator, not the template.** The plug-in decision
principle is not fragile at all -- the same template with an NPMLE estimator is
the best rule in the entire benchmark. Parametric Beta is flat at 0.8536 in all
six cells: neither more data nor a longer prefix helps it. Least squares sits
between and converges toward NPMLE as N grows.

This is a cleaner answer than the draft anticipates, and it argues against
framing plug-in rules as a class that fails.

## 6. Experiment 28: where the failures actually are

| rule | failures | rate | at N=1e3 | at N=1e5 | reasons |
|---|---:|---:|---:|---:|---|
| Parametric Beta | 241 / 3600 | 6.7% | **12.2%** | 1.2% | `nonpositive_variance` 240, `degenerate_beta_density` 1 |
| Worst-case envelope | 5 / 3600 | 0.14% | 0.22% | 0.06% | identified set empty |
| Worst-case LP | 5 / 3600 | 0.14% | 0.22% | 0.06% | identified set empty |
| all others | 0 | 0% | 0% | 0% | -- |

Every parametric-Beta failure is the method-of-moments system having no Beta
solution, exactly as the draft predicts. No other rule fails materially.

## 7. Experiment 29: the worst-case rules quantified

Mean true `t*` is 2.9. Selected dates at N=1e3, t=2:

| rule | mean date | acts before t* | exactly t* | after t* |
|---|---:|---:|---:|---:|
| Oracle | 2.90 | 0% | 100% | 0% |
| NPMLE | 2.43 | 32% | **62%** | 6% |
| Regret-calibrated | 2.40 | 38% | 48% | **14%** |
| Parametric Beta | 2.21 | 50% | 41% | 9% |
| Least squares | 2.04 | 52% | 39% | 10% |
| Act immediately | 1.00 | **93%** | 7% | 0% |
| Worst-case envelope | 1.00 | **93%** | 7% | 0% |
| Worst-case LP | 1.00 | **93%** | 7% | 0% |

The worst-case rules act before the optimal date on **93%** of runs -- the
precise version of "collapse almost entirely to immediate allocation,
forfeiting the value of waiting".

The regret-calibrated rule is the only one that overshoots appreciably (14%
after `t*`), which is the randomisation doing what it is designed to do: it
hedges rather than committing, and sometimes waits one period too long.

## 8. Experiment 30: the rules only separate where timing matters

p10 welfare efficiency by stake bin, N=1e3 (bins fixed in advance from the
stakes distribution, never from rule performance):

| stakes | act now | Beta | NNLS | regret-cal. | NPMLE |
|---|---:|---:|---:|---:|---:|
| <2% | **0.997** | 0.935 | 0.992 | 0.992 | 0.992 |
| 2-5% | 0.955 | 0.963 | 0.963 | 0.963 | **0.996** |
| 5-10% | 0.921 | 0.958 | 0.947 | 0.991 | **0.993** |
| 10-20% | 0.813 | 0.814 | 0.820 | 0.890 | **0.964** |

Two things worth stating in the paper:

1. **The spread widens monotonically with stakes.** At <2% every rule is within
   half a point of every other; at 10-20% the gap between acting immediately and
   NPMLE is 15 points. The benchmark discriminates only where timing is worth
   something, which is exactly why the draft asks for stratification by stakes.
2. **Where stakes are negligible, acting immediately is the best rule** (0.997,
   ahead of NPMLE's 0.992). Any rule that sometimes waits pays a small price for
   nothing. This is not a defect of the sophisticated rules; it is the correct
   behaviour of a benchmark that includes configurations where the decision does
   not matter, and it should be reported rather than averaged away.
