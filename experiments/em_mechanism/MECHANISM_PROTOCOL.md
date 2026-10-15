# Mechanism follow-up: what does continued EM change, and does that change cause the worse date?

Frozen 2026-10-06 before any mechanism quantity was computed.  Uses only the
completed trajectories of the main study (no new data).  Known beforehand:
the aggregate results in `results/em_mechanism/REPORT.md` and the event
counts below.

## Events

Reference datasets, uniform start, tau = 2, G = 200 (the shared benchmark
datasets).  For each dataset compare EM55 (`w55`) with EM at the 1e6 cap
(`wcap`), both saved as weight snapshots.

| class | definition | N = 1e3 | N = 1e5 |
|---|---|---|---|
| worse (primary) | regret_P(cap) > regret_P(55) | 660 | 110 |
| improve (control) | regret_P(cap) < regret_P(55) | 32 | 150 |
| same | equal regret | 808 | 1,240 |

Local switch: `k*` = first iteration after 55 at which the selected date
differs from EM55's, found exactly by re-running EM from the saved `w55`
snapshot with a date check at every iteration inside the recorded bracket;
the pair `(w_{k*-1}, w_{k*})` is analysed alongside `(w55, wcap)`.

## Step 1: what changes (descriptive)

(a) mass shift per risk bin, bins [0,.02), [.02,.05), [.05,.1), [.1,.2),
[.2,.4), [.4,1]; (b) raw moments m1..m10, identified (j <= 3) versus
unidentified (j >= 4); (c) fitted observation probabilities: max change, and
KL to the empirical cell frequencies and to the true cell probabilities, at 55
and at the cap; (d) atomicity: effective support exp(entropy) and the number of
grid points holding 99% of mass.

## Step 2: how the change moves the decision

Contrast `C(w) = [W^{d_cap}(w) - W^{d_55}(w)] / W*_P` (estimated welfare,
normalized by the true optimum; `C(w55) <= 0 <= C(wcap)` by construction).
`C(wcap) - C(w55)` is attributed to risk bins by a path integral of the
envelope gradient of `C` along the straight segment w55 -> wcap (400
midpoint steps; residual reported).  Separately, the change is split by the
observable-matched projection `w' = argmin ||w - w55||_1 s.t. Psi w = Psi wcap`
into an observable-driven part `C(w') - C(w55)` and an unidentified part
`C(wcap) - C(w')`.  Step 2 uses estimated welfare only, never the truth.

## Step 3: constraint tests (each keeps the late estimate's fit exactly)

Every constrained estimate satisfies `Psi w = Psi wcap`: the same likelihood
and KKT residual as the late estimate, so "more fitting" is fully retained.

* T1 minimal change: `min ||w - w55||_1`.
* T2 freeze unidentified moments: `min ||w - wcap||_1` with `m_j(w) = m_j(w55)`,
  j = 4..6 (and j = 4..8 as a variant); infeasibility is reported.
* T3 freeze decision bins: `min ||w - wcap||_1` with the masses of the two bins
  carrying the largest pooled |step-2 attribution| over worse events (chosen
  per N from estimated welfare only) fixed at their w55 values.
* T4 negative control: as T3, for the two bins with the smallest pooled
  |attribution|.
* T5 sufficiency: `min ||w - w55||_1` with `Psi w = Psi w55` (EM55's fit) and
  the T3 bin masses set to their wcap values: does importing only that change
  into EM55 reproduce the late date?

Outcomes per test: share of events whose selected date returns to EM55's
(T1-T4) or moves to the late date (T5); mean true regret under P (and both
pair members) of the constrained estimate versus EM55 and the cap; LP
feasibility.  Tests are applied to worse events (primary) and improve events
(control: a mechanism-specific constraint should not remove only
deterioration).  Exact-probability trajectories (tau = 2, G = 200, all
initializations, 55 -> cap where the date changes) are analysed with
steps 1-2 as the noise-free comparison.

A mechanism is called supported only if (i) step 2 locates the contrast
change, (ii) T3 (and/or T2) restores EM55's decision in most worse events
while T4 does not, and (iii) T5 reproduces the late date.
