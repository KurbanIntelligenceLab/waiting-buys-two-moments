# EM-mechanism study: frozen protocol

Frozen 2026-10-05, before any main-stage (non-pilot) result was computed.
Pilot outputs (`results/em_mechanism/pilot_*`) were used only to measure
runtime and to check the code paths; they are not part of the evidence.

## Question

Why does EM stopped after 30/55/80 iterations select better allocation dates
than the certified NPMLE on the frozen benchmark?  Candidate explanations,
treated as hypotheses: (1) finite-sample noise/overfitting; (2)
initialization-dependent implicit regularization; (3) selection among
observationally (nearly) compatible distributions with different welfare;
(4) numerical/grid effects or implementation errors; (5) favorable alignment
with the selected reference populations.

## Fixed design

| Item | Setting |
|---|---|
| Observation window | common pre-failure window (Window A): exits at the ends of periods 1..tau-1 and the period-tau active cohort by y^tau; never period-tau failures. This is the window of the current code (`individual_functionals`, `record_counts_from_prefix`) and of the archived EM/NPMLE results. |
| Likelihood, grid | `wtm.npmle.record_matrix`, uniform grid `linspace(0,1,G)`, G = 200 (refinement: 400, 800) |
| EM update | `wtm.npmle.fit_npmle`'s update, batched (`core.run_em`); iteration k = state after k updates, k = 0 is the initialization; weights floored at 1e-300 |
| Records | every iteration 0..100, then 20 log-spaced points per decade, plus fixed checkpoints, to the cap |
| Caps | finite samples 1e6; exact probabilities 1e7 (tau=2, G=200), 1e6 otherwise |
| NPMLE | `fit_npmle_certified`, delta = 1e-8, existing validity rule (support <= #cells, finite); status certified / valid_uncertified / invalid |
| EM status | certified (final KKT <= 1e-8), stagnated (no measurable likelihood gain over the last decade), cap_reached, numerical_failure |
| Decision | earliest argmax of the estimated welfare curve over s = 1..T-1 (existing rule; near-ties re-decided with `_trajectory`) |
| Initializations | uniform; low Beta(1,19); middle Beta(4,4); high Beta(5,2); rand1-4 Dirichlet(1) seeds 20261001-4; non-uniform mixed with uniform at eps = 0.01 |
| Populations | the 30 gamma=1 configurations; exact compatible pairs from `extreme_pair_exact.py` (tau = 2, 3, 4) |
| Finite samples | tau = 2, N in {1e3, 1e5}, seeds 1001..1050 per (configuration, N); ref datasets use the archive's seed policy (1001..1020 are the archived datasets), pair datasets reconstruction-v1's (1001..1005 are its datasets) |
| Grid refinement (finite) | G in {400, 800} vs 200; uniform init; ref datasets; N in {1e3, 1e5}; seeds 1001..1020; configurations: every family at T = 20, beta = 0.002, plus Beta(1,32), T = 20, beta = 0.01 (added 2026-10-05, before main results) |
| Figure representatives | every family at T = 20, beta = 0.002 (Fig. 1); the manuscript's headline pair Beta(1,32), T = 20, beta = 0.01 (Fig. 3) |
| Label-free comparator | 2-fold CV EM (uniform init): earliest iteration maximizing held-out log likelihood |

No stopping count, initialization or setting is chosen using true welfare.
"Hindsight-best initialization" appears only as a labelled diagnostic.

## Prespecified headline comparisons (paired, shared datasets)

H1. Exact probabilities: regret of EM_k versus the converged endpoint (cap)
and versus the NPMLE, per family and configuration, for each initialization.
Descriptive (no sampling uncertainty exists here).

H2. Finite samples, ref datasets: MSR(EM55, uniform) - MSR(NPMLE) at each N;
and the same for EM30, EM80, EM at cap, and EM55 averaged over the seven
non-uniform initializations.  Paired over shared datasets; 95% cluster
bootstrap over configurations (2,000 replicates; datasets resampled within
configuration).  Intervals are conditional on this fixed benchmark.

H3. Compatible pairs (pair datasets): pooled worst-member mean regret,
max over members of the trial-mean, then averaged over 30 pairs (the
manuscript's rho), and separately the mean of trialwise maximum regret, for
EM30/55/80/cap and NPMLE, per initialization; paired difference EM55 - NPMLE
with a bootstrap resampling trials within pair (2,000 replicates).

Secondary/descriptive: tenth-percentile efficiency, mean relative regret,
decision-change frequency (vs NPMLE, and along the path), status counts,
runtime, grid refinement, tau = 3, 4 population trajectories, CV stopping.
