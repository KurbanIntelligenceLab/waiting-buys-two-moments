# R15 resolved: the implemented observation window now matches the paper

> **Note (2026-09-26):** the tag `window-b-frozen` referenced below was never pushed to the shared repository. The same pre-correction tree is commit `197fd9e9`; substitute it for the tag in the commands below (e.g. `git show 197fd9e9:results/bagged_full_b200_raw.csv`).

**Status: code corrected, tests passing, all recomputation complete.**
Resolves the open item recorded in `README.md` ("R15, author action required")
and in `results/fig1_teaser_and_exit_record_degree_report.md`.

## The defect

The paper and the implementation described two different experiments, using the
same phrase ("exit record") for both.

For a length-$t$ prefix:

| | exits recorded at | terminal category | kernel | degree | exact rank |
|---|---|---|---|---|---|
| **Window A** — paper | end of periods $1..t-1$ | active *in* period $t$ | $\phi_{t,k}$ | $\le 2t-1$ | $2t$ |
| **Window B** — old code | end of periods $1..t$ | survived the period-$t$ exit | $\phi_{t,k}\,(1-p)$ | $2t$ | $2t+1$ |

Window B lets the planner observe who fails *at* date $t$. That failure is only
visible at $t+1$, so it lies outside a length-$t$ window. Observing it
identifies $m_{2t}$ — one moment beyond the $\Pi_{2t-1}$ that Lemma 1
establishes.

The theory was never wrong. Lemma 1, its proof, and `rem:exit-records` describe
Window A consistently, and the proof of Theorem 1 invokes Window A explicitly
("by the exit-record clause of Lemma 1"). Only
`observation.individual_functionals` and its empirical twin
`regret_calibrated.record_counts_from_prefix` diverged.

## Evidence that it was real

**Exact rank over $\mathbb{Q}$**, $t=1..6$:

| $t$ | cohort counts | Window A | Window B |
|---|---|---|---|
| 1–6 | 2, 4, 6, 8, 10, 12 | 2, 4, 6, 8, 10, 12 | **3, 5, 7, 9, 11, 13** |

The project's own `results/structural_rank.csv` had already recorded this: its
hardcoded `expected_rank_if_gamma1` for `individual_records` was literally
$2t+1$, and every row matched. The expectation was encoding the bug.

**Consequence for Theorem 1.** For the headline pair (Beta(1,32), $T=20$,
$\beta=0.01$, $t=2$), matched on $m_0..m_3$ to $\sim10^{-15}$ and differing on
$m_4$ by $4.5\times10^{-5}$:

| | max record-cell discrepancy | KL | separable at |
|---|---|---|---|
| Window A | $2.7\times10^{-15}$ | $1.7\times10^{-17}$ | never (exact equality) |
| Window B | $9.0\times10^{-5}$ | $1.17\times10^{-5}$ | $N \approx 90{,}000$ |

The benchmark's largest cell is $N=10^5$. Theorem 1's "for every population
size $N$" did not hold of the implemented pipeline, inside the grid that was
actually run.

## The fix

Two definition sites, changed together:

- `wtm/observation.py::individual_functionals` — exits now enumerate
  `range(1, t)` instead of `range(1, t+1)`; the terminal atom is `phi(t, k)`
  instead of `phi_survivor(t, k)`, and is keyed `("active", t, k)` rather than
  `("alive", t, k)` so cached Window-B artifacts cannot be silently mixed in.
- `wtm/regret_calibrated.py::record_counts_from_prefix` — the terminal category
  is now `active_mask[t-1]` (the pool active at the *start* of period $t$)
  rather than `exited_by == -1` (which conditions on surviving the period-$t$
  exit).

`wtm/simulate.py` is deliberately **unchanged**, so every `(config, seed)` pair
still reproduces bit-for-bit. Only what is *extracted* from the simulation
changed, not what is drawn.

`experiments/scripts/structural.py` had its hardcoded expectation corrected
from $2t+1$ to $2t$.

`phi_survivor` is retained — it is a well-defined kernel, still used by
`figure3.py` and the validation suite — but is documented as not part of the
length-$t$ window.

## Tests

`experiments/tests/test_observation_window.py`, 9 tests, all passing. Seven of
them were **verified to fail** against the restored Window-B model rather than
merely assumed to: degree bound, exact rank $2t$, span equality with the cohort
counts, terminal-kernel identity, exit-range, moment-matched-pair equivalence,
and the period-$t$-exit behavioural check.

The remaining two are well-formedness invariants and are labelled as such in the
file: kernels summing to 1 (true of both windows), and count/kernel alignment
(deliberately window-agnostic — it catches the two definition sites drifting
apart, which was verified by patching one side only).

Full suite after the change: **47/47 passing** (9 new + 14 validation + 12 NPMLE
projection + 12 regret rule).

A pre-existing `RuntimeWarning: overflow encountered in matmul` in
`npmle.py:260,301` fires identically under both windows (18 warnings across the
same 6 cells) and is therefore unrelated to this correction. It is noted here
as a separate open item, not fixed as part of R15.

## Dependency map

**Affected** — everything flowing through the NPMLE record likelihood
(`individual_functionals` → `npmle.record_matrix`) or through
`record_counts_from_prefix`:

| Module | Role |
|---|---|
| `wtm/npmle.py` | `record_matrix`, `fit_npmle`, `fit_npmle_certified` |
| `wtm/regret_calibrated.py` | record extraction, the v3 rule |
| `wtm/plugin_rules.py` | the nonparametric plug-in rule |

19 experiment scripts fit an NPMLE. Those feeding the current paper:
`rerun_certified.py`, `holdout_v3.py`, `bagged_full_b200.py`,
`bagging_control_expanded.py`, `bagging_control_72.py`,
`early_stopped_npmle.py`, `early_stopped_robustness.py`,
`realdata_certified_v3.py`, `npmle_ablation.py`, `certified_gate.py`,
plus the downstream analyses `final_inference_recompute.py`,
`final_complete_case_and_date_error.py`, `paired_inference.py`, and the
cheap `structural.py` / `fig1_teaser.py`.

Superseded and **not** scheduled for rerun: `regret_rule_ablation.py` (v1,
192.4 core-h), the `regret_v2` sweep (~111 core-h), `regret_v2_instrument.py`,
`em_regularization_path.py`, `solver_equivalence_gate.py` (warm-start gate,
rejected), `certified_npmle_study.py` (pre-check), `misspecification.py` (not
cited by the current draft), `diagnose_regret_rule.py`, `regret_v2_eval.py`,
`embodied_run.py`.

**Unaffected** — derived from cohort counts, moment recovery, or the
identified-set LPs, none of which touch the record law:

- `wtm/moment_cone.py` (uses `cohort_functionals`), `wtm/lp.py`,
  `wtm/welfare.py`, `wtm/worst_case.py`, `wtm/search.py`
- the extreme-pair construction at $t=2,3,4$ and every number it produces:
  welfare spreads 42.1 / 27.3 / 15.2 %, $\zeta=12.69\%$, the $6.34\%$ floor,
  the 29-of-30 disagreement count, Table `tab:idvsest`
- `identification_welfare_bounds.csv` (72.8 core-h), the reachable-date search,
  the two-point lower-bound frontier, the specification tests, the worst-case
  LP pilots
- the parametric-Beta, least-squares, oracle and act-immediately rules

This is why the identification results and Theorem 1's numerical witnesses
survive the correction untouched: they were always computed from moments, which
is Window A's span by construction.

## Preservation

Nothing was overwritten or deleted.

- Annotated git tag **`window-b-frozen`** holds the complete pre-correction
  tree, bootstrap trajectory caches included.
- `results/window_b/` holds 59 copied artifacts (65 MB) for side-by-side
  comparison, with its own `README.md`. Excluded from the copy, and available
  from the tag: `bagged_full_b200_trajectories/` (90 MB) and the two smaller
  trajectory caches.

---

# Recomputation results

All staged reruns complete. **7.7 measured core-hours total**, against the
~475 the historical inventory implied for these experiments — the gap is that
the historical figure was dominated by the superseded v1/v2 studentized-walk
rules and EM-era fits, while the certified support-reduction solver runs at
about 1 ms per fit.

| Stage | Wall | Core-h |
|---|---|---|
| structural ranks | 0.3 s | ~0 |
| b200 benchmark (7,200 runs, B=200) | 1.83 h | 7.27 |
| early-stopped EM (55) | 11 s | 0.009 |
| early-stopping robustness (30/55/80) | 32 s | 0.03 |
| real-data (NELS:88, ELS:2002) | 8.1 min | 0.46 |
| downstream inference / complete-case / date-error | seconds | ~0 |

## What held

**Every conclusion the paper rests on survives, and two get stronger.**

| | Window B | Window A |
|---|---|---|
| RC == bagging agreement | 99.28% | 99.16% |
| Bagging beats RC on pooled MSR | 0.001045 vs 0.001078 | **0.001371 vs 0.001491** |
| Mean regret, RC − bagging | +0.000172 | **+0.000732** |
| p10 significant, RC vs certified | 0 of 12 | **0 of 12** |
| MSR significant, RC vs certified | 10 of 12 | **10 of 12** |
| Early stopping vs certified, MSR | 44.5× | **42.3×** |

The pre-declared primary endpoint is unchanged: no significant p10 difference
in any of the 12 settings. Bagging's MSR advantage over RC widened, and the
mean-regret point estimate against RC grew roughly fourfold.

The six window-independent rules are **bit-identical** to their Window-B
values across all 43,200 rows, confirming the dependency map: the correction
reaches only the NPMLE record likelihood.

## What changed

Every magnitude degrades, which is what removing a moment of planner
information predicts.

| Quantity | Window B | Window A |
|---|---|---|
| Certified NPMLE picks $t^\star$ | 68.37% | **60.98%** |
| Certified NPMLE p10 | 0.9631 | **0.9475** |
| Certified NPMLE MSR | 0.001571 | **0.002117** |
| Single fit == bagging | 87.08% | 84.58% |
| RC/bagging date disagreements | 52 | 60 |
| Exact-date accuracy significant | 7 of 12 | **4 of 12** |
| Absolute date error significant | 9 of 12 | **6 of 12** |
| p10 point estimate favours RC | 10 of 12 | **12 of 12** |

Early stopping is barely touched (p10 0.9979 → 0.9969) while the certified
comparator degrades sharply. The regularized estimator is robust to losing a
moment; the fully optimized one is not. That strengthens the paper's reading
of the early-stopping effect rather than weakening it.

Real data:

| | Window B | Window A |
|---|---|---|
| NELS:88 p10, certified / RC | 0.9864 / 0.9967 | 0.9864 / 0.9967 |
| ELS:2002 p10, certified / RC | 0.9951 / 0.9951 | **0.9913** / 0.9951 |
| MSR ratio certified/RC, NELS | 5.61× | 4.15× |
| MSR ratio certified/RC, ELS | 5.39× | 7.94× |

## Manuscript edits this forces

Numbers, Section 6 and the abstract:

1. `99.28%` → `99.16%` (RC/bagging agreement), and `7,147 of 7,199` →
   `7,087 of 7,147`.
2. `68.37%` → `60.98%`, and `4,922 of 7,199` → the corrected count.
3. Pooled mean-regret difference `+0.000172` → `+0.000732`; its CI and
   $p$-value need recomputing.
4. `52` date disagreements → `60`; the 20-better/32-worse split →
   14-better/46-worse, mean efficiency difference `-0.024` → `-0.087`.
5. Early stopping "roughly 45-fold" → "roughly 42-fold"; p10
   `0.9979` → `0.9969`; MSR `0.000035` → `0.000050`; certified p10
   `0.9631` → `0.9475` and MSR `0.00157` → `0.00212`.
6. "significant in 7 of 12" (exact-date) → 4 of 12; "significantly lower in 9"
   (absolute date error) → 6 of 12.
7. "The p10 point estimate favors RC in ten settings and ties in two" → favours
   RC in all 12.

Prose:

8. Section 6: "Mean efficiency peaks near 55 iterations, while p10 continues to
   improve through 80." Mean efficiency still peaks at 55, but p10 now improves
   from 30 to 55 and then **plateaus** (0.9969 at both 55 and 80) rather than
   rising to 0.9985.
9. Appendix `tab:cohort-calibration`: "The ELS:2002 p10 values agree at the
   displayed precision" is no longer true (0.9913 vs 0.9951).
10. "MSR is approximately five times lower for RC" → 4.15× (NELS) and 7.94×
    (ELS); state them separately rather than as one figure.
11. The missing-record caveat can be **dropped**. The corrected rerun produced
    a complete 7,200-run grid; the absent record at $N=10^5$, $t=2$ does not
    recur. Every "7,199" becomes "7,200".
12. Compute accounting: the inventory totals need revisiting, since these
    experiments now cost 7.7 core-hours rather than the figures carried in
    `COMPUTE_INVENTORY.md`.

## Open items not closed here

- **The `n_support > n_atoms` guard.** 53 runs (49 at $t=2$) were rejected
  where Window B rejected 1. They are not bad fits — worst KKT certificate
  6.4e-7 — but non-minimal support representations, and the guard was
  implicitly calibrated to Window B's larger atom count. Including them moves
  nothing beyond the fourth decimal, so no conclusion depends on the choice,
  but changing a pre-registered rejection rule is an author decision. The
  recomputed rows are in `results/r15_recovered_skipped_runs.csv`.
- **Pre-existing overflow warnings** in `npmle.py:260,301`, identical under
  both windows (18 warnings, same 6 cells). Unrelated to R15.
- **Figure 1** was built from the old window and should be regenerated.
