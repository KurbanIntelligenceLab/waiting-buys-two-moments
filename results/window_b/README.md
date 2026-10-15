# Window-B artifacts (superseded by the R15 correction)

> **Note (2026-09-26):** the tag `window-b-frozen` referenced below was never pushed to the shared repository. The same pre-correction tree is commit `197fd9e9`; substitute it for the tag in the commands below (e.g. `git show 197fd9e9:results/bagged_full_b200_raw.csv`).

The retained result files in this directory preserve the **Window-B**
observation model for comparison. They are historical evidence, not current
Window-A benchmark outputs. The corrected results used by the manuscript's
main tables live under their original names in `results/`.

## Duplicate cleanup (2026-10-09)

47 files here were byte-identical to copies directly under `results/` and
were removed, eliminating 39,223,135 bytes of duplicate files in a checkout.
No distinct Window-B result was removed. The release inputs and window-audit
controls remain here, including the earlier bagging and benchmark raw runs.

`DEDUPLICATED_FILES.tsv` records each removed path, its retained path, size,
and SHA-256. Retained paths refer to the full repository; not all those
historical files are included in the smaller supplementary archive.
Those hashes identify the historical contents; a retained root
file is not automatically a current publication result. To recover the exact
pre-cleanup file if needed, use `git show 4f4d9b9d:<removed_path>`.

## `certified_benchmark_raw.csv` — one exception, now closed

That statement was not true of this file until 2026-09-18. A byte-identical
Window-B copy of `certified_benchmark_raw.csv` survived at the top level of
`results/` after the R15 correction, under the canonical name, carrying
`n_record_atoms = 19` at $\tau=4$ where Window A has 14. It was never
regenerated because nothing reads it: `experiments/scripts/rerun_certified.py`
only *writes* that name, and no analysis script or manuscript table consumes
it. Regenerating it would have cost 292 core-hours to no purpose.

The top-level copy has been removed. This archived copy is the only one, and
it was verified identical (sha256 `230f76ff…`) before removal; git history
retains the deleted path regardless. If `rerun_certified.py` is ever run again
on a Window-A branch, the file it writes at the top level will be Window A,
which is correct — but anything comparing against the version here must treat
it as Window B.

**This file must not be used for the manuscript's Window-A results.**

## What the two windows are

For a length-$t$ prefix, an individual's record is:

| | exits recorded at | terminal category | kernel degree | exact rank |
|---|---|---|---|---|
| **Window A** (paper, current) | end of periods $1..t-1$ | active *in* period $t$, kernel $\phi_{t,k}$ | $\le 2t-1$ | $2t$ |
| **Window B** (superseded) | end of periods $1..t$ | survived the period-$t$ exit, kernel $\phi_{t,k}(1-p)$ | $2t$ | $2t+1$ |

Window B lets the planner see who fails *at* date $t$. That failure is only
visible at $t+1$, so it is outside the length-$t$ window. Including it
identifies $m_{2t}$ — one moment beyond the $\Pi_{2t-1}$ that Lemma 1 of the
paper establishes.

## Why it mattered

Under Window B the paper's moment-matched extreme pairs were **not**
observationally equivalent. For the headline pair (Beta(1,32), $T=20$,
$\beta=0.01$, $t=2$), matched on $m_0..m_3$:

| | max record-cell discrepancy | KL |
|---|---|---|
| Window A | $2.7\times10^{-15}$ | $1.7\times10^{-17}$ (zero, as Theorem 1 requires) |
| Window B | $9.0\times10^{-5}$ | $1.17\times10^{-5}$ |

Under Window B that pair separates at $N \approx 90{,}000$ — inside the
benchmark's own grid, whose largest cell is $N=10^5$. Theorem 1's "for every
population size $N$" therefore did not hold of the implemented pipeline.

The theory was never wrong: Lemma 1, its proof, and `rem:exit-records` all
describe Window A consistently. Only the implementation diverged.

## What is and is not in here

Distinct files are retained here for studies whose computation flows through
the NPMLE record likelihood (`observation.individual_functionals` →
`npmle.record_matrix`) or through `regret_calibrated.record_counts_from_prefix`.
Shared byte-identical files are listed in `DEDUPLICATED_FILES.tsv` instead.

Absent by choice: the bootstrap trajectory caches
(`bagged_full_b200_trajectories/` 90 MB,
`bagging_control_expanded_trajectories/`, `bagging_control_72_trajectories/`).
They are large intermediate binaries, not reported results.

The complete pre-correction tree, including trajectory caches, is preserved
in git at commit `197fd9e9`. The tag named in older reports was never pushed;
use the commit directly:

```
git show 197fd9e9:results/bagged_full_b200_raw.csv
```

## Unaffected by the correction

Results derived from cohort counts, moment recovery, or the identified-set LPs
do **not** depend on the window and were not recomputed: the extreme-pair
construction, `identification_welfare_bounds.csv`, the reachable-date search,
the two-point lower bound, the specification tests, the worst-case LP pilots,
and the parametric-Beta / least-squares / oracle / act-immediately rules.

See `results/r15_window_correction_report.md` for the full dependency map and
the before/after comparison.

## The run behind the manuscript's Appendix C.2 (recovered)

The manuscript's C.2 retention counts:

> 1 skipped; 7,199 ensembles; 6,792 retain all 200 draws and 407 retain
> between 194 and 199; 519 of 1,439,800 fits discarded (0.036%)

They are computed from `bagged_full_b200_raw.csv` in this directory
(`release/verify_paper.py`, artifact `appendix_c2`). This is the
pre-correction run.

- **Implementation:** `experiments/scripts/bagged_full_b200.py` at commit
  `197fd9e9`.
- **Output:** that commit's `results/bagged_full_b200_raw.csv` is
  byte-identical to this file (sha256 `d0b71337…`).

To regenerate it (about 7.3 core-hours; not rerun during the 2026-10-02
audit):

```
git archive 197fd9e9 | tar -x -C /tmp/window_b_tree
cd /tmp/window_b_tree && python experiments/scripts/bagged_full_b200.py
sha256sum results/bagged_full_b200_raw.csv   # compare with this directory's copy
```

Table 3 uses the later Window-A run (`results/bagged_full_b200_raw.csv`).
