# Search for the original Appendix D.3 experiment (2026-10-02)

Goal: recover the generator, raw outputs, seed derivation, bootstrap RNG
sequence, pair definitions and analysis code behind Tables 6–7 of the
2026-09-30 PDF.

## Searched

| Location | Method | Result |
|---|---|---|
| All 7 remote branches (`main`, `claude/*`) and 2 pull-request refs (`refs/pull/1`, `refs/pull/2`) | `git fetch --all`; `git ls-remote origin` | no D.3 code or output |
| Full commit history, all refs | `git log --all -S` for `worst_member`, `worst-member`, `pair_risk`, `0.0648`, `137/150`, `tab:matched`, `Table 6`; `git log --all -G "both members\|compatible pair"` | only this reconstruction's commit (`3cec99b5`), plus unrelated pair studies: `klq_pair_arm` (withdrawn two-date arm), `matched_pair_construct` (smooth/atomic twins), `extreme_pair_*` (Table 1) |
| Unreachable objects | `git fsck --lost-found --no-reflogs` | 2 dangling objects, both from this session's own packaging work |
| Stashes, reflog | `git stash list`, `git reflog` | none relevant |
| Working tree and `results/` | grep for the PDF's Table 6/7 values (`0.0648`, `0.0357`, `0.0558`, `0.1121`, `-0.0362`, `0.0021`) | coincidental hits only, in unrelated numeric columns |
| Other repositories the account can reach | repository listing | none related to this project |
| Files uploaded for this work | listing | only the 2026-09-30 manuscript PDF |
| Claude Code sessions on the authors' account (2026-10-02, third pass) | session listing and metadata | **Lead, not recovered.** A local Claude Code CLI session titled "RC" (on the author's machine; identifiers in the repository-only `d3_retrieval_guide.md`) last reported, on 2026-09-26 11:22 UTC, "Full experiment run completed successfully (300/300)". 300 equals D.3## Conclusion

The implementation and raw outputs behind the manuscript's Appendix D.3 and
Tables 6–7 are not in any source reachable from this environment. They were
never committed to this repository: `matched_info_pairs.py` was first written
in this audit, on 2026-10-02.

**Missing artifacts:**
1. the D.3 generator: pair construction, data simulation, the ten procedures
   at the D.3 settings, and the seed derivation;
2. the raw per-trial outputs: 300 datasets × 10 procedures, with validity
   flags;
3. the analysis that builds Tables 6–7: joint-validity filter, pooled
   worst-member risk, 2,000-replicate paired bootstrap;
4. the bootstrap seed and stream of that analysis.

**Most likely location:** the author's local machine (Windows 11 laptop), where
the Claude Code session "RC" reported a 300/300 run on
2026-09-26. Session, machine, candidate paths and retrieval
steps: `d3_retrieval_guide.md`.

**Status in the manuscript.** The manuscript's D.3 text and Tables 6–7 are
authoritative and unchanged. Their status in `release/paper_artifacts.yaml` is
`unrecovered`.

**reconstruction-v1.** It is kept separately
(`experiments/scripts/matched_info_pairs.py`, `experiments/reconstruction_v1/`,
`results/matched_info_pairs_*`). It does not reproduce the manuscript, and it
is not used to change it. Its differences are recorded as evidence only, not
as corrections.

ables 6", the paper title, `npmle`, `early_stopped`, `regret_calibrat`, `wtm` | one related file: the project's slide deck (no D.3 code, outputs or numbers) |

## Conclusion

The original D.3 experiment is not recoverable from any source available.
Following the authors' instruction to resolve D.3 by recovering the original
or adopting the reconstruction, the manuscript now reports `reconstruction-v1`
(`experiments/scripts/matched_info_pairs.py`) with all numbers and
interpretations updated together (`paper/reconciliation/README.md`, section
03). The 2026-09-30 values are recorded as `superseded` and unsupported in
`release/paper_artifacts.yaml`; they are not claimed as reproduced.

## What would identify the original

The seed list 1001–1005 does not fix the datasets. Matching the original
needs:

- the per-dataset simulation seed derivation (reconstruction-v1 uses
  `int(config_id("pair_<config_id>", T, beta, N, tau), 16) + seed mod 2^32`);
- which pair member the shared-law data are drawn from (reconstruction-v1
  uses the lower member), or whether the path-cell multinomial is sampled
  directly;
- the bootstrap RNG stream (reconstruction-v1 uses
  `default_rng(sim_seed + 2^31 - 1)` with 40 multinomial draws);
- the pair definitions (reconstruction-v1 reuses `extreme_pair_exact.py`'s
  supports on the 2,001-point grid).

A difference in any one of these changes the retained trials and every table
cell, though not the procedures' definitions.
